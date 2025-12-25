"""Wrapper for PepMDLM/Diffusion model from TR2-D2."""

import torch
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np

try:
    lapep_root = Path(__file__).parent.parent.resolve()
    tr2d2_paths = [
        lapep_root / "lapep" / "tr2d2",
    ]
    
    tr2d2_path = None
    for path in tr2d2_paths:
        path = path.resolve()
        if path.exists():
            diffusion_file = path / "diffusion.py"
            if diffusion_file.exists():
                tr2d2_path = path
                sys.path = [p for p in sys.path if 'tr2d2' not in p and 'TR2-D2' not in p]
                sys.path.insert(0, str(tr2d2_path))
                break
    
    if tr2d2_path is None:
        raise ImportError("TR2-D2 directory not found or missing diffusion.py")
    
    missing_deps = []
    try:
        import lightning
    except ImportError:
        missing_deps.append("pytorch-lightning")
    
    try:
        import timm
    except ImportError:
        missing_deps.append("timm")
    
    try:
        from SmilesPE.tokenizer import SPE_Tokenizer
    except ImportError:
        missing_deps.append("SmilesPE")
    
    if missing_deps:
        raise ImportError(f"Missing dependencies: {', '.join(missing_deps)}. Install with: pip install {' '.join(missing_deps)}")
    
    from diffusion import Diffusion
    from tokenizer.my_tokenizers import SMILES_SPE_Tokenizer
    TR2D2_AVAILABLE = True
except ImportError as e:
    if "Missing dependencies" not in str(e):
        import traceback
        traceback.print_exc()
    TR2D2_AVAILABLE = False
    Diffusion = None

from .base_generator import BaseGenerator


class PepMDLMWrapper(BaseGenerator):
    """
    Wrapper for PepMDLM/Diffusion model that implements BaseGenerator interface.
    """
    
    def __init__(self, model_path: str, config_path: Optional[str] = None, 
                 tokenizer_vocab: Optional[str] = None, 
                 tokenizer_splits: Optional[str] = None,
                 device: str = 'cpu'):
        """
        Initialize PepMDLM wrapper.
        
        Args:
            model_path: Path to peptune-pretrained.ckpt
            config_path: Path to peptune_config.yaml (optional, will try to find)
            tokenizer_vocab: Path to tokenizer vocab file
            tokenizer_splits: Path to tokenizer splits file
            device: Device to run on
        """
        super().__init__()
        self.model_path = model_path
        self.device = torch.device(device)
        self.model = None
        self.tokenizer = None
        
        lapep_root = Path(__file__).parent.parent.resolve()
        if tokenizer_vocab is None:
            tokenizer_vocab = lapep_root / "lapep" / "tr2d2" / "tokenizer" / "new_vocab.txt"
            if not tokenizer_vocab.exists():
                tokenizer_vocab = lapep_root / "tr2d2-pep" / "tokenizer" / "new_vocab.txt"
            if not tokenizer_vocab.exists():
                tokenizer_vocab = lapep_root.parent / "TR2-D2" / "tr2d2-pep" / "tokenizer" / "new_vocab.txt"
        
        if tokenizer_splits is None:
            tokenizer_splits = lapep_root / "lapep" / "tr2d2" / "tokenizer" / "new_splits.txt"
            if not tokenizer_splits.exists():
                tokenizer_splits = lapep_root / "tr2d2-pep" / "tokenizer" / "new_splits.txt"
            if not tokenizer_splits.exists():
                tokenizer_splits = lapep_root.parent / "TR2-D2" / "tr2d2-pep" / "tokenizer" / "new_splits.txt"
        
        self.tokenizer_vocab = str(tokenizer_vocab) if tokenizer_vocab and tokenizer_vocab.exists() else None
        self.tokenizer_splits = str(tokenizer_splits) if tokenizer_splits and tokenizer_splits.exists() else None
        
        if config_path is None:
            config_path = lapep_root / "lapep" / "tr2d2" / "configs" / "peptune_config.yaml"
            if not config_path.exists():
                config_path = lapep_root / "tr2d2-pep" / "configs" / "peptune_config.yaml"
            if not config_path.exists():
                config_path = lapep_root.parent / "TR2-D2" / "tr2d2-pep" / "configs" / "peptune_config.yaml"
        
        self.config_path = str(config_path) if config_path and Path(config_path).exists() else None
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """load diffusion model from checkpoint."""
        if not TR2D2_AVAILABLE:
            self.model = None
            return
        
        try:
            if self.tokenizer_vocab and self.tokenizer_splits:
                vocab_path = Path(self.tokenizer_vocab)
                splits_path = Path(self.tokenizer_splits)
                
                if vocab_path.exists() and splits_path.exists():
                    self.tokenizer = SMILES_SPE_Tokenizer(
                        str(vocab_path),
                        str(splits_path)
                    )
                else:
                    raise FileNotFoundError(f"Tokenizer files not found")
            else:
                raise ValueError("Tokenizer paths not provided")
            
            if self.config_path and Path(self.config_path).exists():
                from omegaconf import OmegaConf
                config = OmegaConf.load(self.config_path)
            else:
                from omegaconf import DictConfig
                config = DictConfig({
                    'noise': {'type': 'loglinear', 'sigma_min': 1e-4, 'sigma_max': 20, 'state_dependent': True},
                    'vocab': 'new_smiles',
                    'backbone': 'roformer',
                    'T': 0,
                    'sampling': {'predictor': 'ddpm_cache', 'steps': 128, 'seq_length': 100},
                    'time_conditioning': False,
                    'roformer': {'hidden_size': 768, 'n_layers': 8, 'n_heads': 8, 'max_position_embeddings': 1035}
                })
            
            if self.tokenizer is None:
                raise ValueError("Tokenizer is None")
            
            self.model = Diffusion(config=config, tokenizer=self.tokenizer, mode="eval", device=self.device)
            
            ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                state = ckpt["state_dict"]
            elif isinstance(ckpt, dict):
                state = ckpt
            else:
                raise ValueError("Checkpoint must be a dict")
            
            def strip_prefix(state_dict, prefixes=("module.", "model.")):
                for p in prefixes:
                    if all(k.startswith(p) for k in state_dict.keys()):
                        return {k[len(p):]: v for k, v in state_dict.items()}
                return state_dict
            
            state = strip_prefix(state)
            self.model.load_state_dict(state, strict=False)
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.model = None
    
    def sample_unconditioned(self, num_samples: int = 1, seq_length: int = 100, 
                            num_steps: int = 128, eps: float = 1e-5) -> List[str]:
        """sample peptides unconditionally from the model."""
        if self.model is None:
            return ["[MASK]" * 10] * num_samples
        
        try:
            self.model.eval()
            peptides = []
            
            batch_size = getattr(self.model.config.eval, 'perplexity_batch_size', 8)
            if batch_size is None or batch_size <= 0:
                batch_size = 8
            
            with torch.no_grad():
                total_batches = (num_samples + batch_size - 1) // batch_size
                for batch_start in range(0, num_samples, batch_size):
                    batch_end = min(batch_start + batch_size, num_samples)
                    batch_num = batch_end - batch_start
                    batch_idx = batch_start // batch_size + 1
                    
                    if num_samples > 1:
                        print(f"    Batch {batch_idx}/{total_batches}: Sampling {batch_num} peptides...")
                        import sys
                        sys.stdout.flush()
                    
                    original_batch_size = self.model.config.eval.perplexity_batch_size
                    self.model.config.eval.perplexity_batch_size = batch_num
                    
                    try:
                        samples = self.model._sample(
                            num_steps=num_steps,
                            eps=eps,
                            x_input=None
                        )
                        
                        if self.tokenizer:
                            batch_peptides = self.tokenizer.batch_decode(samples)
                        else:
                            batch_peptides = [str(sample.tolist()) for sample in samples]
                        
                        peptides.extend(batch_peptides[:batch_num])
                        
                        if num_samples > 1:
                            print(f"    Batch {batch_idx}/{total_batches}: Completed ({len(peptides)}/{num_samples} peptides)")
                            sys.stdout.flush()
                    finally:
                        self.model.config.eval.perplexity_batch_size = original_batch_size
                
                return peptides[:num_samples]
                
        except Exception as e:
            print(f"\nError sampling from PepMDLM: {e}")
            import traceback
            traceback.print_exc()
            return ["[MASK]" * 10] * num_samples
    
    def sample_initial_state(self) -> str:
        """sample initial fully masked state."""
        if self.model is None:
            return "[MASK]" * 10
        
        if self.tokenizer and hasattr(self.tokenizer, 'mask_token_id'):
            mask_id = self.tokenizer.mask_token_id
            seq_length = 100
            masked_seq = torch.full((1, seq_length), mask_id, dtype=torch.long, device=self.device)
            return self.tokenizer.decode(masked_seq[0])
        else:
            return "[MASK]" * 10
    
    def proposal_probability(self, x_prime: str, x: str, tau: int) -> float:
        """compute base proposal probability b_θ(x'|x,τ)."""
        if self.model is None or self.tokenizer is None:
            return 0.0
        
        try:
            num_steps = 128
            eps = 1e-5
            t = 1.0 - (tau / num_steps) * (1.0 - eps)
            t = max(eps, min(1.0, t))
            
            x_token_array = self.tokenizer._tokenize(x)
            x_tokens = self.tokenizer.encode(x_token_array)
            x_ids = x_tokens['input_ids'].to(self.device)
            attn_mask = x_tokens['attention_mask'].to(self.device)
            
            sigma_t, _ = self.model.noise(torch.tensor([[t]], device=self.device))
            logits = self.model.forward(x_ids, attn_mask, sigma_t)
            
            x_prime_token_array = self.tokenizer._tokenize(x_prime)
            x_prime_tokens = self.tokenizer.encode(x_prime_token_array)
            x_prime_ids = x_prime_tokens['input_ids'].to(self.device)
            
            log_prob = 0.0
            seq_len = min(x_ids.shape[1], x_prime_ids.shape[1], logits.shape[1])
            
            for pos in range(seq_len):
                x_token = x_ids[0, pos].item()
                x_prime_token = x_prime_ids[0, pos].item()
                
                if x_token == x_prime_token:
                    continue
                elif x_token == self.model.mask_index:
                    log_prob += logits[0, pos, x_prime_token].item()
                else:
                    return 0.0
            
            return np.exp(log_prob)
            
        except Exception as e:
            print(f"Error computing proposal probability: {e}")
            return 0.0
    
    def get_neighbors(self, x: str) -> List[str]:
        """get local edit neighborhood N(x) - single-token edits."""
        if self.model is None or self.tokenizer is None:
            return []
        
        try:
            neighbors = []
            
            # Tokenize x (tokenizer.encode expects token array, not string)
            x_token_array = self.tokenizer._tokenize(x)
            x_tokens = self.tokenizer.encode(x_token_array)
            x_ids = x_tokens['input_ids'].to(self.device)
            
            seq_len = x_ids.shape[1]
            mask_id = self.model.mask_index
            vocab_size = self.model.vocab_size
            max_neighbors = 50
            
            masked_positions = []
            unmasked_positions = []
            for pos in range(seq_len):
                token_id = x_ids[0, pos].item()
                if token_id == mask_id:
                    masked_positions.append(pos)
                else:
                    unmasked_positions.append(pos)
            
            if masked_positions and len(neighbors) < max_neighbors:
                t = torch.tensor([[0.5]], device=self.device)
                sigma_t, _ = self.model.noise(t)
                logits = self.model.forward(x_ids, x_tokens['attention_mask'].to(self.device), sigma_t)
                
                for pos in masked_positions[:5]:
                    pos_logits = logits[0, pos, :].cpu()
                    top_k = min(10, vocab_size)
                    top_tokens = torch.topk(pos_logits, top_k).indices
                    
                    for token_id in top_tokens[:5]:
                        if len(neighbors) >= max_neighbors:
                            break
                        neighbor_ids = x_ids.clone()
                        neighbor_ids[0, pos] = token_id
                        neighbor_str = self.tokenizer.decode(neighbor_ids[0])
                        if neighbor_str != x and neighbor_str not in neighbors:
                            neighbors.append(neighbor_str)
            
            if unmasked_positions and len(neighbors) < max_neighbors:
                for pos in unmasked_positions[:5]:
                    if len(neighbors) >= max_neighbors:
                        break
                    neighbor_ids = x_ids.clone()
                    neighbor_ids[0, pos] = mask_id
                    neighbor_str = self.tokenizer.decode(neighbor_ids[0])
                    if neighbor_str != x and neighbor_str not in neighbors:
                        neighbors.append(neighbor_str)
            
            return neighbors[:max_neighbors]
            
        except Exception as e:
            print(f"Error generating neighbors: {e}")
            return []


def load_peptune_generator(path: str, device: str = 'cpu') -> PepMDLMWrapper:
    """load peptune generator from checkpoint."""
    return PepMDLMWrapper(path, device=device)

