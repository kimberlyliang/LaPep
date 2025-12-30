"""
PepDFM (Discrete Flow Matching) Wrapper for LaPep

Loads MOG-DFM models from HuggingFace: ChatterjeeLab/MOG-DFM
"""

import os
import warnings
from pathlib import Path
from typing import Optional
import torch
import numpy as np

from .base_generator import BaseGenerator


def _try_load_pepdfm_model(checkpoint: dict, device: str):
    """
    Try to instantiate PepDFM model from checkpoint using the structure from example files.
    
    This attempts to load:
    1. CNNModel from models.peptide_models
    2. Flow matching solver and path components
    3. ESM tokenizer
    
    Returns:
        Dict with model components or None if cannot instantiate
    """
    components = {}
    
    # Try to add MOG-DFM repository to path if it exists
    # Check common locations for the repository
    import sys
    from pathlib import Path
    
    possible_repo_paths = [
        Path(__file__).parent.parent / "external" / "mog-dfm",
        Path(__file__).parent.parent / "mog-dfm",
        Path(__file__).parent.parent.parent / "MOG-DFM",
        Path.cwd() / "external" / "mog-dfm",
        Path.cwd() / "mog-dfm",
    ]
    
    repo_added = False
    for repo_path in possible_repo_paths:
        if repo_path.exists() and (repo_path / "models" / "peptide_models.py").exists():
            if str(repo_path) not in sys.path:
                sys.path.insert(0, str(repo_path))
                repo_added = True
                print(f"    ✓ Added MOG-DFM repository to path: {repo_path}")
                break
    
    # Strategy 1: Try loading CNNModel (the probability denoiser)
    try:
        from models.peptide_models import CNNModel
        print(f"    ✓ Found CNNModel class")
        
        vocab_size = 24  # 20 amino acids + 4 special tokens
        embed_dim = 512
        hidden_dim = 256
        
        # Try to get config from checkpoint
        config = checkpoint.get('config', {})
        if 'embed_dim' in config:
            embed_dim = config['embed_dim']
        if 'hidden_dim' in config:
            hidden_dim = config['hidden_dim']
        
        probability_denoiser = CNNModel(
            alphabet_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim
        )
        
        # Load state dict
        state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
        if isinstance(state_dict, dict):
            probability_denoiser.load_state_dict(state_dict, strict=False)
        else:
            probability_denoiser.load_state_dict(state_dict, strict=False)
        
        probability_denoiser = probability_denoiser.to(device)
        probability_denoiser.eval()
        components['probability_denoiser'] = probability_denoiser
        components['vocab_size'] = vocab_size
        print(f"    ✓ Loaded CNNModel (embed_dim={embed_dim}, hidden_dim={hidden_dim})")
        
    except ImportError as e:
        print(f"    ⚠ Could not import CNNModel: {e}")
        print(f"    Make sure PepDFM repository code is available (models/peptide_models.py)")
        return None
    except Exception as e:
        print(f"    ⚠ Error loading CNNModel: {e}")
        return None
    
    # Strategy 2: Try loading flow matching components
    try:
        from flow_matching.path import MixtureDiscreteProbPath
        from flow_matching.path.scheduler import PolynomialConvexScheduler
        from flow_matching.solver import MixtureDiscreteEulerSolver
        from flow_matching.utils import ModelWrapper
        print(f"    ✓ Found flow matching components")
        
        # Create path and scheduler
        scheduler = PolynomialConvexScheduler(n=2.0)
        path = MixtureDiscreteProbPath(scheduler=scheduler)
        
        # Wrap model for solver
        class WrappedModel(ModelWrapper):
            def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
                return torch.softmax(self.model(x, t), dim=-1)
        
        wrapped_model = WrappedModel(probability_denoiser)
        solver = MixtureDiscreteEulerSolver(
            model=wrapped_model,
            path=path,
            vocabulary_size=vocab_size
        )
        
        components['solver'] = solver
        components['path'] = path
        components['wrapped_model'] = wrapped_model
        print(f"    ✓ Created flow matching solver")
        
    except ImportError as e:
        print(f"    ⚠ Could not import flow matching components: {e}")
        print(f"    Make sure flow_matching package is installed")
        return None
    except Exception as e:
        print(f"    ⚠ Error setting up flow matching: {e}")
        return None
    
    # Strategy 3: Load ESM tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        components['tokenizer'] = tokenizer
        print(f"    ✓ Loaded ESM tokenizer")
    except Exception as e:
        print(f"    ⚠ Could not load ESM tokenizer: {e}")
        # Tokenizer is optional, continue without it
    
    return components if components else None


class DFMWrapper(BaseGenerator):
    """
    Wrapper for PepDFM (MOG-DFM) discrete flow matching model.
    
    Works with WT amino acid sequences (20 amino acids).
    """
    
    def __init__(self, model=None, tokenizer=None, device: str = 'cpu', checkpoint=None, state_dict=None, config=None, components=None):
        """
        Initialize DFM wrapper.
        
        Args:
            model: Loaded MOG-DFM model (legacy, use components instead)
            tokenizer: Optional tokenizer for amino acid sequences
            device: Device to run on
            checkpoint: Optional checkpoint dict (if model not instantiated)
            state_dict: Optional state dict (if model not instantiated)
            config: Optional config dict (if model not instantiated)
            components: Dict with PepDFM components (solver, probability_denoiser, tokenizer, etc.)
        """
        self.model = model  # Legacy support
        self.device = torch.device(device)
        self.checkpoint = checkpoint
        self.state_dict = state_dict
        self.config = config
        
        # New: PepDFM components
        if components:
            self.solver = components.get('solver')
            self.probability_denoiser = components.get('probability_denoiser')
            self.tokenizer = components.get('tokenizer', tokenizer)
            self.vocab_size = components.get('vocab_size', 24)
            self.path = components.get('path')
            self.wrapped_model = components.get('wrapped_model')
        else:
            self.solver = None
            self.probability_denoiser = None
            self.tokenizer = tokenizer
            self.vocab_size = 24
            self.path = None
            self.wrapped_model = None
        
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.model.eval()
        
        if self.probability_denoiser is not None:
            self.probability_denoiser = self.probability_denoiser.to(self.device)
            self.probability_denoiser.eval()
    
    def proposal_probability(self, x_prime: str, x: str, tau: int) -> float:
        """
        Compute proposal probability b_θ(x'|x,τ) from base generator.
        
        Args:
            x_prime: Candidate next state (WT amino acid sequence)
            x: Current state (WT amino acid sequence)
            tau: Time step index
            
        Returns:
            Probability of proposing x' from x at step tau
        """
        # Use PepDFM model if available
        if self.probability_denoiser is not None and self.tokenizer is not None:
            try:
                # Encode sequences
                x_encoded = self.tokenizer(x, return_tensors='pt', add_special_tokens=False)['input_ids'].to(self.device)
                x_prime_encoded = self.tokenizer(x_prime, return_tensors='pt', add_special_tokens=False)['input_ids'].to(self.device)
                
                # Convert tau (step index) to time t in [0, 1]
                # Assuming tau is in [0, num_steps], normalize to [0, 1]
                num_steps = 100  # Default, could be configurable
                t = torch.tensor([min(tau / num_steps, 1.0 - 1e-6)], device=self.device)
                
                # Use model to compute probability distribution at time t
                # This is a simplified approach - the full implementation would need
                # to compute the transition probability using the flow matching path
                with torch.no_grad():
                    # Get probability distribution from model
                    logits = self.probability_denoiser(x_encoded, t)
                    probs = torch.softmax(logits, dim=-1)
                    
                    # Compute probability of x_prime given x at time t
                    # This is approximate - a full implementation would use the path
                    # For now, return a simple heuristic based on sequence similarity
                    if x_encoded.shape[1] == x_prime_encoded.shape[1]:
                        # Same length - compute probability of the differences
                        diff_mask = (x_encoded != x_prime_encoded).float()
                        if diff_mask.sum() > 0:
                            # Use the probability of the changed tokens
                            prob = (probs * diff_mask.unsqueeze(-1)).sum().item()
                            return max(prob, 1e-6)  # Avoid zero probability
                
                # Fallback: return small probability for different sequences
                return 0.1
                
            except Exception as e:
                # Fallback to simple heuristic
                pass
        
        # Fallback: simple heuristic based on sequence similarity
        if x == x_prime:
            return 1.0
        
        # Count differences
        if len(x) == len(x_prime):
            differences = sum(1 for a, b in zip(x, x_prime) if a != b)
            if differences == 1:
                # Single mutation - more likely
                return 0.5
            elif differences <= 3:
                # Few mutations
                return 0.1
            else:
                # Many mutations - less likely
                return 0.01
        else:
            # Different lengths - less likely
            return 0.01
    
    def get_neighbors(self, x: str) -> list:
        """
        Get neighbor states (one-edit mutations) from current state.
        
        Args:
            x: Current state (WT amino acid sequence)
            
        Returns:
            List of neighbor sequences (one amino acid mutations)
        """
        if self.model is None:
            return []
        
        try:
            if hasattr(self.model, 'get_edit_neighbors'):
                return self.model.get_edit_neighbors(x)
            elif hasattr(self.model, 'get_neighbors'):
                return self.model.get_neighbors(x)
            else:
                # Fallback: generate simple one-edit mutations
                return self._generate_simple_neighbors(x)
        except Exception as e:
            print(f"Warning: Could not get neighbors: {e}")
            return self._generate_simple_neighbors(x)
    
    def _generate_simple_neighbors(self, x: str) -> list:
        """
        Generate simple one-edit neighbors (single amino acid mutations).
        
        This is a fallback if the model doesn't provide neighbor generation.
        """
        neighbors = []
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        
        # Single amino acid mutations
        for i in range(len(x)):
            for aa in amino_acids:
                if aa != x[i]:
                    neighbor = x[:i] + aa + x[i+1:]
                    neighbors.append(neighbor)
        
        return neighbors[:20]  # Limit to first 20 to avoid explosion
    
    def sample_initial_state(self) -> str:
        """
        Sample initial state from prior distribution using PepDFM solver.
        
        Returns:
            Initial WT amino acid sequence
        """
        # Use PepDFM solver if available
        if self.solver is not None:
            try:
                import numpy as np
                # Sample length
                length = np.random.randint(6, 20)
                n_samples = 1
                source_distribution = "uniform"
                
                # Create initial state from uniform prior
                if source_distribution == "uniform":
                    x_init = torch.randint(
                        low=4, high=self.vocab_size, 
                        size=(n_samples, length), 
                        device=self.device
                    )
                else:
                    x_init = (torch.zeros(size=(n_samples, length), device=self.device) + 3).long()
                
                # Add special tokens [CLS] (0) and [SEP] (2)
                zeros = torch.zeros((n_samples, 1), dtype=x_init.dtype, device=self.device)
                twos = torch.full((n_samples, 1), 2, dtype=x_init.dtype, device=self.device)
                x_init = torch.cat([zeros, x_init, twos], dim=1)
                
                # Sample using solver
                step_size = 1.0 / 100  # Default step size
                sol = self.solver.sample(
                    x_init=x_init,
                    step_size=step_size,
                    verbose=False,
                    time_grid=torch.tensor([0.0, 1.0-1e-3], device=self.device)
                )
                
                # Decode sequence
                if self.tokenizer is not None:
                    sol_list = sol.tolist()
                    sequence = self.tokenizer.decode(sol_list[0]).replace(' ', '')
                    # Remove [CLS] and [SEP] tokens (first 5 and last 5 chars)
                    if len(sequence) > 10:
                        sequence = sequence[5:-5]
                    return sequence
                else:
                    # Fallback: convert token IDs to amino acids manually
                    # This is a simplified conversion
                    return self._tokens_to_sequence(sol[0].cpu().numpy())
                    
            except Exception as e:
                print(f"Warning: Could not sample using PepDFM solver: {e}")
                # Fall back to simple random generation
                pass
        
        # Fallback: random sequence
        import numpy as np
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        length = np.random.randint(5, 20)
        return ''.join(np.random.choice(list(amino_acids), length))
    
    def _tokens_to_sequence(self, tokens):
        """Convert token IDs to amino acid sequence (simplified)."""
        # ESM tokenizer mapping (simplified)
        # Token IDs 4-23 typically map to amino acids
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        sequence = []
        for token_id in tokens:
            if 4 <= token_id < 24:
                idx = token_id - 4
                if idx < len(amino_acids):
                    sequence.append(amino_acids[idx])
        return ''.join(sequence)


def load_dfm_model(
    path: Optional[str] = None,
    model_name: Optional[str] = None,
    device: str = 'cpu'
) -> DFMWrapper:
    """
    Load PepDFM (MOG-DFM) model from HuggingFace or local path.
    
    Args:
        path: Optional local path to model checkpoint. If provided, loads from local path.
              If None and model_name provided, loads from HuggingFace.
              If both None, uses default HuggingFace model "ChatterjeeLab/MOG-DFM"
        model_name: HuggingFace model identifier (default: "ChatterjeeLab/MOG-DFM")
        device: Device to load model on
        
    Returns:
        DFMWrapper instance with loaded model
    """
    # Check if path is a local file or HuggingFace model name
    path_obj = None
    if path:
        path_obj = Path(path)
        # If not absolute, check relative to current directory and project root
        if not path_obj.is_absolute():
            # Try current directory first
            cwd_path = Path.cwd() / path
            project_root_path = Path(__file__).parent.parent / path
            
            if cwd_path.exists():
                path_obj = cwd_path
            elif project_root_path.exists():
                path_obj = project_root_path
            else:
                # Path doesn't exist, will try HuggingFace later
                path_obj = None
        elif not path_obj.exists():
            # Absolute path doesn't exist
            path_obj = None
    
    # If local path exists, load from it
    if path and path_obj and path_obj.exists():
        print(f"Loading PepDFM (MOG-DFM) model from local checkpoint...")
        print(f"  Path: {path_obj}")
        print(f"  Device: {device}")
        print(f"  File size: {path_obj.stat().st_size / 1e6:.2f} MB")
    
    # Set up cache directory for HuggingFace models
    cache_dir = os.environ.get('HF_HOME') or os.environ.get('TRANSFORMERS_CACHE') or os.environ.get('HF_HUB_CACHE')
    
    if cache_dir is None and os.path.exists('/scratch'):
        import getpass
        username = getpass.getuser()
        possible_dirs = [
            f'/scratch/pranamlab/{username}/.cache/huggingface',
            f'/scratch/{username}/.cache/huggingface',
            os.path.join(os.getcwd(), '.cache', 'huggingface'),
        ]
        for possible_dir in possible_dirs:
            try:
                os.makedirs(possible_dir, exist_ok=True)
                test_file = os.path.join(possible_dir, '.test_write')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                cache_dir = possible_dir
                os.environ['HF_HOME'] = cache_dir
                os.environ['TRANSFORMERS_CACHE'] = cache_dir
                print(f"  Using cache directory: {cache_dir}")
                break
            except (OSError, PermissionError):
                continue
    
    if cache_dir is None:
        cache_dir = os.path.expanduser('~/.cache/huggingface')
        os.makedirs(cache_dir, exist_ok=True)
        os.environ['HF_HOME'] = cache_dir
    
    # If local path exists, load from it
    if path and path_obj and path_obj.exists():
        print(f"  Loading from local checkpoint: {path_obj}")
        try:
            # Load to CPU first to avoid device mapping issues (e.g., CUDA on Mac)
            # Then move to the appropriate device if needed
            checkpoint = torch.load(str(path_obj), map_location='cpu', weights_only=False)
            print(f"  ✓ Checkpoint loaded successfully (loaded to CPU)")
            
            # Move checkpoint tensors to the appropriate device if needed
            if device != 'cpu':
                # Determine actual available device
                if device == 'cuda' and torch.cuda.is_available():
                    actual_device = 'cuda'
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    actual_device = 'mps'
                else:
                    actual_device = 'cpu'
                    print(f"  ⚠ Requested device '{device}' not available, using '{actual_device}'")
                
                # Recursively move tensors to device
                def move_to_device(obj, target_device):
                    if isinstance(obj, torch.Tensor):
                        return obj.to(target_device)
                    elif isinstance(obj, dict):
                        return {k: move_to_device(v, target_device) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return type(obj)(move_to_device(item, target_device) for item in obj)
                    else:
                        return obj
                
                if actual_device != 'cpu':
                    checkpoint = move_to_device(checkpoint, actual_device)
                    print(f"  ✓ Moved checkpoint to {actual_device}")
            
            # Inspect checkpoint structure
            if isinstance(checkpoint, dict):
                checkpoint_keys = list(checkpoint.keys())
                print(f"  Checkpoint keys: {checkpoint_keys[:10]}...")  # Show first 10 keys
                
                # Try to find model state dict
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    print(f"  Found 'state_dict' key")
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    print(f"  Found 'model_state_dict' key")
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                    print(f"  Found 'model' key")
                else:
                    # Assume the whole dict is the state dict
                    state_dict = checkpoint
                    print(f"  Using entire checkpoint as state_dict")
                
                # Check for config/hyperparameters
                config = checkpoint.get('config', checkpoint.get('hyperparameters', checkpoint.get('args', None)))
                if config:
                    print(f"  Found config/hyperparameters in checkpoint")
                
                # Try to instantiate PepDFM model from checkpoint
                components = _try_load_pepdfm_model(checkpoint, actual_device)
                
                if components is None:
                    print(f"  ⚠ Could not instantiate PepDFM model architecture")
                    print(f"  Storing checkpoint for later use (will use fallback methods)")
                    print(f"  To use full PepDFM functionality:")
                    print(f"    1. Install PepDFM dependencies (flow_matching, models.peptide_models)")
                    print(f"    2. Ensure checkpoint contains the model state_dict")
                    # Create a wrapper that stores the checkpoint
                    wrapper = DFMWrapper(model=None, device=actual_device)
                    wrapper.checkpoint = checkpoint
                    wrapper.state_dict = state_dict
                    wrapper.config = config
                    return wrapper
                else:
                    print(f"  ✓ PepDFM model architecture instantiated and weights loaded")
                    return DFMWrapper(components=components, device=actual_device)
            else:
                # Checkpoint is not a dict - might be state dict directly
                print(f"  Checkpoint is not a dict, treating as state_dict")
                components = _try_load_pepdfm_model({'state_dict': checkpoint}, actual_device)
                if components is None:
                    wrapper = DFMWrapper(model=None, device=actual_device)
                    wrapper.checkpoint = checkpoint
                    wrapper.state_dict = checkpoint
                    return wrapper
                return DFMWrapper(components=components, device=actual_device)
                
        except Exception as e:
            print(f"  ✗ Error loading local checkpoint: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    # Load from HuggingFace
    if model_name is None:
        model_name = "ChatterjeeLab/MOG-DFM"
    
    try:
        # Try loading from HuggingFace
        from huggingface_hub import snapshot_download, hf_hub_download
        
        print(f"  Downloading model from HuggingFace...")
        
        # Download the entire repository
        repo_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            ignore_patterns=["*.md", "*.txt"]  # Skip documentation files
        )
        
        print(f"  Model downloaded to: {repo_path}")
        
        # Look for checkpoint files
        repo_path_obj = Path(repo_path)
        checkpoint_files = list(repo_path_obj.glob("**/*.pt")) + list(repo_path_obj.glob("**/*.pth")) + list(repo_path_obj.glob("**/*.ckpt"))
        
        if not checkpoint_files:
            # Try loading using transformers or the model's own loading mechanism
            print(f"  No checkpoint files found, trying to load model directly...")
            
            # Check if there's a model loading script or config
            config_file = repo_path_obj / "config.json"
            if config_file.exists():
                import json
                with open(config_file, 'r') as f:
                    config = json.load(f)
                print(f"  Found config.json")
            
            # Try to import and load the model
            # MOG-DFM might have its own loading mechanism
            # This is a placeholder - you'll need to adapt based on actual MOG-DFM API
            try:
                # Option 1: Try loading as a transformers model
                from transformers import AutoModel, AutoConfig
                config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
                model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
                print(f"  ✓ Loaded model using transformers")
            except Exception as e1:
                print(f"  Warning: Could not load with transformers: {e1}")
                # Option 2: Try loading checkpoint directly
                if checkpoint_files:
                    checkpoint_path = checkpoint_files[0]
                    print(f"  Loading checkpoint from: {checkpoint_path}")
                    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                    # You'll need to instantiate the model architecture first
                    # This depends on MOG-DFM's actual implementation
                    model = None
                    print(f"  ⚠ Checkpoint loaded but model architecture needs to be instantiated")
                else:
                    model = None
                    print(f"  ⚠ Could not find model files. Model loading may need manual implementation.")
        else:
            # Load from checkpoint
            checkpoint_path = checkpoint_files[0]
            print(f"  Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            # Instantiate model (this depends on MOG-DFM's architecture)
            # You'll need to adapt this based on the actual MOG-DFM model class
            model = None
            print(f"  ⚠ Checkpoint loaded but model architecture needs to be instantiated")
            print(f"  Please check MOG-DFM repository for model class definition")
        
        if model is None:
            print(f"\n  ⚠ WARNING: Model loading not fully implemented.")
            print(f"  The MOG-DFM model architecture needs to be instantiated.")
            print(f"  Please refer to: https://huggingface.co/{model_name}")
            print(f"  Or provide the model class/loading code from MOG-DFM repository.")
            print(f"  Returning wrapper with None model (will use fallback methods).")
        
        return DFMWrapper(model=model, device=device)
        
    except ImportError:
        print(f"  ✗ Error: huggingface_hub not installed")
        print(f"  Install with: pip install huggingface_hub")
        raise
    except Exception as e:
        print(f"  ✗ Error loading model from HuggingFace: {e}")
        print(f"  Falling back to local path if provided...")
        
        # Try local path if provided
        if path and Path(path).exists():
            print(f"  Loading from local path: {path}")
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            # Still need model architecture
            model = None
            print(f"  ⚠ Local checkpoint loaded but model architecture needs to be instantiated")
            return DFMWrapper(model=model, device=device)
        else:
            print(f"  ✗ Could not load model. Please:")
            print(f"    1. Install: pip install huggingface_hub")
            print(f"    2. Authenticate: huggingface-cli login")
            print(f"    3. Accept license at: https://huggingface.co/{model_name}")
            print(f"    4. Or provide local model path")
            raise RuntimeError(f"Failed to load MOG-DFM model: {e}")
