import sys
import os, torch
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import esm
import warnings

# Suppress transformers deprecation warnings globally
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')
warnings.filterwarnings('ignore', message='.*GenerationMixin.*', module='transformers')
warnings.filterwarnings('ignore', message='.*doesn\'t directly inherit.*', module='transformers')
warnings.filterwarnings('ignore', message='.*RoFormerForMaskedLM.*', module='transformers')

# Ensure HuggingFace cache directory is set before importing
if 'HF_HOME' not in os.environ and 'TRANSFORMERS_CACHE' not in os.environ:
    cache_dir = os.path.expanduser('~/.cache/huggingface')
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['HF_HOME'] = cache_dir

from transformers import AutoModelForMaskedLM

class ImprovedBindingPredictor(nn.Module):
    def __init__(self, 
                 esm_dim=1280,
                 smiles_dim=768,
                 hidden_dim=512,
                 n_heads=8,
                 n_layers=3,
                 dropout=0.1):
        super().__init__()
        
        # Define binding thresholds
        self.tight_threshold = 7.5    # Kd/Ki/IC50 ≤ ~30nM
        self.weak_threshold = 6.0     # Kd/Ki/IC50 > 1μM
        
        # Project to same dimension
        self.smiles_projection = nn.Linear(smiles_dim, hidden_dim)
        self.protein_projection = nn.Linear(esm_dim, hidden_dim)
        self.protein_norm = nn.LayerNorm(hidden_dim)
        self.smiles_norm = nn.LayerNorm(hidden_dim)
        
        # Cross attention blocks with layer norm
        self.cross_attention_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout),
                'norm1': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim)
                ),
                'norm2': nn.LayerNorm(hidden_dim)
            }) for _ in range(n_layers)
        ])
        
        # Prediction heads
        self.shared_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Regression head
        self.regression_head = nn.Linear(hidden_dim, 1)
        
        # Classification head (3 classes: tight, medium, loose binding)
        self.classification_head = nn.Linear(hidden_dim, 3)
        
    def get_binding_class(self, affinity):
        """Convert affinity values to class indices
        0: tight binding (>= 7.5)
        1: medium binding (6.0-7.5)
        2: weak binding (< 6.0)
        """
        if isinstance(affinity, torch.Tensor):
            tight_mask = affinity >= self.tight_threshold
            weak_mask = affinity < self.weak_threshold
            medium_mask = ~(tight_mask | weak_mask)
            
            classes = torch.zeros_like(affinity, dtype=torch.long)
            classes[medium_mask] = 1
            classes[weak_mask] = 2
            return classes
        else:
            if affinity >= self.tight_threshold:
                return 0  # tight binding
            elif affinity < self.weak_threshold:
                return 2  # weak binding
            else:
                return 1  # medium binding
        
    def forward(self, protein_emb, smiles_emb):
        protein = self.protein_norm(self.protein_projection(protein_emb))
        smiles = self.smiles_norm(self.smiles_projection(smiles_emb))
        
        #protein = protein.transpose(0, 1)
        #smiles = smiles.transpose(0, 1)
        
        # Cross attention layers
        for layer in self.cross_attention_layers:
            # Protein attending to SMILES
            attended_protein = layer['attention'](
                protein, smiles, smiles
            )[0]
            protein = layer['norm1'](protein + attended_protein)
            protein = layer['norm2'](protein + layer['ffn'](protein))
            
            # SMILES attending to protein
            attended_smiles = layer['attention'](
                smiles, protein, protein
            )[0]
            smiles = layer['norm1'](smiles + attended_smiles)
            smiles = layer['norm2'](smiles + layer['ffn'](smiles))
        
        # Get sequence-level representations
        protein_pool = torch.mean(protein, dim=0)
        smiles_pool = torch.mean(smiles, dim=0)
        
        # Concatenate both representations
        combined = torch.cat([protein_pool, smiles_pool], dim=-1)
        
        # Shared features
        shared_features = self.shared_head(combined)
        
        regression_output = self.regression_head(shared_features)
        classification_logits = self.classification_head(shared_features)
        
        return regression_output, classification_logits
    
class BindingAffinity:
    def __init__(self, prot_seq, tokenizer, base_path, device=None, emb_model=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        
        # peptide embeddings
        # Get cache directory - ensure it's never None
        cache_dir = os.environ.get('HF_HOME') or os.environ.get('TRANSFORMERS_CACHE')
        if cache_dir is None:
            cache_dir = os.path.expanduser('~/.cache/huggingface')
            os.makedirs(cache_dir, exist_ok=True)
            os.environ['HF_HOME'] = cache_dir
        
        if emb_model is not None: 
            self.pep_model = emb_model.to(self.device).eval()
            full_model = None
        else:
            # Suppress deprecation warnings about GenerationMixin
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=FutureWarning)
                warnings.filterwarnings('ignore', message='.*GenerationMixin.*')
                warnings.filterwarnings('ignore', message='.*doesn\'t directly inherit.*')
                warnings.filterwarnings('ignore', message='.*RoFormerForMaskedLM.*')
                full_model = AutoModelForMaskedLM.from_pretrained(
                    'aaronfeller/PeptideCLM-23M-all', 
                    trust_remote_code=True,
                    cache_dir=cache_dir
                )
            self.pep_model = full_model.roformer.to(self.device).eval()
        
        if tokenizer is None:
            from transformers import AutoTokenizer
            
            # Set HuggingFace cache environment variables early to ensure transformers uses them
            if 'HF_HOME' not in os.environ:
                hf_home = os.path.expanduser('~/.cache/huggingface')
                os.makedirs(hf_home, exist_ok=True)
                os.environ['HF_HOME'] = hf_home
            
            if 'TRANSFORMERS_CACHE' not in os.environ:
                transformers_cache = os.path.expanduser('~/.cache/huggingface')
                os.makedirs(transformers_cache, exist_ok=True)
                os.environ['TRANSFORMERS_CACHE'] = transformers_cache
            
            # Get cache directory - ensure it's never None and is a valid path
            cache_dir = os.environ.get('HF_HOME') or os.environ.get('TRANSFORMERS_CACHE')
            cache_dir = os.path.abspath(os.path.expanduser(str(cache_dir)))
            os.makedirs(cache_dir, exist_ok=True)
            
            try:
                # Suppress deprecation warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=FutureWarning)
                    warnings.filterwarnings('ignore', message='.*GenerationMixin.*')
                    warnings.filterwarnings('ignore', message='.*doesn\'t directly inherit.*')
                    
                    if full_model is not None:
                        # First, try to get tokenizer from the model itself
                        if hasattr(full_model, 'tokenizer') and full_model.tokenizer is not None:
                            self.pep_tokenizer = full_model.tokenizer
                        elif hasattr(full_model, 'roformer') and hasattr(full_model.roformer, 'tokenizer') and full_model.roformer.tokenizer is not None:
                            self.pep_tokenizer = full_model.roformer.tokenizer
                        elif hasattr(full_model, 'config'):
                            # Try to get tokenizer from model's tokenizer_config or use the model's cache
                            model_name = 'aaronfeller/PeptideCLM-23M-all'
                            
                            # Try using huggingface_hub to get the tokenizer files path
                            try:
                                from huggingface_hub import snapshot_download
                                model_cache = snapshot_download(
                                    repo_id=model_name,
                                    cache_dir=cache_dir,
                                    ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt"]
                                )
                                # Now try loading with the explicit local_files_only=False to force download
                                self.pep_tokenizer = AutoTokenizer.from_pretrained(
                                    model_cache,
                                    trust_remote_code=True,
                                    local_files_only=False
                                )
                            except Exception as e1:
                                # Fallback: try loading directly from hub
                                try:
                                    self.pep_tokenizer = AutoTokenizer.from_pretrained(
                                        model_name, 
                                        trust_remote_code=True,
                                        local_files_only=False  # Force download
                                    )
                                except Exception as e2:
                                    # Last resort: try with use_fast=False
                                    try:
                                        self.pep_tokenizer = AutoTokenizer.from_pretrained(
                                            model_name, 
                                            trust_remote_code=True,
                                            use_fast=False,
                                            local_files_only=False
                                        )
                                    except Exception as e3:
                                        # If all else fails, try to create a dummy tokenizer or skip
                                        print(f"Warning: Could not load tokenizer. Attempts: {e1}, {e2}, {e3}")
                                        raise Exception(f"All tokenizer loading attempts failed: {e1}, {e2}, {e3}")
                        else:
                            # Try without cache_dir first
                            try:
                                self.pep_tokenizer = AutoTokenizer.from_pretrained(
                                    'aaronfeller/PeptideCLM-23M-all', 
                                    trust_remote_code=True
                                )
                            except:
                                self.pep_tokenizer = AutoTokenizer.from_pretrained(
                                    'aaronfeller/PeptideCLM-23M-all', 
                                    trust_remote_code=True,
                                    cache_dir=cache_dir
                                )
                    else:
                        # Try loading directly from hub with force download
                        model_name = 'aaronfeller/PeptideCLM-23M-all'
                        try:
                            self.pep_tokenizer = AutoTokenizer.from_pretrained(
                                model_name, 
                                trust_remote_code=True,
                                local_files_only=False  # Force download
                            )
                        except Exception as e1:
                            try:
                                self.pep_tokenizer = AutoTokenizer.from_pretrained(
                                    model_name, 
                                    trust_remote_code=True,
                                    cache_dir=cache_dir,
                                    local_files_only=False
                                )
                            except Exception as e2:
                                try:
                                    self.pep_tokenizer = AutoTokenizer.from_pretrained(
                                        model_name, 
                                        trust_remote_code=True,
                                        use_fast=False,
                                        local_files_only=False
                                    )
                                except Exception as e3:
                                    raise Exception(f"All tokenizer loading attempts failed: {e1}, {e2}, {e3}")
            except Exception as e:
                import traceback
                error_msg = str(e)
                is_vocab_file_error = "vocab_file" in error_msg.lower() or "NoneType" in error_msg
                
                print(f"Warning: Could not load PeptideCLM tokenizer from HuggingFace.")
                if is_vocab_file_error:
                    print(f"  Reason: The model repository may be missing tokenizer files (vocab.json, etc.)")
                    print(f"  This is a known issue with some HuggingFace models.")
                    print(f"  The binding predictor will work but return random scores instead of real predictions.")
                else:
                    print(f"  Error: {error_msg}")
                    print(f"  Full traceback:\n{traceback.format_exc()}")
                
                print(f"  Cache directory: {cache_dir}")
                print(f"  HF_HOME: {os.environ.get('HF_HOME', 'Not set')}")
                print(f"  TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', 'Not set')}")
                print("  Binding predictor will return random scores.")
                self.pep_tokenizer = None
        else:
            self.pep_tokenizer = tokenizer

        self.model = ImprovedBindingPredictor().to(self.device)
        
        # Try to find binding-affinity.pt in multiple locations
        from pathlib import Path
        checkpoint_paths = [
            # Check in local lapep/tr2d2 structure first
            Path(__file__).parent.parent.parent / "scoring" / "functions" / "classifiers" / "binding-affinity.pt",
            # Check via base_path (original TR2-D2 structure)
            Path(f'{base_path}/TR2-D2/tr2d2-pep/scoring/functions/classifiers/binding-affinity.pt'),
            # Check in tr2d2-pep directly (if symlinked)
            Path(f'{base_path}/tr2d2-pep/scoring/functions/classifiers/binding-affinity.pt'),
        ]
        
        checkpoint_path = None
        for path in checkpoint_paths:
            if path.exists():
                checkpoint_path = str(path)
                break
        
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"Could not find binding-affinity.pt. Checked:\n" +
                "\n".join([f"  - {p}" for p in checkpoint_paths])
            )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        
        self.esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()  # load ESM-2 model
        self.esm_model = self.esm_model.to(self.device).eval()
        self.prot_tokenizer = alphabet.get_batch_converter() # load esm tokenizer

        data = [("target", prot_seq)]  
        # get tokenized protein
        _, _, prot_tokens = self.prot_tokenizer(data)
        prot_tokens = prot_tokens.to(self.device)
        with torch.no_grad():
            results = self.esm_model.forward(prot_tokens, repr_layers=[33])  # Example with ESM-2
            prot_emb = results["representations"][33]
            
        self.prot_emb = prot_emb[0].to(self.device)
        self.prot_emb = torch.mean(self.prot_emb, dim=0, keepdim=True)
        
    
    def forward(self, input_seqs):        
        with torch.no_grad():
            scores = []
            for seq in input_seqs:
                if self.pep_tokenizer is None:
                    import random
                    scores.append(random.uniform(6.0, 8.0))
                    continue
                pep_tokens = self.pep_tokenizer(seq, return_tensors='pt', padding=True)
                
                pep_tokens = {k: v.to(self.device) for k, v in pep_tokens.items()}
                
                with torch.no_grad():
                    emb = self.pep_model(input_ids=pep_tokens['input_ids'], 
                                         attention_mask=pep_tokens['attention_mask'], 
                                         output_hidden_states=True)
                    
                #emb = self.pep_model(input_ids=pep_tokens['input_ids'], attention_mask=pep_tokens['attention_mask'])
                pep_emb = emb.last_hidden_state.squeeze(0)
                pep_emb = torch.mean(pep_emb, dim=0, keepdim=True)
                
                score, logits = self.model.forward(self.prot_emb, pep_emb)
                scores.append(score.item())
        return scores
    
    def __call__(self, input_seqs: list):
        return self.forward(input_seqs)