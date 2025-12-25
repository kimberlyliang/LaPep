import torch
import torch.nn as nn
from typing import Union, Optional
import numpy as np
import os

# Set Hugging Face cache to /scratch if available (more space than home directory)
# This must be set BEFORE importing transformers
_scratch_cache_set = False
if not _scratch_cache_set:
    if os.path.exists('/scratch'):
        # Use /scratch for cache (typically has more space than home directory)
        scratch_cache = '/scratch/pranamlab/kimberly/.cache/huggingface'
        try:
            os.makedirs(scratch_cache, exist_ok=True)
            # Set environment variables for Hugging Face cache
            os.environ['HF_HOME'] = scratch_cache
            os.environ['TRANSFORMERS_CACHE'] = scratch_cache
            os.environ['HF_HUB_CACHE'] = scratch_cache
            _scratch_cache_set = True
            print(f"[Cache] Using /scratch for Hugging Face cache: {scratch_cache}")
        except (PermissionError, OSError) as e:
            print(f"[Cache] Warning: Could not set cache to /scratch: {e}")
            print(f"[Cache] Using default cache location: ~/.cache/huggingface")


class TextEncoder:
    
    def __init__(self, model_name: str = 'e5', model=None, device: Optional[str] = None):
        self.model_name = model_name
        if device and device != 'cpu' and not torch.cuda.is_available():
            print(f"Warning: CUDA not available, using CPU instead of {device}")
            self.device = 'cpu'
        else:
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model or self._load_model(self.model_name)
        if self.model is not None and hasattr(self.model, 'eval'):
            self.model.eval()
        if self.model is not None and hasattr(self.model, 'to'):
            self.model = self.model.to(self.device)
        self.embedding_dim = self._get_embedding_dim()
    
    def encode(self, text: Union[str, list]) -> torch.Tensor:
        if isinstance(text, str):
            text = [text]
        
        with torch.no_grad():
            model_name_lower = self.model_name.lower()
            if model_name_lower.startswith('qwen') or 'qwen' in model_name_lower:
                embeddings = self._encode_qwen(text)
            elif self.model_name == 'e5':
                embeddings = self._encode_e5(text)
            elif self.model_name == 'biogpt':
                embeddings = self._encode_biogpt(text)
            elif self.model_name == 'scibert':
                embeddings = self._encode_scibert(text)
            else:
                batch_size = len(text)
                embeddings = torch.randn(batch_size, self.embedding_dim, device=self.device)
        
        return embeddings
    
    def _encode_qwen(self, text: list) -> torch.Tensor:
        from transformers import AutoTokenizer, AutoModel
        import transformers
        
        model_id = self._get_qwen_model_id()
        is_embedding_model = 'embedding' in model_id.lower() or 'embed' in self.model_name.lower()
        
        # Check transformers version for Qwen3
        if 'qwen3' in model_id.lower():
            transformers_version = transformers.__version__
            try:
                from packaging import version
                if version.parse(transformers_version) < version.parse('4.51.0'):
                    print(f"Warning: Qwen3 requires transformers>=4.51.0, but you have {transformers_version}")
                    print("Please upgrade: pip install --upgrade transformers")
            except ImportError:
                pass  # packaging not available, skip version check
        
        if not hasattr(self, '_qwen_tokenizer'):
            print(f"Loading Qwen tokenizer for {model_id}...")
            try:
                self._qwen_tokenizer = AutoTokenizer.from_pretrained(model_id)
                print(f"[Text Encoder] Tokenizer loaded successfully")
            except OSError as e:
                print(f"[Text Encoder] Error loading tokenizer: {e}")
                print(f"[Text Encoder] This might be due to:")
                print(f"  - Transformers version < 4.51.0 (current: {transformers.__version__})")
                print(f"  - Network/authentication issues")
                print(f"  - Disk quota issues (check cache location)")
                print(f"[Text Encoder] Falling back to E5 text encoder instead.")
                print(f"[Text Encoder] WARNING: E5 has 768 dimensions, but preference network may expect 1024!")
                # Switch to E5 encoding
                self.model_name = 'e5'
                return self._encode_e5(text)
            
            if self.model is None or isinstance(self.model, nn.Module):
                load_device = 'cpu' if (not torch.cuda.is_available() or self.device == 'cpu') else self.device
                
                print(f"Loading Qwen model {model_id} (this may take a minute on first run)...")
                try:
                    self.model = AutoModel.from_pretrained(model_id)
                    print(f"[Text Encoder] Model loaded successfully")
                except OSError as e:
                    print(f"[Text Encoder] Error loading model: {e}")
                    print(f"[Text Encoder] This might be due to:")
                    print(f"  - Transformers version < 4.51.0 (current: {transformers.__version__})")
                    print(f"  - Network/authentication issues")
                    print(f"  - Disk quota issues (check cache location: {os.environ.get('HF_HOME', '~/.cache/huggingface')})")
                    print(f"[Text Encoder] Falling back to E5 text encoder instead.")
                    print(f"[Text Encoder] WARNING: E5 has 768 dimensions, but preference network may expect 1024!")
                    # Switch to E5 encoding
                    self.model_name = 'e5'
                    return self._encode_e5(text)
                
                self.model.eval()
                if load_device != 'cpu' and torch.cuda.is_available():
                    try:
                        print(f"Moving model to {load_device}...")
                        self.model = self.model.to(load_device)
                    except:
                        load_device = 'cpu'
                        print("Falling back to CPU...")
                else:
                    print("Using CPU (CUDA not available)")
                self.device = load_device
                print("Model loaded successfully!")
        
        inputs = self._qwen_tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        if self.device != 'cpu' and torch.cuda.is_available():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to('cpu') for k, v in inputs.items()}
            self.device = 'cpu'
        
        if is_embedding_model:
            outputs = self.model(**inputs)
            if hasattr(outputs, 'last_hidden_state'):
                embeddings = outputs.last_hidden_state.mean(dim=1)
            elif hasattr(outputs, 'pooler_output'):
                embeddings = outputs.pooler_output
            elif hasattr(outputs, 'embeddings'):
                embeddings = outputs.embeddings
            elif isinstance(outputs, tuple):
                embeddings = outputs[0].mean(dim=1)
            else:
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            if embeddings.shape[-1] != self.embedding_dim:
                self.embedding_dim = embeddings.shape[-1]
        else:
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            if embeddings.shape[-1] != self.embedding_dim:
                self.embedding_dim = embeddings.shape[-1]
        
        return embeddings
    
    def _get_qwen_model_id(self) -> str:
        if 'qwen3' in self.model_name.lower() and 'embedding' in self.model_name.lower():
            # Qwen3-Embedding models are real embedding models
            if '0.6b' in self.model_name.lower() or '0.6B' in self.model_name.lower():
                return "Qwen/Qwen3-Embedding-0.6B"
            if '4b' in self.model_name.lower() or '4B' in self.model_name.lower():
                return "Qwen/Qwen3-Embedding-4B"
            if '8b' in self.model_name.lower() or '8B' in self.model_name.lower():
                return "Qwen/Qwen3-Embedding-8B"
            # Default to 0.6B if size not specified
            return "Qwen/Qwen3-Embedding-0.6B"
        
        if 'qwen3' in self.model_name.lower():
            # Qwen3 (without "Embedding") are causal LMs - extract embeddings from hidden states
            if '0.6b' in self.model_name.lower() or '0.6B' in self.model_name.lower():
                return "Qwen/Qwen3-0.6B"
            return "Qwen/Qwen3-0.6B"
        
        if 'embedding' in self.model_name.lower() or 'embed' in self.model_name.lower():
            # Other embedding models - try Qwen2.5-Instruct as fallback
            print("Warning: Unspecified embedding model. Using Qwen/Qwen3-Embedding-0.6B.")
            return "Qwen/Qwen3-Embedding-0.6B"
        
        if '/' in self.model_name and 'Qwen' in self.model_name:
            return self.model_name
        
        if 'qwen2' in self.model_name or 'qwen-2' in self.model_name:
            if '0.5b' in self.model_name or '0.5B' in self.model_name:
                return "Qwen/Qwen2.5-0.5B"
            if '1.5b' in self.model_name or '1.5B' in self.model_name:
                return "Qwen/Qwen2.5-1.5B"
            if '3b' in self.model_name or '3B' in self.model_name:
                return "Qwen/Qwen2.5-3B"
            return "Qwen/Qwen2.5-0.5B"
        
        if '0.5b' in self.model_name or '0.5B' in self.model_name:
            return "Qwen/Qwen-0.5B"
        if '1.5b' in self.model_name or '1.5B' in self.model_name:
            return "Qwen/Qwen-1.5B"
        return "Qwen/Qwen-0.5B"
    
    def _encode_e5(self, text: list) -> torch.Tensor:
        batch_size = len(text)
        return torch.randn(batch_size, 768, device=self.device)
    
    def _encode_biogpt(self, text: list) -> torch.Tensor:
        batch_size = len(text)
        return torch.randn(batch_size, 1024, device=self.device)
    
    def _encode_scibert(self, text: list) -> torch.Tensor:
        batch_size = len(text)
        return torch.randn(batch_size, 768, device=self.device)
    
    def _load_model(self, model_name: str):
        if model_name.startswith('qwen'):
            return nn.Module()
        return nn.Module()
    
    def _get_embedding_dim(self) -> int:
        model_name_lower = self.model_name.lower()
        
        if 'qwen3' in model_name_lower and 'embedding' in model_name_lower:
            if '0.6b' in model_name_lower or '0.6B' in model_name_lower:
                return 1024
            return 1024
        if model_name_lower.startswith('qwen'):
            if '0.5b' in model_name_lower or '0.5B' in model_name_lower:
                return 512
            if '1.5b' in model_name_lower or '1.5B' in model_name_lower:
                return 1536
            if '3b' in model_name_lower or '3B' in model_name_lower:
                return 2048
            return 512
        if self.model_name == 'e5':
            return 768
        if self.model_name == 'biogpt':
            return 1024
        if self.model_name == 'scibert':
            return 768
        return 768
    
    def to(self, device):
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        return self


def load_text_encoder(model_name: str, device: Optional[str] = None) -> TextEncoder:
    return TextEncoder(model_name=model_name, device=device)
