"""
text encoder: frozen (starting with Qwen but could also do E5, BioGPT, SciBERT)

"""

import torch
import torch.nn as nn
from typing import Union, Optional
import numpy as np


class TextEncoder:
    """
    Frozen text encoder for encoding prompts.
    """
    
    def __init__(self, model_name: str = 'e5', model=None, device: Optional[str] = None):
        """ 
        Args:
            model_name: Name of encoder ('e5', 'biogpt', 'scibert', 'qwen', 'qwen2', 
                        'Qwen/Qwen3-Embedding-0.6B', etc.)
            model: Pre-loaded model (if None, will load from model_name)
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        if device and device != 'cpu' and not torch.cuda.is_available():
            print(f"Warning: CUDA not available, using CPU instead of {device}")
            self.device = 'cpu'
        else:
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model or self._load_model(self.model_name)
        # should set model to eval mode if it's a real PyTorch model
        if self.model is not None and hasattr(self.model, 'eval'):
            self.model.eval()  # freeze
        # move model to device if it's a real PyTorch model
        if self.model is not None and hasattr(self.model, 'to'):
            self.model = self.model.to(self.device)
        self.embedding_dim = self._get_embedding_dim()
    
    def encode(self, text: Union[str, list]) -> torch.Tensor:
        """
        Encode text prompt(s) to embeddings.
        
        Args:
            text: String or list of strings
            
        Returns:
            Embedding tensor of shape (batch_size, embedding_dim)
        """
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
                # Fallback: dummy embeddings
                batch_size = len(text)
                embeddings = torch.randn(batch_size, self.embedding_dim, device=self.device)
        
        return embeddings
    
    def _encode_qwen(self, text: list) -> torch.Tensor:
        """Encode using Qwen model (including embedding models)."""
        try:
            from transformers import AutoTokenizer, AutoModel
            
            model_id = self._get_qwen_model_id()
            
            # Check if this is a dedicated embedding model
            is_embedding_model = 'embedding' in model_id.lower() or 'embed' in self.model_name.lower()
            
            if not hasattr(self, '_qwen_tokenizer'):
                print(f"Loading Qwen tokenizer for {model_id}...")
                self._qwen_tokenizer = AutoTokenizer.from_pretrained(model_id)
                if self.model is None or isinstance(self.model, nn.Module):
                    if not torch.cuda.is_available() or self.device == 'cpu':
                        load_device = 'cpu'
                    else:
                        load_device = self.device
                    
                    print(f"Loading Qwen model {model_id} (this may take a minute on first run)...")
                    self.model = AutoModel.from_pretrained(model_id)
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
            
            # Tokenize
            inputs = self._qwen_tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            # Only move to CUDA if available, otherwise keep on CPU
            if self.device != 'cpu' and torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                # Ensure on CPU
                inputs = {k: v.to('cpu') for k, v in inputs.items()}
                self.device = 'cpu'
            
            # Get embeddings
            if is_embedding_model:
                # For embedding models, use the model's forward pass
                # Qwen embedding models typically return embeddings directly
                outputs = self.model(**inputs)
                
                # Try different ways to extract embeddings
                if hasattr(outputs, 'last_hidden_state'):
                    # Mean pooling over sequence length
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                elif hasattr(outputs, 'pooler_output'):
                    # Use pooler output if available
                    embeddings = outputs.pooler_output
                elif hasattr(outputs, 'embeddings'):
                    # Direct embeddings
                    embeddings = outputs.embeddings
                else:
                    # Fallback: mean pooling of first output
                    if isinstance(outputs, tuple):
                        embeddings = outputs[0].mean(dim=1)
                    else:
                        # Try to get last_hidden_state from model attributes
                        embeddings = outputs.last_hidden_state.mean(dim=1)
                
                # Update embedding dimension if we got actual embeddings
                if embeddings.shape[-1] != self.embedding_dim:
                    self.embedding_dim = embeddings.shape[-1]
            else:
                # For regular Qwen models, use mean pooling of last hidden states
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
                # Update embedding dimension if we got actual embeddings
                if embeddings.shape[-1] != self.embedding_dim:
                    self.embedding_dim = embeddings.shape[-1]
            
            return embeddings
            
        except ImportError:
            print("Warning: transformers not installed. Install with: pip install transformers")
            return torch.randn(len(text), self.embedding_dim, device=self.device)
        except Exception as e:
            print(f"Error encoding with Qwen: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: always use CPU for dummy embeddings
            return torch.randn(len(text), self.embedding_dim, device='cpu')
    
    def _get_qwen_model_id(self) -> str:
        """Get Qwen model ID based on model_name."""
        # Check for Qwen3 embedding models first
        if 'qwen3' in self.model_name.lower() or 'embedding' in self.model_name.lower():
            if '0.6b' in self.model_name.lower() or '0.6B' in self.model_name.lower():
                return "Qwen/Qwen3-Embedding-0.6B"
            elif 'embedding' in self.model_name.lower():
                # Default to 0.6B if embedding is specified
                return "Qwen/Qwen3-Embedding-0.6B"
        
        # Check for full model path (e.g., "Qwen/Qwen3-Embedding-0.6B")
        if '/' in self.model_name and 'Qwen' in self.model_name:
            return self.model_name  # Use as-is if it's a full path
        
        # Qwen2.5 models
        if 'qwen2' in self.model_name or 'qwen-2' in self.model_name:
            if '0.5b' in self.model_name or '0.5B' in self.model_name:
                return "Qwen/Qwen2.5-0.5B"
            elif '1.5b' in self.model_name or '1.5B' in self.model_name:
                return "Qwen/Qwen2.5-1.5B"
            elif '3b' in self.model_name or '3B' in self.model_name:
                return "Qwen/Qwen2.5-3B"
            else:
                return "Qwen/Qwen2.5-0.5B"  # Default
        else:
            # Original Qwen models
            if '0.5b' in self.model_name or '0.5B' in self.model_name:
                return "Qwen/Qwen-0.5B"
            elif '1.5b' in self.model_name or '1.5B' in self.model_name:
                return "Qwen/Qwen-1.5B"
            else:
                return "Qwen/Qwen-0.5B"  # Default
    
    def _encode_e5(self, text: list) -> torch.Tensor:
        """Encode using E5 model (placeholder)."""
        # In practice, would use sentence-transformers or similar
        batch_size = len(text)
        return torch.randn(batch_size, 768, device=self.device)
    
    def _encode_biogpt(self, text: list) -> torch.Tensor:
        """Encode using BioGPT (placeholder)."""
        batch_size = len(text)
        return torch.randn(batch_size, 1024, device=self.device)
    
    def _encode_scibert(self, text: list) -> torch.Tensor:
        """Encode using SciBERT (placeholder)."""
        batch_size = len(text)
        return torch.randn(batch_size, 768, device=self.device)
    
    def _load_model(self, model_name: str):
        """Load model based on name."""
        if model_name.startswith('qwen'):
            # Qwen models are loaded lazily in _encode_qwen
            return nn.Module()  # Placeholder
        else:
            # Other models would be loaded here
            return nn.Module()  # Placeholder
    
    def _get_embedding_dim(self) -> int:
        """Get embedding dimension for the model."""
        model_name_lower = self.model_name.lower()
        
        if 'qwen3' in model_name_lower and 'embedding' in model_name_lower:
            # Qwen3-Embedding-0.6B has 1024 dimensions
            if '0.6b' in model_name_lower or '0.6B' in model_name_lower:
                return 1024
            else:
                return 1024  # Default for Qwen3 embedding models
        elif model_name_lower.startswith('qwen'):
            # Qwen2.5-0.5B has 512, Qwen2.5-1.5B has 1536, etc.
            if '0.5b' in model_name_lower or '0.5B' in model_name_lower:
                return 512
            elif '1.5b' in model_name_lower or '1.5B' in model_name_lower:
                return 1536
            elif '3b' in model_name_lower or '3B' in model_name_lower:
                return 2048
            else:
                return 512  # Default
        elif self.model_name == 'e5':
            return 768
        elif self.model_name == 'biogpt':
            return 1024
        elif self.model_name == 'scibert':
            return 768
        else:
            return 768  # Default
    
    def to(self, device):
        """Move model to device."""
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        if hasattr(self, '_qwen_tokenizer'):
            # Tokenizer doesn't need to move, but inputs will be on device
            pass
        return self


def load_text_encoder(model_name: str, device: Optional[str] = None) -> TextEncoder:
    """
    Load frozen text encoder.
    
    Args:
        model_name: Name of encoder to load ('qwen', 'qwen2', 'qwen2-0.5b', 'e5', etc.)
        device: Device to load on ('cuda' or 'cpu')
        
    Returns:
        TextEncoder instance
        
    Examples:
        >>> encoder = load_text_encoder('Qwen/Qwen3-Embedding-0.6B')
        >>> encoder = load_text_encoder('qwen2-0.5b')
        >>> encoder = load_text_encoder('qwen')
        >>> encoder = load_text_encoder('e5')
    """
    return TextEncoder(model_name=model_name, device=device)
