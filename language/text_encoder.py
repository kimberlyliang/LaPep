import torch
import torch.nn as nn
from typing import Union, Optional
import numpy as np
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

class TextEncoder:
    
    def __init__(self, model_name: str, model=None, tokenizer=None, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if model is None or tokenizer is None:
            self.model, self.tokenizer = self._load_model(self.model_name)
        else:
            self.model = model
            self.tokenizer = tokenizer
        if self.model is not None and hasattr(self.model, 'eval'):
            self.model.eval()
        if self.model is not None and hasattr(self.model, 'to'):
            self.model = self.model.to(self.device)
        self.embedding_dim = self._get_embedding_dim()
    
    def encode(self, text: Union[str, list]) -> torch.Tensor:
        if isinstance(text, str):
            text = [text]
        with torch.no_grad():
            embeddings = self._encode_qwen(text)
        return embeddings
    
    def _encode_qwen(self, text: list) -> torch.Tensor:
        all_embeddings = []
        for prompt in text:
            messages = [
                {"role": "user", "content": prompt}
            ]
            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            model_inputs = self.tokenizer([formatted_text], return_tensors="pt").to(self.device)
        
            outputs = self.model(**model_inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1] 
            
            # Mean pooling over sequence length (excluding padding)
            attention_mask = model_inputs['attention_mask']
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * attention_mask_expanded, dim=1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
            embedding = sum_hidden / sum_mask
            
            all_embeddings.append(embedding.squeeze(0))
        return torch.stack(all_embeddings)
    
    def _load_model(self, model_name: str):
        # Set up cache directory to avoid disk quota issues
        cache_dir = os.environ.get('HF_HOME') or os.environ.get('TRANSFORMERS_CACHE') or os.environ.get('HF_HUB_CACHE')
        
        # If no cache dir set and we're on a system with /scratch, use it
        if cache_dir is None:
            if os.path.exists('/scratch'):
                # Try to find a writable scratch directory
                import getpass
                username = getpass.getuser()
                
                # Try multiple possible scratch locations
                possible_dirs = [
                    f'/scratch/pranamlab/{username}/.cache/huggingface',  # Lab-specific
                    f'/scratch/{username}/.cache/huggingface',  # User-specific
                    os.path.join(os.getcwd(), '.cache', 'huggingface'),  # Current directory
                ]
                
                for possible_dir in possible_dirs:
                    try:
                        os.makedirs(possible_dir, exist_ok=True)
                        # Test if we can write
                        test_file = os.path.join(possible_dir, '.test_write')
                        with open(test_file, 'w') as f:
                            f.write('test')
                        os.remove(test_file)
                        cache_dir = possible_dir
                        os.environ['HF_HOME'] = cache_dir
                        os.environ['TRANSFORMERS_CACHE'] = cache_dir
                        print(f"[Text Encoder] Using cache directory: {cache_dir}")
                        break
                    except (OSError, PermissionError):
                        continue
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=cache_dir
        )
        return model, tokenizer
    
    def _get_embedding_dim(self) -> int:
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'hidden_size'):
                return self.model.config.hidden_size
            elif hasattr(self.model.config, 'd_model'):
                return self.model.config.d_model

        test_embedding = self.encode("test")
        return test_embedding.shape[-1]
        
    def to(self, device):
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        return self


def load_text_encoder(model_name: str, device: Optional[str] = None) -> TextEncoder:
    return TextEncoder(model_name=model_name, device=device)
