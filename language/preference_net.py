import torch
import torch.nn as nn
from typing import Optional


class PreferenceNet(nn.Module):
    """
    Preference Network g_ψ: Maps text embeddings to preference parameters η.
    
    Architecture: MLP with 2 hidden layers
    - Input: Text embeddings e (from frozen E_text)
    - Output: Preference parameters η (used in preference functional G_η)
    
    Dimension choices:
    - input_dim: Determined by text encoder (1024 for Qwen3-0.6B, 768 for BERT-like models)
    - hidden_dim: 256 - Standard size for MLPs, balances capacity vs. overfitting
    - output_dim: 64 - Should be >= num_predictors for linear mode, can be larger for nonlinear
    - num_predictors: Number of predictor functions (typically 3: binding, toxicity, halflife)
    
    These are reasonable defaults but can be tuned based on:
    - Task complexity (more complex prompts may need larger hidden_dim)
    - Number of predictors (output_dim should accommodate predictor space)
    - Available compute (larger networks train slower)
    """
    
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256,
                 output_dim: int = 64, num_predictors: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_predictors = num_predictors
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        if output_dim < num_predictors:
            self.linear_projection = nn.Linear(output_dim, num_predictors)
        else:
            self.linear_projection = None
    
    def forward(self, text_embedding: torch.Tensor) -> torch.Tensor:
        return self.network(text_embedding)
    
    def forward_linear(self, text_embedding: torch.Tensor) -> torch.Tensor:
        eta = self.forward(text_embedding)
        if self.linear_projection is not None:
            eta = self.linear_projection(eta)
        return eta


def load_preference_net(path: str, device: Optional[str] = None) -> PreferenceNet:
    load_device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(path, map_location=load_device)
    model = PreferenceNet(
        input_dim=checkpoint.get('input_dim', 1024),
        hidden_dim=checkpoint.get('hidden_dim', 256),
        output_dim=checkpoint.get('output_dim', 64),
        num_predictors=checkpoint.get('num_predictors', 3)
    )
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Move to device if specified
    if device and device != 'cpu':
        if torch.cuda.is_available():
            model = model.to(device)
            print(f"Preference network moved to {device}")
        else:
            print(f"Warning: CUDA not available, keeping preference network on CPU")
    else:
        print(f"Preference network on CPU")
    
    return model
