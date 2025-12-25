import torch
import torch.nn as nn
from typing import Optional


class PreferenceNet(nn.Module):
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256,
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


def load_preference_net(path: str) -> PreferenceNet:
    checkpoint = torch.load(path, map_location='cpu')
    model = PreferenceNet(
        input_dim=checkpoint.get('input_dim', 768),
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
    return model
