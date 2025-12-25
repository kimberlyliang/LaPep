"""
preference Network: g_ψ

Learnable module that maps text embeddings to preference functional parameters.
"""

import torch
import torch.nn as nn
from typing import Optional


class PreferenceNet(nn.Module):
    """
    Preference network g_ψ that maps text embeddings to preference parameters.
    
    Input: text embedding e(t) ∈ R^d
    Output: preference parameters η(t) that define G_η(t)
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        output_dim: int = 64,
        num_predictors: int = 3
    ):
        """
        Initialize preference network.
        
        Args:
            input_dim: text embeddings dimension
            hidden_dim: hidden layer dimension
            output_dim: output dimension (size of η)
            num_predictors: number of predictors (for linear case)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_predictors = num_predictors
        
        # MLP to map embeddings to preference parameters
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # output_dim should match num_predictors
        if output_dim < num_predictors:
            # projection for linear case
            self.linear_projection = nn.Linear(output_dim, num_predictors)
        else:
            self.linear_projection = None
    
    def forward(self, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Map text embedding to preference parameters.
        
        Args:
            text_embedding: Tensor of shape (batch_size, input_dim)
            
        Returns:
            Preference parameters η of shape (batch_size, output_dim)
        """
        eta = self.network(text_embedding)
        return eta
    
    def forward_linear(self, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Map to linear preference parameters (for ablation).
        
        Returns parameters for linear functional: R(x;t) = η^T u(x)
        """
        eta = self.forward(text_embedding)
        if self.linear_projection is not None:
            eta = self.linear_projection(eta)
        return eta


def load_preference_net(path: str) -> PreferenceNet:
    """
    Load trained preference network from checkpoint.
    
    Args:
        path: Path to model checkpoint
        
    Returns:
        PreferenceNet instance
    """
    checkpoint = torch.load(path, map_location='cpu')
    
    # extract architecture parameters from checkpoint or use defaults
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
