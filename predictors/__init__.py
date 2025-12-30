"""
Unified Predictor Interface for LaPep

Supports both WT (wild-type amino acid sequences) and SMILES formats.
Automatically detects format and loads appropriate predictors.
"""

from typing import Dict, Optional, Union
import numpy as np


def detect_sequence_format(sequence: str) -> str:
    """
    Detect if a sequence is WT (amino acids) or SMILES.
    
    Args:
        sequence: Input sequence
    
    Returns:
        'wt' or 'smiles'
    """
    sequence = sequence.strip().upper()
    
    # WT sequences contain only standard amino acid letters
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    
    # Check if it's WT: all characters are valid amino acids
    if all(c in valid_aa for c in sequence):
        return 'wt'
    
    # Otherwise, assume SMILES (contains special characters like [, ], =, (, ), etc.)
    return 'smiles'


def is_wt_sequence(sequence: str) -> bool:
    """Check if sequence is WT format."""
    return detect_sequence_format(sequence) == 'wt'


def is_smiles_sequence(sequence: str) -> bool:
    """Check if sequence is SMILES format."""
    return detect_sequence_format(sequence) == 'smiles'

