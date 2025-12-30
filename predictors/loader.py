"""
Unified Predictor Loader

Loads predictors for both WT and SMILES formats based on configuration.
Automatically selects the appropriate predictor based on generator type or explicit format.
"""

import json
from typing import Dict, Optional, Any
from pathlib import Path


def load_predictors(
    config: Dict,
    format_type: Optional[str] = None,
    device: str = 'cuda',
    protein_seq: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load predictors based on format type (WT or SMILES).
    
    Args:
        config: Configuration dictionary with predictor paths
        format_type: 'wt' or 'smiles' (auto-detect from generator if None)
        device: Device to load predictors on
        protein_seq: Protein sequence for binding predictor
    
    Returns:
        Dictionary of loaded predictors
    """
    predictors = {}
    predictors_config = config.get('predictors', {})
    
    # Determine format if not specified
    # Mapping: WT → PepDFM, SMILES → PepMDLM
    if format_type is None:
        # Try to infer from generator type
        generator_type = config.get('generator_type', config.get('base_generator_type', 'pepmdlm'))
        if generator_type == 'pepdfm':
            format_type = 'wt'  # PepDFM generates WT amino acid sequences
        else:
            format_type = 'smiles'  # PepMDLM generates SMILES sequences
    
    format_type = format_type.lower()
    
    print(f"\n[Predictors] Loading {format_type.upper()} format predictors...")
    
    # Load binding predictor
    if 'binding' in predictors_config:
        pred_config = predictors_config['binding']
        if format_type == 'wt':
            from predictors.wt.wt_binding_wrapper import WTBindingPredictor
            predictors['binding'] = WTBindingPredictor.load(
                pred_config.get('path'),
                device=device,
                protein_seq=protein_seq or config.get('protein_seq'),
                model_type=pred_config.get('model_type', 'pooled')
            )
        else:  # SMILES
            from predictors.smiles.binding import BindingPredictor
            predictors['binding'] = BindingPredictor.load(
                pred_config.get('path'),
                device=device,
                protein_seq=protein_seq or config.get('protein_seq')
            )
        print(f"  ✓ Binding predictor loaded ({format_type})")
    
    # Load toxicity predictor
    if 'toxicity' in predictors_config:
        pred_config = predictors_config['toxicity']
        if format_type == 'wt':
            from predictors.wt.wt_toxicity import WTToxicityPredictor
            predictors['toxicity'] = WTToxicityPredictor.load(
                pred_config.get('path'),
                model=pred_config.get('model', 2),
                threshold=pred_config.get('threshold', 0.38)
            )
        else:  # SMILES
            from predictors.smiles.toxicity import ToxicityPredictor
            predictors['toxicity'] = ToxicityPredictor.load(
                pred_config.get('path'),
                device=device
            )
        print(f"  ✓ Toxicity predictor loaded ({format_type})")
    
    # Load half-life predictor
    if 'halflife' in predictors_config:
        pred_config = predictors_config['halflife']
        if format_type == 'wt':
            from predictors.wt.wt_halflife import Halflife as WTHalflife
            # WT halflife uses a different interface
            halflife_model = WTHalflife(device=device)
            # Wrap it to match interface
            class WTHalflifeWrapper:
                def __init__(self, model):
                    self.model = model
                def predict(self, peptide: str) -> float:
                    return float(self.model.predict_hours([peptide])[0])
                def normalize(self, value: float) -> float:
                    return float(np.clip(value / 100.0, 0.0, 1.0))  # Normalize hours to [0,1]
            predictors['halflife'] = WTHalflifeWrapper(halflife_model)
        else:  # SMILES
            from predictors.smiles.halflife import HalfLifePredictor
            predictors['halflife'] = HalfLifePredictor.load(
                pred_config.get('path')
            )
        print(f"  ✓ Half-life predictor loaded ({format_type})")
    
    # Load hemolysis predictor (SMILES only for now)
    if 'hemolysis' in predictors_config:
        pred_config = predictors_config['hemolysis']
        if format_type == 'wt':
            print("  ⚠ Hemolysis predictor not available for WT format, skipping")
        else:  # SMILES
            from predictors.smiles.hemolysis import HemolysisPredictor
            hemolysis_path = pred_config.get('path')
            # Handle null/None path - use placeholder
            if hemolysis_path is None or hemolysis_path == 'null':
                predictors['hemolysis'] = HemolysisPredictor(model=None, device=device)
                print(f"  ✓ Hemolysis predictor loaded ({format_type}) - using placeholder")
            else:
                predictors['hemolysis'] = HemolysisPredictor.load(
                    hemolysis_path,
                    device=device
                )
                print(f"  ✓ Hemolysis predictor loaded ({format_type})")
    
    print(f"\n[Predictors] Loaded {len(predictors)} predictor(s)")
    return predictors


def load_predictors_from_config(
    config_path: str,
    format_type: Optional[str] = None,
    device: str = 'cuda',
    protein_seq: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load predictors from a config file.
    
    Args:
        config_path: Path to config.json
        format_type: 'wt' or 'smiles' (auto-detect if None)
        device: Device to load predictors on
        protein_seq: Protein sequence for binding predictor
    
    Returns:
        Dictionary of loaded predictors
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return load_predictors(config, format_type=format_type, device=device, protein_seq=protein_seq)

