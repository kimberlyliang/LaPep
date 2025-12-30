# Unified Predictor System for WT and SMILES

This directory contains a unified predictor system that supports both WT (wild-type amino acid sequences) and SMILES formats.

## Architecture

```
predictors/
├── __init__.py          # Format detection utilities
├── loader.py            # Unified predictor loader
├── wt/                  # WT format predictors
│   ├── wt_binding.py
│   ├── wt_binding_wrapper.py  # Wrapper for LaPep interface
│   ├── wt_halflife.py
│   └── wt_toxicity.py
└── smiles/             # SMILES format predictors
    ├── binding.py
    ├── toxicity.py
    ├── halflife.py
    └── hemolysis.py
```

## Usage

### Automatic Format Detection

The system automatically detects format based on generator type:

```python
from predictors.loader import load_predictors

# Load predictors - format auto-detected from generator_type
predictors = load_predictors(
    config,
    format_type=None,  # Auto-detect from generator_type
    device='cuda',
    protein_seq=config.get('protein_seq')
)
```

### Explicit Format Specification

You can also explicitly specify the format:

```python
# For WT sequences (PepDFM)
predictors = load_predictors(
    config,
    format_type='wt',
    device='cuda'
)

# For SMILES sequences (PepMDLM)
predictors = load_predictors(
    config,
    format_type='smiles',
    device='cuda'
)
```

### Format Detection Utilities

```python
from predictors import detect_sequence_format, is_wt_sequence, is_smiles_sequence

# Detect format
format_type = detect_sequence_format("HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR")  # Returns 'wt'
format_type = detect_sequence_format("CC(=O)NCCC1=CNc2c1cc(OC)cc2")     # Returns 'smiles'

# Check format
if is_wt_sequence(peptide):
    # Handle WT sequence
    pass
elif is_smiles_sequence(peptide):
    # Handle SMILES sequence
    pass
```

## Predictor Interface

All predictors implement a consistent interface:

```python
class Predictor:
    def predict(self, peptide: str) -> float:
        """Predict score for a peptide."""
        pass
    
    def normalize(self, value: float) -> float:
        """Normalize score to [0, 1] range."""
        pass
    
    @classmethod
    def load(cls, path: str, **kwargs):
        """Load predictor from file."""
        pass
```

## Supported Predictors

### WT Format
- **Binding**: `WTBindingPredictor` (uses ESM for protein embeddings)
- **Toxicity**: `WTToxicityPredictor` (uses ToxinPred3)
- **Half-life**: `Halflife` (XGBoost model)

### SMILES Format
- **Binding**: `BindingPredictor`
- **Toxicity**: `ToxicityPredictor` (Transformer-based)
- **Half-life**: `HalfLifePredictor`
- **Hemolysis**: `HemolysisPredictor`

## Configuration

In your `config.json`, specify predictor paths:

```json
{
  "generator_type": "pepmdlm",  // or "pepdfm" for WT
  "predictors": {
    "binding": {
      "path": "path/to/binding/model.pt",
      "model_type": "pooled"  // For WT binding
    },
    "toxicity": {
      "path": "path/to/toxicity/model.pt",
      "model": 2,  // For WT: 1=ML, 2=Hybrid
      "threshold": 0.38
    },
    "halflife": {
      "path": "path/to/halflife/model.pth"
    },
    "hemolysis": {
      "path": "path/to/hemolysis/model.pkl"
    }
  }
}
```

## Integration with Scripts

All scripts have been updated to use the unified loader:

- `scripts/controlled_optimization.py`
- `scripts/run_batch_optimization.py`
- `scripts/train_preferences.py`
- `scripts/run_eval.py`
- `scripts/train_and_eval.py`

The format is automatically detected from `generator_type` in the config:
- `generator_type: "pepdfm"` → WT format predictors
- `generator_type: "pepmdlm"` → SMILES format predictors

## Example

```python
import json
from predictors.loader import load_predictors

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

# Load predictors (format auto-detected)
predictors = load_predictors(
    config,
    device='cuda',
    protein_seq=config.get('protein_seq')
)

# Use predictors
peptide = "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR"  # WT or SMILES
binding_score = predictors['binding'].predict(peptide)
toxicity_score = predictors['toxicity'].predict(peptide)
```

## Notes

- WT predictors require protein sequence for binding prediction
- SMILES predictors work with SMILES strings directly
- Format conversion (WT ↔ SMILES) is handled automatically when needed
- All predictors return normalized scores in [0, 1] range via `normalize()`

