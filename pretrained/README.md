# Pretrained Models Directory

This directory contains pretrained model checkpoints for LaPep.

## Directory Structure

```
pretrained/
├── generators/          # Base generator models
│   └── peptune-pretrained.ckpt  # PepTune/MDLM pretrained model
├── predictors/          # Predictor models
│   └── binding-affinity.pt      # Binding affinity predictor
└── language/            # Language model checkpoints (if needed)
    └── preference_net.ckpt       # Trained preference network (after training)
```

## Model Files

### Generators
- **`generators/peptune-pretrained.ckpt`**: Pretrained masked discrete diffusion model (MDLM) from PepTune
  - Download from: https://drive.google.com/file/d/1oXGDpKLNF0KX0ZdOcl1NZj5Czk2lSFUn/view?usp=sharing
  - Place in: `pretrained/generators/peptune-pretrained.ckpt`

### Predictors
- **`predictors/binding-affinity.pt`**: Pretrained binding affinity Transformer model
  - Download from: https://drive.google.com/file/d/128shlEP_-rYAxPgZRCk_n0HBWVbOYSva/view?usp=sharing
  - Place in: `pretrained/predictors/binding-affinity.pt`

### Language Models
- **`language/preference_net.ckpt`**: Trained preference network (generated after training)
  - This will be created when you run `scripts/train_preferences.py`

## Usage

After placing the model files, update `config.json` with the correct paths:

```json
{
  "base_generator_path": "pretrained/generators/peptune-pretrained.ckpt",
  "predictors": {
    "binding": {
      "path": "pretrained/predictors/binding-affinity.pt",
      "type": "binding"
    }
  }
}
```

## Notes

- The `.gitignore` file should exclude these model files (they're too large for git)
- Make sure to download the models before running training or evaluation
- Model paths in `config.json` can be relative to the project root or absolute paths

