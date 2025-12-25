# LaPep: Soft Language with Hard Predictor Constraints for Peptide Generation

Inital implementation and evaluation code for LaPep, a conservative discrete generative framework that integrates soft natural-language preferences with hard predictor-based constraints.

## Quick Start

```bash
# 1. Setup virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Download models from previous paper
# Place in pretrained/generators/ and pretrained/predictors/

# 3. Create config
cp config_template.json config.json

# 4. Train preference network
python scripts/train_preferences.py --config config.json

# 5. Run experiments
python scripts/run_eval.py --config config.json --output_dir ./eval_results
```

See [QUICKSTART.md](QUICKSTART.md) for detailed instructions.

## Structure

```
lapep/
├── generators/          # Base generator wrappers (b_θ)
├── predictors/          # Hard predictor constraints
├── language/            # Text encoder and preference network (g_ψ)
│   └── text_encoder.py  # Supports Qwen, E5, BioGPT, SciBERT
├── lapep/               # Core LaPep framework
│   ├── potential.py    # U(x;t) = -R(x;t) + Ψ(x)
│   ├── kernel.py       # Transition kernel (Eq 10)
│   └── sampler.py       # Sampling procedure
├── eval/                # Evaluation experiments
│   ├── distribution_shift.py    # Section 4.1
│   ├── circulation.py           # Section 4.2
│   ├── motif_analysis.py        # Section 4.3
│   └── ablations.py             # Section 4.4
├── scripts/             # Training and evaluation scripts
└── pretrained/          # Model checkpoints directory
```

## Setup

See [SETUP.md](SETUP.md) for detailed setup instructions.

### Quick Setup

1. **Create virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Download pretrained models**:
   - **Base generator**: Download `peptune-pretrained.ckpt` from [PepTune](https://drive.google.com/file/d/1oXGDpKLNF0KX0ZdOcl1NZj5Czk2lSFUn/view?usp=sharing)
     - Place in: `pretrained/generators/peptune-pretrained.ckpt`
   - **Binding predictor**: Download `binding-affinity.pt` from [TR2-D2](https://drive.google.com/file/d/128shlEP_-rYAxPgZRCk_n0HBWVbOYSva/view?usp=sharing)
     - Place in: `pretrained/predictors/binding-affinity.pt`

3. **Create configuration file**:
```bash
cp config_template.json config.json
```
   The template already has the correct paths to the `pretrained/` directory and uses Qwen embeddings by default.

## Running Experiments

### Training Preference Network

First, train the preference network with Qwen embeddings:

```bash
python scripts/train_preferences.py \
    --config config.json \
    --output pretrained/language/preference_net.ckpt \
    --epochs 10
```

### Running Experiments

```bash
# All experiments
python scripts/run_eval.py --config config.json --output_dir ./eval_results

# Specific experiment
python scripts/run_eval.py --config config.json --experiments 4.1
```

### Generating Peptides

```bash
python scripts/sample_peptides.py \
    --config config.json \
    --prompt "Generate a peptide with high binding affinity and low toxicity" \
    --num_samples 100 \
    --output peptides.txt
```

## Text Encoder Options

The code supports multiple text encoders via HuggingFace:

- **Qwen3 Embedding** (default): `Qwen/Qwen3-Embedding-0.6B` (initially implemented this)
- **Qwen2.5**: `qwen2-0.5b`, `qwen2-1.5b`, `qwen2-3b`
- **E5**: `e5`
- **BioGPT**: `biogpt`
- **SciBERT**: `scibert`

Set in `config.json`:
```json
{
  "text_encoder_name": "Qwen/Qwen3-Embedding-0.6B"
}
```
