# LaPep: Soft Language with Hard Predictor Constraints for Peptide Generation

Inital implementation and evaluation code for LaPep, a conservative discrete generative framework that integrates soft natural-language preferences with hard predictor-based constraints.


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

## Running Experiments

### Training Preference Network

Train the preference network with Qwen embeddings:

```bash
python scripts/train_preferences.py \
    --config config.json \
    --output pretrained/language/preference_net.ckpt \
    --epochs 10
```

### Running Experiments

```bash
# all experiments
python scripts/run_eval.py --config config.json --output_dir ./eval_results

# specific experiment
python scripts/run_eval.py --config config.json --experiments 4.1
```

# Potential options: 
Run specific experiments:

```bash
# Section 4.1: Language conditioning effect
python scripts/run_eval.py \
    --config config.json \
    --output_dir ./eval_results \
    --experiments 4.1

# Section 4.2: Path independence
python scripts/run_eval.py \
    --config config.json \
    --output_dir ./eval_results \
    --experiments 4.2

# Section 4.3: Unlabeled objectives
python scripts/run_eval.py \
    --config config.json \
    --output_dir ./eval_results \
    --experiments 4.3

# Section 4.4: Ablations
python scripts/run_eval.py \
    --config config.json \
    --output_dir ./eval_results \
    --experiments 4.4

# Section 4.5: Generality across generators
python scripts/run_eval.py \
    --config config.json \
    --output_dir ./eval_results \
    --experiments 4.5
```

### Generating Peptides

```bash
python scripts/sample_peptides.py \
    --config config.json \
    --prompt "Generate a peptide with high binding affinity and low toxicity" \
    --num_samples 100 \
    --output peptides.txt
```

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
