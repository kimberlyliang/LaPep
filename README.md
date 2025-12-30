# LaPep: Soft Language with Hard Predictor Constraints for Peptide Generation

Inital implementation and evaluation code for LaPep, a conservative discrete generative framework that integrates soft natural-language preferences with hard predictor-based constraints.


## Structure

```
LaPep/
├── lapep/               # Core LaPep framework
│   ├── sampler.py       # Algorithm 2 & 3: Sampling procedures
│   ├── kernel.py        # Transition kernel computation
│   ├── potential.py     # Scalar potential U(x;t) = -R(x;t) + Ψ(x)
│   └── tr2d2/           # TR2D2 integration (PepMDLM)
├── generators/           # Base generator wrappers (b_θ)
│   ├── peptune_wrapper.py  # PepMDLM (SMILES, discrete diffusion)
│   └── dfm_wrapper.py      # PepDFM (WT, discrete flow matching)
├── predictors/          # Hard predictor constraints
│   ├── loader.py        # Unified predictor loader
│   ├── wt/              # WT amino acid predictors
│   └── smiles/          # SMILES predictors
├── language/             # Language components
│   ├── text_encoder.py  # Text encoder (E_text) - Supports Qwen, E5, BioGPT, SciBERT
│   ├── preference_net.py # Preference network (g_ψ)
│   └── llm_judge.py     # LLM judge for pairwise comparisons
├── eval/                 # Evaluation experiments
│   ├── distribution_shift.py    # Section 4.1
│   ├── circulation.py           # Section 4.2
│   ├── motif_analysis.py        # Section 4.3
│   └── ablations.py             # Section 4.4
├── scripts/              # Executable scripts (see scripts/README.md)
├── data/                 # Data files and benchmarks
├── pretrained/           # Model checkpoints directory
├── docs/                  # Documentation (see docs/README.md)
└── tests/                 # Unit tests
```

For detailed documentation, see:
- **[docs/README.md](docs/README.md)**: Documentation index
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Architecture overview
- **[scripts/README.md](scripts/README.md)**: Script usage guide
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

Use the controlled optimization script or prototype experiment for peptide generation:

```bash
# Controlled optimization (Algorithm 3)
python scripts/controlled_optimization.py \
    --config config.json \
    --protein_target <target_id> \
    --starting_peptide <peptide_sequence> \
    --num_samples 100

# Prototype experiment (Algorithm 2)
python scripts/run_prototype_experiment.py \
    --config config.json \
    --peptune_model <path> \
    --dfm_model <path> \
    --num_samples 500
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
