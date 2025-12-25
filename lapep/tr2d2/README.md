# TR2-D2 Files for LaPep

This directory contains the necessary files from TR2-D2 that LaPep needs.

## Structure

```
lapep/tr2d2/
├── diffusion.py              # Diffusion model (PepMDLM)
├── roformer.py              # RoFormer architecture
├── noise_schedule.py         # Noise schedule
├── utils/
│   ├── app.py               # PeptideAnalyzer
│   ├── utils.py             # Utilities
│   └── timer.py             # Timer utilities
├── scoring/
│   └── functions/
│       ├── binding.py       # BindingAffinity class
│       └── binding_utils.py # Binding utilities
├── tokenizer/
│   ├── my_tokenizers.py     # SMILES tokenizer
│   ├── new_vocab.txt        # Vocabulary
│   └── new_splits.txt       # Splits
└── configs/
    └── peptune_config.yaml  # Config file
```

## Usage

The wrappers in `generators/peptune_wrapper.py` and `predictors/binding_wrapper.py` automatically find and use these files.

No manual path configuration needed!

