# LaPep Scripts

This directory contains all executable scripts for the LaPep framework.

## Training Scripts

- **`train_preferences.py`**: Train the preference network (g_ψ) using pairwise comparisons
- **`train_and_eval.py`**: Train the preference network and immediately run evaluation experiments

## Evaluation Scripts

- **`run_eval.py`**: Run all evaluation experiments (Section 4.1-4.5)
- **`run_prototype_experiment.py`**: Prototype experiment comparing PepDFM and PepMDLM under different language conditions
- **`test_checkpoint.py`**: Test a specific checkpoint from training

## Data Processing Scripts

- **`extract_benchmark_binders.py`**: Extract protein-peptide pairs from Benchmark_moPPIt_v3.xlsx
- **`extract_proteins_without_binders.py`**: Extract proteins without pre-existing binders from benchmark
- **`merge_binders.py`**: Merge extracted benchmark binders with existing peptide data
- **`load_test_set.py`**: Helper script to manage test datasets

## Optimization Scripts

- **`controlled_optimization.py`**: Controlled peptide optimization experiment (Algorithm 3)
- **`run_batch_optimization.py`**: Run controlled optimization for multiple protein targets
- **`sample_peptides.py`**: Sample peptides using LaPep framework (Algorithm 2)

## Usage Examples

See the main [README.md](../README.md) for detailed usage instructions.

