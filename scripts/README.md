# LaPep Scripts

This directory contains all executable scripts for the LaPep framework.

## Training Scripts

- **`train_preferences.py`**: Train the preference network (g_ψ) using pairwise comparisons
- **`train_and_eval.py`**: Train the preference network and immediately run evaluation experiments

## Evaluation Scripts

- **`run_eval.py`**: Run all evaluation experiments (Section 4.1-4.5)
- **`run_prototype_experiment.py`**: Prototype experiment comparing PepDFM and PepMDLM under different language conditions
- **`run_comprehensive_ablations.py`**: Comprehensive ablation studies (Section 4.4)
- **`run_language_weight_sweep.py`**: Language weight sweep experiment
- **`test_checkpoint.py`**: Test a specific checkpoint from training
- **`compare_baselines.py`**: Compare LaPep against external text-guided protein design models

## Data Processing Scripts

- **`extract_benchmark_binders.py`**: Extract protein-peptide pairs from Benchmark_moPPIt_v3.xlsx
- **`extract_proteins_without_binders.py`**: Extract proteins without pre-existing binders from benchmark
- **`load_test_set.py`**: Helper script to manage test datasets

## Optimization Scripts

- **`controlled_optimization.py`**: Controlled peptide optimization experiment (Algorithm 3)
- **`run_batch_optimization.py`**: Run controlled optimization for multiple protein targets

## Utility Scripts

- **`get_sample_smiles.py`**: Generate or extract sample SMILES sequences

