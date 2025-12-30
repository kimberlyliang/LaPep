"""
Language Weight Sweep Experiment

Sweeps over different language weights α to observe how stability effects
increase smoothly as language conditioning strength increases.

The potential is: U(x;t) = -α·R(x;t) + Ψ(x)
- α = 0: Predictor-only (no language effect)
- α = 1: Normal language conditioning
- α > 1: Stronger language effect relative to constraints

This script runs the prototype experiment with varying language weights
and generates plots showing how metrics change with α.
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from generators.peptune_wrapper import load_peptune_generator
from generators.dfm_wrapper import load_dfm_model
from language.text_encoder import load_text_encoder
from language.preference_net import load_preference_net
from predictors.loader import load_predictors
from lapep.sampler import sample_peptide, sample_from_fixed_seeds
from scripts.run_prototype_experiment import (
    evaluate_wt_sequences, evaluate_smiles_sequences,
    _smiles_to_sequence_simple, _wt_to_smiles
)


def run_sweep_for_generator(
    generator_name: str,
    generator: any,
    prompt: str,
    predictors: Dict,
    text_encoder: any,
    preference_net: any,
    constraints: Dict,
    language_weights: List[float],
    num_samples: int = 100,
    device: str = 'cuda',
    test_set_path: Optional[str] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Run language weight sweep for a single generator.
    
    Returns:
        Dict mapping language_weight -> Dict of metrics
    """
    results = {}
    
    print(f"\n{'='*80}")
    print(f"Running language weight sweep for {generator_name.upper()}")
    print(f"Prompt: {prompt}")
    print(f"Language weights: {language_weights}")
    print(f"{'='*80}")
    
    # Load test set seeds if provided
    seed_set = None
    use_algorithm_3 = False
    if test_set_path:
        from scripts.load_test_set import load_test_set_csv, load_test_set_json
        test_set_path_obj = Path(test_set_path)
        if test_set_path_obj.suffix == '.csv':
            test_pairs = load_test_set_csv(test_set_path)
            seed_set = [pair['peptide'] for pair in test_pairs]
            use_algorithm_3 = True
        elif test_set_path_obj.suffix == '.json':
            test_pairs = load_test_set_json(test_set_path)
            seed_set = [pair['peptide'] for pair in test_pairs]
            use_algorithm_3 = True
    
    for alpha in language_weights:
        print(f"\nLanguage weight α = {alpha:.2f}")
        print("-" * 80)
        
        # Generate samples
        samples = []
        if use_algorithm_3 and seed_set:
            print(f"  Using Algorithm 3 with {len(seed_set)} seeds...")
            completions_data = sample_from_fixed_seeds(
                base_generator=generator,
                seed_set=seed_set[:min(10, len(seed_set))],  # Limit seeds for speed
                prompt=prompt,
                predictors=predictors,
                constraints=constraints,
                text_encoder=text_encoder,
                preference_net=preference_net,
                num_steps=50,
                completions_per_seed=max(1, num_samples // min(10, len(seed_set))),
                language_weight=alpha,
                seed=42
            )
            samples = [comp['completion'] for comp in completions_data]
        else:
            print(f"  Using Algorithm 2 with {num_samples} samples...")
            for i in range(num_samples):
                try:
                    peptide = sample_peptide(
                        generator,
                        prompt=prompt,
                        predictors=predictors,
                        constraints=constraints,
                        text_encoder=text_encoder,
                        preference_net=preference_net,
                        num_steps=50,
                        language_weight=alpha,
                        seed=42 + i  # Different seed per sample
                    )
                    samples.append(peptide)
                except Exception as e:
                    print(f"    Warning: Failed to generate sample {i}: {e}")
                    continue
        
        print(f"  Generated {len(samples)} valid samples")
        
        if len(samples) == 0:
            print(f"  ⚠ No valid samples generated for α={alpha:.2f}")
            continue
        
        # Evaluate predictors (convert WT to SMILES if needed)
        binding_scores = []
        hemolysis_scores = []
        smiles_for_predictors = []
        wt_sequences = []
        
        for sample in samples:
            is_wt = all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in sample.upper())
            
            if is_wt:
                wt_seq = sample.upper()
                wt_sequences.append(wt_seq)
                smiles = _wt_to_smiles(wt_seq)
                smiles_for_predictors.append(smiles if smiles else "")
            else:
                smiles_for_predictors.append(sample)
                wt_seq = _smiles_to_sequence_simple(sample)
                wt_sequences.append(wt_seq if wt_seq else "")
        
        # Evaluate predictors
        for smiles in smiles_for_predictors:
            if smiles:
                try:
                    binding_scores.append(predictors['binding'].predict(smiles))
                    hemolysis_scores.append(predictors['hemolysis'].predict(smiles))
                except Exception as e:
                    binding_scores.append(np.nan)
                    hemolysis_scores.append(np.nan)
            else:
                binding_scores.append(np.nan)
                hemolysis_scores.append(np.nan)
        
        binding_scores = np.array(binding_scores)
        hemolysis_scores = np.array(hemolysis_scores)
        
        # Compute generator-specific metrics
        if generator_name == 'pepdfm':
            valid_wt = [s for s in wt_sequences if s]
            metrics = evaluate_wt_sequences(valid_wt)
        else:
            metrics = evaluate_smiles_sequences(samples)
        
        results[alpha] = {
            'samples': samples,
            'binding_scores': binding_scores,
            'hemolysis_scores': hemolysis_scores,
            'metrics': metrics
        }
        
        print(f"  Binding (mean): {np.nanmean(binding_scores):.4f}")
        print(f"  Hemolysis (mean): {np.nanmean(hemolysis_scores):.4f}")
    
    return results


def create_sweep_plots(
    all_results: Dict[str, Dict[float, Dict]],
    output_dir: Path,
    prompt_name: str
):
    """Create plots showing how metrics change with language weight."""
    fig_dir = output_dir / "figures" / "sweep"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data for plotting
    for generator_name, generator_results in all_results.items():
        if not generator_results:
            continue
        
        # Collect data across language weights
        alphas = sorted(generator_results.keys())
        
        # Predictor scores
        binding_means = []
        binding_stds = []
        hemolysis_means = []
        hemolysis_stds = []
        
        # Generator-specific metrics
        metric_means = defaultdict(list)
        metric_stds = defaultdict(list)
        
        for alpha in alphas:
            data = generator_results[alpha]
            binding_scores = data['binding_scores']
            hemolysis_scores = data['hemolysis_scores']
            metrics = data['metrics']
            
            binding_valid = binding_scores[~np.isnan(binding_scores)]
            hemolysis_valid = hemolysis_scores[~np.isnan(hemolysis_scores)]
            
            binding_means.append(np.mean(binding_valid) if len(binding_valid) > 0 else np.nan)
            binding_stds.append(np.std(binding_valid) if len(binding_valid) > 0 else np.nan)
            hemolysis_means.append(np.mean(hemolysis_valid) if len(hemolysis_valid) > 0 else np.nan)
            hemolysis_stds.append(np.std(hemolysis_valid) if len(hemolysis_valid) > 0 else np.nan)
            
            # Collect generator-specific metrics
            for metric_name, metric_values in metrics.items():
                if isinstance(metric_values, np.ndarray):
                    valid_values = metric_values[~np.isnan(metric_values)]
                    if len(valid_values) > 0:
                        metric_means[metric_name].append(np.mean(valid_values))
                        metric_stds[metric_name].append(np.std(valid_values))
                    else:
                        metric_means[metric_name].append(np.nan)
                        metric_stds[metric_name].append(np.nan)
        
        # Plot predictor scores
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Binding
        axes[0].errorbar(alphas, binding_means, yerr=binding_stds, 
                        marker='o', capsize=5, capthick=2, linewidth=2)
        axes[0].set_xlabel('Language Weight α', fontsize=12)
        axes[0].set_ylabel('Binding Score (mean ± std)', fontsize=12)
        axes[0].set_title(f'Binding Score vs Language Weight\n{generator_name.upper()}', 
                         fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Hemolysis
        axes[1].errorbar(alphas, hemolysis_means, yerr=hemolysis_stds,
                        marker='o', capsize=5, capthick=2, linewidth=2, color='orange')
        axes[1].set_xlabel('Language Weight α', fontsize=12)
        axes[1].set_ylabel('Hemolysis Score (mean ± std)', fontsize=12)
        axes[1].set_title(f'Hemolysis Score vs Language Weight\n{generator_name.upper()}',
                         fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(fig_dir / f"predictors_sweep_{generator_name}_{prompt_name}.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot generator-specific metrics
        if metric_means:
            n_metrics = len(metric_means)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
            if n_metrics == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes if isinstance(axes, list) else [axes]
            else:
                axes = axes.flatten()
            
            for idx, (metric_name, means) in enumerate(metric_means.items()):
                if idx >= len(axes):
                    break
                ax = axes[idx]
                stds = metric_stds[metric_name]
                ax.errorbar(alphas, means, yerr=stds, marker='o', capsize=5, 
                           capthick=2, linewidth=2)
                ax.set_xlabel('Language Weight α', fontsize=11)
                ax.set_ylabel(f'{metric_name.replace("_", " ").title()}', fontsize=11)
                ax.set_title(f'{metric_name.replace("_", " ").title()}\n{generator_name.upper()}',
                           fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for idx in range(len(metric_means), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(fig_dir / f"metrics_sweep_{generator_name}_{prompt_name}.png",
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"✓ Sweep plots saved to {fig_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run language weight sweep experiment"
    )
    parser.add_argument('--config', type=str, required=True, help='Path to config.json')
    parser.add_argument('--peptune_model', type=str, required=True, help='Path to PepMDLM model')
    parser.add_argument('--dfm_model', type=str, default=None, help='Path to PepDFM model')
    parser.add_argument('--prompt', type=str, required=True, 
                       help='Prompt to use (e.g., "stability" or full prompt text)')
    parser.add_argument('--language_weights', type=str, default='0.0,0.5,1.0,1.5,2.0,3.0',
                       help='Comma-separated list of language weights to sweep')
    parser.add_argument('--num_samples', type=int, default=100, help='Samples per weight')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--test_set', type=str, default=None, help='Test set for Algorithm 3')
    parser.add_argument('--constraint_strength', type=float, default=2.0, help='Constraint strength')
    
    args = parser.parse_args()
    
    # Parse language weights
    language_weights = [float(w.strip()) for w in args.language_weights.split(',')]
    
    # Determine prompt
    prompt_map = {
        'stability': "Design a peptide that is protease-resistant and stable in vivo, while maintaining good binding and low hemolysis.",
        'neutral': "Design a peptide with good binding and low hemolysis.",
        'no_language': None
    }
    prompt = prompt_map.get(args.prompt.lower(), args.prompt)
    prompt_name = args.prompt.lower()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"eval_results/language_weight_sweep_{prompt_name}_{timestamp}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("LANGUAGE WEIGHT SWEEP EXPERIMENT")
    print("=" * 80)
    print(f"Prompt: {prompt if prompt else 'None (predictor-only)'}")
    print(f"Language weights: {language_weights}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Load predictors
    print("\nLoading predictors...")
    smiles_config = config.copy()
    smiles_config['generator_type'] = 'pepmdlm'
    predictors = load_predictors(
        smiles_config, format_type='smiles', device=args.device,
        protein_seq=config.get('protein_seq')
    )
    
    if 'hemolysis' not in predictors:
        from predictors.smiles.hemolysis import HemolysisPredictor
        predictors['hemolysis'] = HemolysisPredictor(device=args.device)
    
    print(f"✓ Loaded {len(predictors)} predictor(s)")
    
    # Load language models
    print("\nLoading language models...")
    text_encoder = load_text_encoder(
        config.get('text_encoder_name', 'Qwen/Qwen3-Embedding-0.6B'),
        device=args.device
    )
    preference_net_path = config.get('preference_net_path')
    if preference_net_path:
        preference_net = load_preference_net(preference_net_path, device=args.device)
    else:
        preference_net = None
        print("⚠ No preference network - will use predictor-only")
    
    # Setup constraints
    constraints = {
        'strength': args.constraint_strength,
        'weights': {'binding': 1.0, 'hemolysis': 1.0}
    }
    
    # Run sweep for each generator
    all_results = {}
    
    # PepMDLM
    if args.peptune_model:
        print("\n" + "="*80)
        print("PepMDLM (SMILES)")
        print("="*80)
        try:
            generator = load_peptune_generator(args.peptune_model, device=args.device)
            if generator.model is None:
                raise RuntimeError("Failed to load PepMDLM")
            
            results = run_sweep_for_generator(
                'pepmdlm', generator, prompt, predictors, text_encoder,
                preference_net, constraints, language_weights, args.num_samples,
                args.device, args.test_set
            )
            all_results['pepmdlm'] = results
        except Exception as e:
            print(f"✗ Failed to run PepMDLM sweep: {e}")
    
    # PepDFM
    if args.dfm_model:
        print("\n" + "="*80)
        print("PepDFM (WT)")
        print("="*80)
        try:
            generator = load_dfm_model(args.dfm_model)
            if generator is None:
                raise NotImplementedError("PepDFM not implemented")
            
            results = run_sweep_for_generator(
                'pepdfm', generator, prompt, predictors, text_encoder,
                preference_net, constraints, language_weights, args.num_samples,
                args.device, args.test_set
            )
            all_results['pepdfm'] = results
        except Exception as e:
            print(f"✗ Failed to run PepDFM sweep: {e}")
    
    # Create plots
    print("\n" + "="*80)
    print("Creating sweep plots...")
    print("="*80)
    create_sweep_plots(all_results, output_dir, prompt_name)
    
    # Save results
    print("\nSaving results...")
    results_to_save = {}
    for gen_name, gen_results in all_results.items():
        results_to_save[gen_name] = {}
        for alpha, data in gen_results.items():
            results_to_save[gen_name][str(alpha)] = {
                'binding_scores': data['binding_scores'].tolist(),
                'hemolysis_scores': data['hemolysis_scores'].tolist(),
                'metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v
                           for k, v in data['metrics'].items()}
            }
    
    with open(output_dir / "sweep_results.json", 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"\n{'='*80}")
    print("Language weight sweep complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

