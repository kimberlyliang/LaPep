"""
Batch script to run controlled optimization for multiple protein targets.

Reads protein targets and starting peptides from a JSON file and runs
the controlled optimization experiment for each target.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.controlled_optimization import ControlledOptimizationExperiment
from predictors import detect_sequence_format


def main():
    parser = argparse.ArgumentParser(
        description="Run controlled optimization for multiple protein targets"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config.json'
    )
    parser.add_argument(
        '--targets_file',
        type=str,
        required=True,
        help='JSON file with protein targets and starting peptides'
    )
    parser.add_argument(
        '--generator_type',
        type=str,
        choices=['pepmdlm', 'pepdfm'],
        default='pepmdlm',
        help='Generator type: pepmdlm (SMILES) or pepdfm (WT)'
    )
    parser.add_argument(
        '--generator_path',
        type=str,
        default=None,
        help='Path to generator model (overrides config)'
    )
    parser.add_argument(
        '--num_steps',
        type=int,
        default=50,
        help='Number of optimization steps'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=1,
        help='Number of optimized samples per condition'
    )
    parser.add_argument(
        '--output_base_dir',
        type=str,
        default='eval_results',
        help='Base output directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run on'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--targets',
        type=str,
        nargs='+',
        default=None,
        help='Specific targets to run (if None, runs all)'
    )
    
    args = parser.parse_args()
    
    # Load targets
    with open(args.targets_file, 'r') as f:
        targets_data = json.load(f)
    
    # Filter targets if specified
    if args.targets:
        targets_to_run = {k: v for k, v in targets_data.items() if k in args.targets}
    else:
        targets_to_run = targets_data
    
    print("="*80)
    print("BATCH CONTROLLED OPTIMIZATION EXPERIMENT")
    print("="*80)
    print(f"Targets file: {args.targets_file}")
    print(f"Number of targets: {len(targets_to_run)}")
    print(f"Generator type: {args.generator_type}")
    print(f"Targets: {list(targets_to_run.keys())}")
    print("="*80)
    
    # Initialize experiment once (shared across all targets)
    experiment = ControlledOptimizationExperiment(
        config_path=args.config,
        generator_type=args.generator_type,
        generator_path=args.generator_path,
        device=args.device
    )
    
    # Run experiment for each target
    all_results = {}
    
    for target_name, target_info in targets_to_run.items():
        protein_id = target_info.get('protein_id', target_name)
        starting_peptide = target_info.get('starting_peptide')
        
        if not starting_peptide:
            print(f"\n⚠ Skipping {target_name}: No starting_peptide found")
            continue
        
        print(f"\n{'='*80}")
        print(f"PROCESSING TARGET: {target_name}")
        print(f"{'='*80}")
        
        try:
            results = experiment.run_experiment(
                protein_target_id=protein_id,
                starting_peptide=starting_peptide,
                num_optimization_steps=args.num_steps,
                num_samples=args.num_samples,
                output_dir=f"{args.output_base_dir}/controlled_optimization_{target_name}",
                seed=args.seed
            )
            all_results[target_name] = results
        except Exception as e:
            print(f"\n✗ Error processing {target_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save combined results
    combined_output_dir = Path(args.output_base_dir) / "batch_optimization_results"
    combined_output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(combined_output_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("Batch experiment complete!")
    print(f"Combined results saved to: {combined_output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

