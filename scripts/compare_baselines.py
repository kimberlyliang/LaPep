"""
Baseline Comparison: LaPep vs Text-Guided Protein Design Models

Compares LaPep against baseline models:
- ProteinDT
- BioM3
- InstructPro

For each model, generates peptides using the same prompts and evaluates:
- Constraint satisfaction (binding, hemolysis)
- Stability metrics (protease motifs, proline, modifications)
- Diversity (Shannon entropy)

This script provides a framework for comparing external models. You'll need to
implement the model-specific interfaces based on how each baseline model works.
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from predictors.loader import load_predictors
from scripts.run_prototype_experiment import (
    evaluate_wt_sequences, evaluate_smiles_sequences,
    count_protease_motifs, compute_proline_fraction, compute_shannon_entropy,
    _smiles_to_sequence_simple, _wt_to_smiles
)
from lapep.sampler import sample_peptide


# ============================================================================
# Baseline Model Interfaces (To Be Implemented)
# ============================================================================

class BaselineModelInterface:
    """Base class for baseline model interfaces."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda'):
        self.model_path = model_path
        self.device = device
        self.model = None
    
    def load_model(self):
        """Load the baseline model."""
        raise NotImplementedError("Subclasses must implement load_model")
    
    def generate(self, prompt: str, num_samples: int = 100, **kwargs) -> List[str]:
        """
        Generate peptides from a prompt.
        
        Args:
            prompt: Natural language prompt
            num_samples: Number of peptides to generate
            **kwargs: Model-specific arguments
            
        Returns:
            List of generated peptide sequences (WT or SMILES format)
        """
        raise NotImplementedError("Subclasses must implement generate")
    
    def get_output_format(self) -> str:
        """Return 'wt' or 'smiles' indicating output format."""
        raise NotImplementedError("Subclasses must implement get_output_format")


class ProteinDTInterface(BaselineModelInterface):
    """
    Interface for ProteinDT model.
    
    ProteinDT is a text-guided protein design model. You'll need to adapt
    this based on how ProteinDT is actually used.
    """
    
    def load_model(self):
        """Load ProteinDT model."""
        try:
            # Try to import and load ProteinDT
            # Uncomment and adapt when ProteinDT is available:
            # from proteindt import ProteinDTModel
            # self.model = ProteinDTModel.load(self.model_path)
            # return
            
            # If not available, set model to None and mark as unavailable
            self.model = None
            self._available = False
            print("⚠ ProteinDT interface not yet implemented")
            print("  To implement:")
            print("    1. Install ProteinDT: pip install proteindt")
            print("    2. Load model in load_model() method")
            print("    3. Implement generate() method")
        except ImportError as e:
            self.model = None
            self._available = False
            print(f"⚠ ProteinDT not available: {e}")
            print("  Install with: pip install proteindt")
    
    def generate(self, prompt: str, num_samples: int = 100, **kwargs) -> List[str]:
        """Generate peptides using ProteinDT."""
        if not hasattr(self, '_available') or not self._available:
            raise NotImplementedError(
                "ProteinDT interface not implemented. "
                "To use ProteinDT:\n"
                "  1. Install: pip install proteindt\n"
                "  2. Implement load_model() and generate() methods in scripts/compare_baselines.py"
            )
        
        if self.model is None:
            raise RuntimeError("ProteinDT model not loaded. Call load_model() first.")
        
        # Implementation would go here:
        # peptides = []
        # for _ in range(num_samples):
        #     peptide = self.model.generate(prompt=prompt, **kwargs)
        #     peptides.append(peptide)
        # return peptides
        
        raise NotImplementedError("ProteinDT.generate() not yet implemented")
    
    def get_output_format(self) -> str:
        """ProteinDT likely outputs WT sequences."""
        return 'wt'


class BioM3Interface(BaselineModelInterface):
    """
    Interface for BioM3 model.
    
    BioM3 is a multimodal model for protein design. Adapt based on actual usage.
    """
    
    def load_model(self):
        """Load BioM3 model."""
        try:
            # Try to import and load BioM3
            # Uncomment and adapt when BioM3 is available:
            # from biom3 import BioM3Model
            # self.model = BioM3Model.load(self.model_path)
            # return
            
            # If not available, set model to None and mark as unavailable
            self.model = None
            self._available = False
            print("⚠ BioM3 interface not yet implemented")
            print("  To implement:")
            print("    1. Install BioM3: pip install biom3")
            print("    2. Load model in load_model() method")
            print("    3. Implement generate() method")
        except ImportError as e:
            self.model = None
            self._available = False
            print(f"⚠ BioM3 not available: {e}")
            print("  Install with: pip install biom3")
    
    def generate(self, prompt: str, num_samples: int = 100, **kwargs) -> List[str]:
        """Generate peptides using BioM3."""
        if not hasattr(self, '_available') or not self._available:
            raise NotImplementedError(
                "BioM3 interface not implemented. "
                "To use BioM3:\n"
                "  1. Install: pip install biom3\n"
                "  2. Implement load_model() and generate() methods in scripts/compare_baselines.py"
            )
        
        if self.model is None:
            raise RuntimeError("BioM3 model not loaded. Call load_model() first.")
        
        # Implementation would go here:
        # peptides = []
        # for _ in range(num_samples):
        #     peptide = self.model.generate(prompt=prompt, **kwargs)
        #     peptides.append(peptide)
        # return peptides
        
        raise NotImplementedError("BioM3.generate() not yet implemented")
    
    def get_output_format(self) -> str:
        """BioM3 likely outputs WT sequences."""
        return 'wt'


class InstructProInterface(BaselineModelInterface):
    """
    Interface for InstructPro model.
    
    InstructPro is an instruction-tuned protein design model. Adapt based on actual usage.
    """
    
    def load_model(self):
        """Load InstructPro model."""
        try:
            # Try to import and load InstructPro
            # Uncomment and adapt when InstructPro is available:
            # from instructpro import InstructProModel
            # self.model = InstructProModel.load(self.model_path)
            # return
            
            # If not available, set model to None and mark as unavailable
            self.model = None
            self._available = False
            print("⚠ InstructPro interface not yet implemented")
            print("  To implement:")
            print("    1. Install InstructPro: pip install instructpro")
            print("    2. Load model in load_model() method")
            print("    3. Implement generate() method")
        except ImportError as e:
            self.model = None
            self._available = False
            print(f"⚠ InstructPro not available: {e}")
            print("  Install with: pip install instructpro")
    
    def generate(self, prompt: str, num_samples: int = 100, **kwargs) -> List[str]:
        """Generate peptides using InstructPro."""
        if not hasattr(self, '_available') or not self._available:
            raise NotImplementedError(
                "InstructPro interface not implemented. "
                "To use InstructPro:\n"
                "  1. Install: pip install instructpro\n"
                "  2. Implement load_model() and generate() methods in scripts/compare_baselines.py"
            )
        
        if self.model is None:
            raise RuntimeError("InstructPro model not loaded. Call load_model() first.")
        
        # Implementation would go here:
        # peptides = []
        # for _ in range(num_samples):
        #     peptide = self.model.generate(prompt=prompt, **kwargs)
        #     peptides.append(peptide)
        # return peptides
        
        raise NotImplementedError("InstructPro.generate() not yet implemented")
    
    def get_output_format(self) -> str:
        """InstructPro likely outputs WT sequences."""
        return 'wt'


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_peptides(
    peptides: List[str],
    predictors: Dict,
    output_format: str = 'wt',
    generator_name: str = 'unknown'
) -> Dict:
    """
    Evaluate peptides using standardized metrics.
    
    Args:
        peptides: List of peptide sequences
        predictors: Dict of predictor objects
        output_format: 'wt' or 'smiles'
        generator_name: Name of generator (for logging)
        
    Returns:
        Dict with all evaluation metrics
    """
    if not peptides:
        return {}
    
    print(f"  Evaluating {len(peptides)} peptides from {generator_name}...")
    
    # Convert to appropriate format for predictors (predictors expect SMILES)
    binding_scores = []
    hemolysis_scores = []
    
    for peptide in peptides:
        if output_format == 'wt':
            # Convert WT to SMILES for predictor evaluation
            smiles = _wt_to_smiles(peptide)
            if not smiles:
                binding_scores.append(np.nan)
                hemolysis_scores.append(np.nan)
                continue
        else:
            smiles = peptide
        
        try:
            binding_scores.append(predictors['binding'].predict(smiles))
            hemolysis_scores.append(predictors['hemolysis'].predict(smiles))
        except Exception as e:
            print(f"    Warning: Predictor evaluation failed: {e}")
            binding_scores.append(np.nan)
            hemolysis_scores.append(np.nan)
    
    metrics = {
        'binding_scores': np.array(binding_scores),
        'hemolysis_scores': np.array(hemolysis_scores)
    }
    
    # Constraint satisfaction
    binding_valid = np.array(binding_scores)[~np.isnan(binding_scores)]
    hemolysis_valid = np.array(hemolysis_scores)[~np.isnan(hemolysis_scores)]
    
    binding_satisfied = np.mean(binding_valid >= 0.7) if len(binding_valid) > 0 else 0.0
    hemolysis_satisfied = np.mean(hemolysis_valid <= 0.3) if len(hemolysis_valid) > 0 else 0.0
    metrics['constraint_satisfaction'] = (binding_satisfied + hemolysis_satisfied) / 2.0
    
    # Generator-specific metrics
    if output_format == 'wt':
        valid_wt = [p for p in peptides if all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in p.upper())]
        if valid_wt:
            wt_metrics = evaluate_wt_sequences(valid_wt)
            metrics.update(wt_metrics)
            
            # Diversity
            if 'shannon_entropy' in metrics:
                metrics['diversity'] = np.mean(metrics['shannon_entropy'])
            else:
                metrics['diversity'] = np.mean([compute_shannon_entropy(seq) for seq in valid_wt[:100]])
    else:
        # SMILES format
        smiles_metrics = evaluate_smiles_sequences(peptides)
        metrics.update(smiles_metrics)
        
        # Diversity (convert to sequences first)
        sequences = [_smiles_to_sequence_simple(s) for s in peptides if _smiles_to_sequence_simple(s)]
        if sequences:
            metrics['diversity'] = np.mean([compute_shannon_entropy(seq) for seq in sequences])
        else:
            metrics['diversity'] = 0.0
    
    return metrics


# ============================================================================
# Main Comparison Function
# ============================================================================

def compare_baselines(
    config_path: str,
    prompts: Dict[str, str],
    num_samples: int = 500,
    output_dir: Optional[str] = None,
    device: str = 'cuda',
    baseline_models: Optional[Dict[str, BaselineModelInterface]] = None,
    lapep_generator: Optional[any] = None,
    lapep_text_encoder: Optional[any] = None,
    lapep_preference_net: Optional[any] = None,
    lapep_constraints: Optional[Dict] = None
) -> Dict:
    """
    Compare LaPep against baseline models.
    
    Args:
        config_path: Path to config.json
        prompts: Dict mapping prompt names to prompt text
        num_samples: Number of samples per prompt per model
        output_dir: Output directory for results
        device: Device to run on
        baseline_models: Dict mapping model names to BaselineModelInterface instances
        lapep_*: LaPep components (if None, will be loaded from config)
        
    Returns:
        Dict with comparison results
    """
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"eval_results/baseline_comparison_{timestamp}")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("BASELINE COMPARISON: LaPep vs Text-Guided Protein Design Models")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Prompts: {list(prompts.keys())}")
    print(f"Samples per prompt: {num_samples}")
    print("=" * 80)
    
    # Load predictors
    print("\nLoading predictors...")
    predictors = load_predictors(
        config, format_type='smiles', device=device,
        protein_seq=config.get('protein_seq')
    )
    if 'hemolysis' not in predictors:
        from predictors.smiles.hemolysis import HemolysisPredictor
        predictors['hemolysis'] = HemolysisPredictor(device=device)
    print(f"✓ Loaded {len(predictors)} predictor(s)")
    
    # Load LaPep components if not provided
    if lapep_generator is None:
        print("\nLoading LaPep components...")
        from generators.peptune_wrapper import load_peptune_generator
        from language.text_encoder import load_text_encoder
        from language.preference_net import load_preference_net
        
        generator_type = config.get('generator_type', 'pepmdlm')
        if generator_type == 'pepmdlm':
            lapep_generator = load_peptune_generator(
                config['base_generator_path'], device=device
            )
        else:
            from generators.dfm_wrapper import load_dfm_model
            lapep_generator = load_dfm_model(config['dfm_model_path'])
        
        lapep_text_encoder = load_text_encoder(
            config.get('text_encoder_name', 'Qwen/Qwen3-Embedding-0.6B'),
            device=device
        )
        lapep_preference_net = load_preference_net(
            config.get('preference_net_path'),
            device=device
        )
        
        lapep_constraints = {
            'strength': config.get('constraints', {}).get('strength', 2.0),
            'weights': config.get('constraints', {}).get('weights', {'binding': 1.0, 'hemolysis': 1.0})
        }
        print("✓ LaPep components loaded")
    
    # Initialize baseline models if not provided
    if baseline_models is None:
        baseline_models = {
            'ProteinDT': ProteinDTInterface(device=device),
            'BioM3': BioM3Interface(device=device),
            'InstructPro': InstructProInterface(device=device)
        }
        
        # Try to load baseline models
        for name, model in baseline_models.items():
            try:
                model.load_model()
                if model.model is not None:
                    print(f"✓ {name} loaded")
                else:
                    print(f"⚠ {name} not available (interface not implemented)")
            except Exception as e:
                print(f"⚠ {name} failed to load: {e}")
    
    # Run comparison
    all_results = {}
    
    for prompt_name, prompt_text in prompts.items():
        print(f"\n{'='*80}")
        print(f"Prompt: {prompt_name}")
        print(f"Text: {prompt_text}")
        print(f"{'='*80}")
        
        prompt_results = {}
        
        # Generate with LaPep
        print("\n[LaPep] Generating peptides...")
        lapep_peptides = []
        for i in range(num_samples):
            try:
                peptide = sample_peptide(
                    lapep_generator,
                    prompt=prompt_text,
                    predictors=predictors,
                    constraints=lapep_constraints,
                    text_encoder=lapep_text_encoder,
                    preference_net=lapep_preference_net,
                    num_steps=50,
                    seed=42 + i,
                    language_weight=1.0
                )
                lapep_peptides.append(peptide)
            except Exception as e:
                print(f"  Warning: Failed to generate sample {i}: {e}")
                continue
        
        lapep_metrics = evaluate_peptides(
            lapep_peptides, predictors, output_format='smiles', generator_name='LaPep'
        )
        prompt_results['LaPep'] = {
            'peptides': lapep_peptides,
            'metrics': lapep_metrics
        }
        print(f"  Generated {len(lapep_peptides)} peptides")
        
        # Generate with baseline models
        for model_name, model in baseline_models.items():
            if model.model is None:
                print(f"\n[{model_name}] Skipping (model not loaded)")
                continue
            
            print(f"\n[{model_name}] Generating peptides...")
            try:
                baseline_peptides = model.generate(
                    prompt=prompt_text,
                    num_samples=num_samples
                )
                
                if baseline_peptides:
                    baseline_metrics = evaluate_peptides(
                        baseline_peptides, predictors,
                        output_format=model.get_output_format(),
                        generator_name=model_name
                    )
                    prompt_results[model_name] = {
                        'peptides': baseline_peptides,
                        'metrics': baseline_metrics
                    }
                    print(f"  Generated {len(baseline_peptides)} peptides")
                else:
                    print(f"  ⚠ No peptides generated")
            except Exception as e:
                print(f"  ✗ Generation failed: {e}")
                continue
        
        all_results[prompt_name] = prompt_results
    
    # Save results
    print(f"\n{'='*80}")
    print("Saving results...")
    print(f"{'='*80}")
    
    results_to_save = {}
    for prompt_name, prompt_results in all_results.items():
        results_to_save[prompt_name] = {}
        for model_name, model_results in prompt_results.items():
            results_to_save[prompt_name][model_name] = {
                'metrics': {k: v.tolist() if isinstance(v, np.ndarray) else v
                           for k, v in model_results['metrics'].items()}
            }
    
    with open(output_dir / "comparison_results.json", 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    # Create comparison tables
    print("Creating comparison tables...")
    _create_comparison_tables(all_results, output_dir)
    
    # Create comparison plots
    print("Creating comparison plots...")
    _create_comparison_plots(all_results, output_dir)
    
    print(f"\n{'='*80}")
    print("Baseline comparison complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")
    
    return all_results


def _create_comparison_tables(all_results: Dict, output_dir: Path):
    """Create comparison tables."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    
    # Overall comparison table
    table_data = []
    
    for prompt_name, prompt_results in all_results.items():
        for model_name, model_results in prompt_results.items():
            metrics = model_results['metrics']
            
            row = {
                'Prompt': prompt_name,
                'Model': model_name,
            }
            
            # Add metrics
            if 'constraint_satisfaction' in metrics:
                row['Constraint Satisfaction'] = f"{metrics['constraint_satisfaction']:.4f}"
            
            if 'binding_scores' in metrics:
                binding = metrics['binding_scores']
                if isinstance(binding, np.ndarray):
                    binding_valid = binding[~np.isnan(binding)]
                    if len(binding_valid) > 0:
                        row['Binding (mean±std)'] = f"{np.mean(binding_valid):.4f}±{np.std(binding_valid):.4f}"
            
            if 'hemolysis_scores' in metrics:
                hemolysis = metrics['hemolysis_scores']
                if isinstance(hemolysis, np.ndarray):
                    hemolysis_valid = hemolysis[~np.isnan(hemolysis)]
                    if len(hemolysis_valid) > 0:
                        row['Hemolysis (mean±std)'] = f"{np.mean(hemolysis_valid):.4f}±{np.std(hemolysis_valid):.4f}"
            
            if 'diversity' in metrics:
                row['Diversity'] = f"{metrics['diversity']:.4f}"
            
            if 'protease_motifs' in metrics:
                motifs = metrics['protease_motifs']
                if isinstance(motifs, np.ndarray):
                    motifs_valid = motifs[~np.isnan(motifs)]
                    if len(motifs_valid) > 0:
                        row['Protease Motifs'] = f"{np.mean(motifs_valid):.2f}"
            
            if 'proline_fraction' in metrics:
                proline = metrics['proline_fraction']
                if isinstance(proline, np.ndarray):
                    proline_valid = proline[~np.isnan(proline)]
                    if len(proline_valid) > 0:
                        row['Proline Fraction'] = f"{np.mean(proline_valid):.4f}"
            
            table_data.append(row)
    
    if table_data:
        df = pd.DataFrame(table_data)
        df.to_csv(tables_dir / "comparison_table.csv", index=False)
        
        # LaTeX version
        latex_table = df.to_latex(index=False, float_format="%.4f", escape=False)
        with open(tables_dir / "comparison_table.tex", 'w') as f:
            f.write(latex_table)
        
        print(f"✓ Comparison table saved to {tables_dir}")


def _create_comparison_plots(all_results: Dict, output_dir: Path):
    """Create comparison plots."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # Plot constraint satisfaction comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = []
    constraint_scores = []
    
    for prompt_name, prompt_results in all_results.items():
        for model_name, model_results in prompt_results.items():
            metrics = model_results['metrics']
            if 'constraint_satisfaction' in metrics:
                models.append(f"{model_name}\n({prompt_name})")
                constraint_scores.append(metrics['constraint_satisfaction'])
    
    if models:
        ax.bar(range(len(models)), constraint_scores)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Constraint Satisfaction', fontsize=12)
        ax.set_title('Constraint Satisfaction Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(fig_dir / "constraint_satisfaction_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Comparison plots saved to {fig_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare LaPep against baseline text-guided protein design models"
    )
    parser.add_argument('--config', type=str, required=True, help='Path to config.json')
    parser.add_argument('--num_samples', type=int, default=500, help='Samples per prompt per model')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    # Prompt options
    parser.add_argument('--prompts', type=str, default='neutral,stability',
                       help='Comma-separated list of prompts to use')
    parser.add_argument('--custom_prompts', type=str, default=None,
                       help='JSON file with custom prompts')
    
    args = parser.parse_args()
    
    # Define prompts
    default_prompts = {
        'neutral': "Design a peptide with good binding and low hemolysis.",
        'stability': "Design a peptide that is protease-resistant and stable in vivo, while maintaining good binding and low hemolysis."
    }
    
    if args.custom_prompts:
        with open(args.custom_prompts, 'r') as f:
            prompts = json.load(f)
    else:
        prompt_names = [p.strip() for p in args.prompts.split(',')]
        prompts = {name: default_prompts[name] for name in prompt_names if name in default_prompts}
    
    compare_baselines(
        config_path=args.config,
        prompts=prompts,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == '__main__':
    main()

