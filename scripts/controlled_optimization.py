"""
Controlled Peptide Optimization Experiment

For each protein target, starts from a fixed reference peptide and optimizes it
under hard predictor constraints (binding + hemolysis), while varying language conditioning.

This implements a controlled optimization setup where:
- Starting peptide is fixed across all conditions
- Predictors and thresholds are identical
- Only language conditioning varies
- Results are saved for downstream analysis
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from generators.base_generator import BaseGenerator
from generators.peptune_wrapper import PepMDLMWrapper, load_peptune_generator
from generators.dfm_wrapper import DFMWrapper, load_dfm_model
from language.text_encoder import load_text_encoder
from language.preference_net import load_preference_net
from predictors.loader import load_predictors
from predictors import detect_sequence_format
from lapep.sampler import sample_peptide, sample_from_fixed_seeds


class ControlledOptimizationExperiment:
    """
    Controlled peptide optimization experiment scaffold.
    
    Optimizes a fixed starting peptide under different language conditioning
    while keeping predictor constraints constant.
    """
    
    def __init__(
        self,
        config_path: str,
        generator_type: str = 'pepmdlm',  # options are 'pepmdlm' or 'pepdfm'
        generator_path: str = None,
        device: str = 'cuda'
    ):
        """
        Initialize the experiment.
        
        Args:
            config_path: Path to config.json
            generator_type: 'pepmdlm' (SMILES) or 'pepdfm' (WT)
            generator_path: Path to generator model
            device: Device to run on
        """
        self.device = device
        self.generator_type = generator_type
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load generator
        print(f"Loading {generator_type.upper()} generator...")
        if generator_type == 'pepmdlm':
            if generator_path is None:
                generator_path = self.config.get('base_generator_path')
            self.generator = load_peptune_generator(generator_path, device=device)
            if self.generator.model is None:
                raise RuntimeError(f"Failed to load PepMDLM model from {generator_path}")
        elif generator_type == 'pepdfm':
            if generator_path is None:
                generator_path = self.config.get('dfm_model_path')
            self.generator = load_dfm_model(generator_path)
            if self.generator is None:
                raise NotImplementedError("PepDFM not yet implemented")
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")
        print(f"✓ Generator loaded")
        
        # Load predictors based on generator type
        # Mapping: WT → PepDFM, SMILES → PepMDLM
        print("Loading predictors...")
        format_type = 'wt' if generator_type == 'pepdfm' else 'smiles'
        self.predictors = load_predictors(
            self.config,
            format_type=format_type,
            device=device,
            protein_seq=self.config.get('protein_seq')
        )
        
        # Load language models (optional)
        self.text_encoder = None
        self.preference_net = None
        
        preference_net_path = self.config.get('preference_net_path')
        if preference_net_path:
            self.text_encoder = load_text_encoder(
                self.config.get('text_encoder_name', 'Qwen/Qwen3-Embedding-0.6B'),
                device=device
            )
            self.preference_net = load_preference_net(preference_net_path, device=device)
            print("✓ Language models loaded")
        else:
            print("⚠ No preference network found - will use predictor-only conditioning")
        
        # Define language conditions
        self.conditions = {
            'no_language': {
                'prompt': None,
                'description': 'Predictor-only conditioning (no language)'
            },
            'neutral': {
                'prompt': "Design a peptide with good binding and low hemolysis.",
                'description': 'Neutral language prompt'
            },
            'stability': {
                'prompt': "Design a peptide that is protease-resistant and stable in vivo, while maintaining good binding and low hemolysis.",
                'description': 'Stability-focused language prompt'
            }
        }
        
        # Predictor constraints (same across all conditions)
        self.constraints = {
            'strength': self.config.get('constraints', {}).get('strength', 1.0),
            'weights': self.config.get('constraints', {}).get('weights', {
                'binding': 1.0,
                'hemolysis': 1.0
            })
        }
    
    def optimize_peptide(
        self,
        starting_peptide: str,
        protein_target_id: str,
        num_optimization_steps: int = 50,
        num_samples: int = 1,
        condition: str = 'no_language',
        seed: Optional[int] = None
    ) -> Dict:
        """
        Optimize a starting peptide under specified condition.
        
        Args:
            starting_peptide: Initial peptide sequence (SMILES or WT)
            protein_target_id: Identifier for the protein target
            num_optimization_steps: Number of optimization steps
            num_samples: Number of samples to generate (for Algorithm 3)
            condition: Which language condition to use
            seed: Random seed for reproducibility
            
        Returns:
            Dict with optimization results
        """
        if condition not in self.conditions:
            raise ValueError(f"Unknown condition: {condition}. Choose from {list(self.conditions.keys())}")
        
        cond_info = self.conditions[condition]
        prompt = cond_info['prompt']
        
        print(f"\n{'='*60}")
        print(f"Optimizing peptide for target: {protein_target_id}")
        print(f"Condition: {condition} - {cond_info['description']}")
        print(f"Starting peptide: {starting_peptide[:50]}...")
        print(f"{'='*60}")
        
        # Evaluate starting peptide
        starting_scores = self._evaluate_peptide(starting_peptide)
        print(f"Starting scores:")
        print(f"  Binding: {starting_scores['binding']:.4f}")
        print(f"  Hemolysis: {starting_scores['hemolysis']:.4f}")
        
        # Generate optimized peptides
        optimized_peptides = []
        
        if num_samples > 1:
            # Use Algorithm 3: Generate multiple completions from fixed seed
            print(f"Generating {num_samples} optimized variants...")
            seed_set = [starting_peptide]
            
            completions = sample_from_fixed_seeds(
                base_generator=self.generator,
                seed_set=seed_set,
                prompt=prompt,
                predictors=self.predictors,
                constraints=self.constraints,
                text_encoder=self.text_encoder if prompt else None,
                preference_net=self.preference_net if prompt else None,
                num_steps=num_optimization_steps,
                use_linear_preferences=False,
                schedule=None,
                mask_rate=0.5,  # Mask 50% of the starting peptide
                mask_positions=None,
                completions_per_seed=num_samples,
                seed=seed
            )
            
            optimized_peptides = [comp['completion'] for comp in completions]
        else:
            # Use Algorithm 2: Single optimization run
            print("Generating optimized peptide...")
            try:
                optimized_peptide = sample_peptide(
                    base_generator=self.generator,
                    prompt=prompt,
                    predictors=self.predictors,
                    constraints=self.constraints,
                    text_encoder=self.text_encoder if prompt else None,
                    preference_net=self.preference_net if prompt else None,
                    num_steps=num_optimization_steps,
                    use_linear_preferences=False,
                    seed=seed
                )
                optimized_peptides = [optimized_peptide]
            except Exception as e:
                print(f"Error during optimization: {e}")
                optimized_peptides = []
        
        # Evaluate optimized peptides
        results = {
            'protein_target_id': protein_target_id,
            'condition': condition,
            'prompt': prompt,
            'starting_peptide': starting_peptide,
            'starting_scores': starting_scores,
            'optimized_peptides': [],
            'optimized_scores': []
        }
        
        for opt_peptide in optimized_peptides:
            opt_scores = self._evaluate_peptide(opt_peptide)
            results['optimized_peptides'].append(opt_peptide)
            results['optimized_scores'].append(opt_scores)
            
            print(f"\nOptimized peptide: {opt_peptide[:50]}...")
            print(f"  Binding: {opt_scores['binding']:.4f} (change: {opt_scores['binding'] - starting_scores['binding']:+.4f})")
            print(f"  Hemolysis: {opt_scores['hemolysis']:.4f} (change: {opt_scores['hemolysis'] - starting_scores['hemolysis']:+.4f})")
        
        return results
    
    def _evaluate_peptide(self, peptide: str) -> Dict[str, float]:
        """
        Evaluate a peptide with all predictors.
        
        Args:
            peptide: Peptide sequence (SMILES or WT)
            
        Returns:
            Dict with predictor scores
        """
        scores = {}
        
        # Convert to SMILES if needed (predictors expect SMILES)
        smiles_peptide = self._to_smiles(peptide)
        
        if smiles_peptide:
            for pred_name, predictor in self.predictors.items():
                try:
                    scores[pred_name] = predictor.predict(smiles_peptide)
                except Exception as e:
                    print(f"Warning: Failed to evaluate {pred_name}: {e}")
                    scores[pred_name] = np.nan
        else:
            # If conversion failed, try direct evaluation
            for pred_name, predictor in self.predictors.items():
                try:
                    scores[pred_name] = predictor.predict(peptide)
                except Exception:
                    scores[pred_name] = np.nan
        
        return scores
    
    def _detect_format(self, peptide: str) -> str:
        """Detect if peptide is WT or SMILES format."""
        return detect_sequence_format(peptide)
    
    def _to_smiles(self, peptide: str) -> str:
        """
        Convert peptide to SMILES format if needed.
        
        Args:
            peptide: Peptide in SMILES or WT format
            
        Returns:
            SMILES string (or original if conversion fails)
        """
        # Check if already SMILES (doesn't look like WT sequence)
        is_wt = all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in peptide.upper())
        
        if is_wt:
            # Try to convert WT to SMILES using RDKit
            try:
                from rdkit import Chem
                
                # Standard amino acid side chain SMILES
                aa_side_chains = {
                    'A': 'C', 'C': 'CS', 'D': 'CC(=O)O', 'E': 'CCC(=O)O',
                    'F': 'CC1=CC=CC=C1', 'G': '', 'H': 'CC1=CNC=N1',
                    'I': 'CC(C)CC', 'K': 'CCCCN', 'L': 'CC(C)CC',
                    'M': 'CCSC', 'N': 'CC(=O)N', 'P': 'CC1CCNC1',
                    'Q': 'CCC(=O)N', 'R': 'CCCCNC(=N)N', 'S': 'CO',
                    'T': 'CC(C)O', 'V': 'CC(C)C', 'W': 'CC1=CNC2=C1C=CC=C2',
                    'Y': 'CC1=CC=C(C=C1)O',
                }
                
                if not peptide:
                    return ""
                
                # Build peptide SMILES
                smiles_parts = []
                for i, aa in enumerate(peptide.upper()):
                    if aa not in aa_side_chains:
                        continue
                    
                    side_chain = aa_side_chains[aa]
                    
                    if i == 0:
                        # N-terminal
                        if side_chain:
                            smiles_parts.append(f"NC({side_chain})C(=O)")
                        else:
                            smiles_parts.append("NCC(=O)")
                    else:
                        # Middle residues
                        if side_chain:
                            smiles_parts.append(f"NC({side_chain})C(=O)")
                        else:
                            smiles_parts.append("NCC(=O)")
                
                # C-terminal
                if smiles_parts:
                    smiles = ''.join(smiles_parts[:-1])
                    if smiles_parts:
                        last_part = smiles_parts[-1]
                        smiles = smiles + last_part.replace("C(=O)", "C(=O)O", 1)
                    else:
                        smiles = smiles_parts[0].replace("C(=O)", "C(=O)O", 1)
                    
                    # Validate SMILES
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        return Chem.MolToSmiles(mol)
                
                # If conversion failed, return original
                print(f"Warning: Could not convert WT to SMILES, using original: {peptide[:30]}...")
                return peptide
                
            except ImportError:
                # RDKit not available - return original
                print("Warning: RDKit not available. Using original peptide.")
                return peptide
            except Exception as e:
                print(f"Warning: WT to SMILES conversion failed: {e}. Using original.")
                return peptide
        else:
            # Already SMILES
            return peptide
    
    def run_experiment(
        self,
        protein_target_id: str,
        starting_peptide: str,
        num_optimization_steps: int = 50,
        num_samples: int = 1,
        output_dir: Optional[str] = None,
        seed: Optional[int] = 42
    ) -> Dict:
        """
        Run the full controlled optimization experiment.
        
        Optimizes the starting peptide under all three conditions and saves results.
        
        Args:
            protein_target_id: Identifier for the protein target
            starting_peptide: Initial peptide sequence
            num_optimization_steps: Number of optimization steps
            num_samples: Number of samples per condition
            output_dir: Output directory for results
            seed: Random seed
            
        Returns:
            Dict with all results
        """
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"eval_results/controlled_optimization_{protein_target_id}_{timestamp}")
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print("CONTROLLED PEPTIDE OPTIMIZATION EXPERIMENT")
        print("="*80)
        print(f"Protein Target: {protein_target_id}")
        print(f"Starting Peptide: {starting_peptide}")
        print(f"Generator Type: {self.generator_type.upper()}")
        print(f"Optimization Steps: {num_optimization_steps}")
        print(f"Samples per Condition: {num_samples}")
        print(f"Output Directory: {output_dir}")
        print("="*80)
        
        all_results = {}
        
        # Run optimization under each condition
        for condition_name in self.conditions.keys():
            print(f"\n{'='*80}")
            print(f"CONDITION: {condition_name.upper()}")
            print(f"{'='*80}")
            
            results = self.optimize_peptide(
                starting_peptide=starting_peptide,
                protein_target_id=protein_target_id,
                num_optimization_steps=num_optimization_steps,
                num_samples=num_samples,
                condition=condition_name,
                seed=seed
            )
            
            all_results[condition_name] = results
        
        # Save results
        print(f"\n{'='*80}")
        print("Saving results...")
        print(f"{'='*80}")
        
        # Save detailed JSON
        results_json = {
            'protein_target_id': protein_target_id,
            'starting_peptide': starting_peptide,
            'generator_type': self.generator_type,
            'num_optimization_steps': num_optimization_steps,
            'num_samples': num_samples,
            'conditions': all_results
        }
        
        with open(output_dir / "results.json", 'w') as f:
            json.dump(results_json, f, indent=2)
        
        # Save summary table
        self._save_summary_table(all_results, output_dir)
        
        # Save individual peptide files
        self._save_peptides(all_results, output_dir)
        
        print(f"\n✓ Results saved to: {output_dir}")
        print(f"  - results.json: Full results")
        print(f"  - summary_table.csv: Summary table")
        print(f"  - peptides/: Individual peptide files")
        
        return all_results
    
    def _save_summary_table(self, results: Dict, output_dir: Path):
        """Save summary table comparing all conditions."""
        rows = []
        
        for condition_name, condition_results in results.items():
            starting_scores = condition_results['starting_scores']
            
            for i, (peptide, scores) in enumerate(zip(
                condition_results['optimized_peptides'],
                condition_results['optimized_scores']
            )):
                row = {
                    'condition': condition_name,
                    'sample_index': i,
                    'starting_binding': starting_scores['binding'],
                    'starting_hemolysis': starting_scores['hemolysis'],
                    'optimized_binding': scores['binding'],
                    'optimized_hemolysis': scores['hemolysis'],
                    'binding_improvement': scores['binding'] - starting_scores['binding'],
                    'hemolysis_improvement': starting_scores['hemolysis'] - scores['hemolysis'],  # Lower is better
                    'peptide': peptide
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / "summary_table.csv", index=False)
        
        # Also save LaTeX table
        latex_table = df.to_latex(index=False, float_format="%.4f", longtable=True)
        with open(output_dir / "summary_table.tex", 'w') as f:
            f.write(latex_table)
    
    def _save_peptides(self, results: Dict, output_dir: Path):
        """Save individual peptide sequences."""
        peptides_dir = output_dir / "peptides"
        peptides_dir.mkdir(exist_ok=True)
        
        for condition_name, condition_results in results.items():
            for i, peptide in enumerate(condition_results['optimized_peptides']):
                filename = peptides_dir / f"{condition_name}_sample_{i}.txt"
                with open(filename, 'w') as f:
                    f.write(peptide)
        
        # Also save starting peptide
        starting_peptide = results[list(results.keys())[0]]['starting_peptide']
        with open(peptides_dir / "starting_peptide.txt", 'w') as f:
            f.write(starting_peptide)


def main():
    parser = argparse.ArgumentParser(
        description="Controlled peptide optimization experiment"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config.json'
    )
    parser.add_argument(
        '--protein_target_id',
        type=str,
        required=True,
        help='Protein target identifier (e.g., "P12345")'
    )
    parser.add_argument(
        '--starting_peptide',
        type=str,
        required=True,
        help='Starting peptide sequence (SMILES or WT amino acids)'
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
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: eval_results/controlled_optimization_<target>_<timestamp>)'
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
    
    args = parser.parse_args()
    
    # Initialize experiment
    experiment = ControlledOptimizationExperiment(
        config_path=args.config,
        generator_type=args.generator_type,
        generator_path=args.generator_path,
        device=args.device
    )
    
    # Run experiment
    results = experiment.run_experiment(
        protein_target_id=args.protein_target_id,
        starting_peptide=args.starting_peptide,
        num_optimization_steps=args.num_steps,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    print("\n" + "="*80)
    print("Experiment complete!")
    print("="*80)


if __name__ == '__main__':
    main()

