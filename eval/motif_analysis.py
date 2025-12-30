"""
Evaluation: Language-Guided Control of Unlabeled Design Objectives

This module implements experiments from Section 4.3 to test whether LaPep can express
and enforce design intent for properties lacking reliable labeled predictors.
"""

import numpy as np
from typing import List, Dict, Optional, Set
import re
from collections import Counter, defaultdict

from lapep.sampler import sample_peptide


# Common peptide motifs and patterns from literature
PROTEASE_RESISTANCE_MOTIFS = [
    r'[P][^P]',  # Proline-containing patterns
    r'[D][E]',   # Acidic residues
    r'[K][R]',   # Basic residues
    r'[A][A]',   # Alanine repeats
]

STABILITY_MOTIFS = [
    r'[C][C]',   # Disulfide bonds (Cys-Cys)
    r'[G][G]',   # Glycine flexibility
    r'[P].*[P]', # Proline-rich
]

MODIFICATION_PATTERNS = [
    r'[N][^P][S|T]',  # N-linked glycosylation sites
    r'[S|T].*[P]',    # Phosphorylation sites
]


def evaluate_unlabeled_objective_control(
    base_generator,
    text_encoder,
    preference_net,
    predictors: Dict,
    prompts: Dict[str, str],
    num_samples: int = 500,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate language-guided control for properties without labeled predictors.
    
    Args:
        base_generator: Base generator
        text_encoder: Frozen text encoder
        preference_net: Trained preference network
        predictors: Dict of available predictors
        prompts: Dict mapping prompt types to prompt strings
        num_samples: Number of samples per prompt
        seed: Random seed
        
    Returns:
        Dict mapping prompt types to evaluation metrics
    """
    np.random.seed(seed)
    
    results = {}
    
    # Generate baseline (no language conditioning)
    print("Generating baseline samples (no language)...")
    baseline_samples = _generate_baseline(
        base_generator,
        predictors,
        num_samples
    )
    baseline_metrics = _compute_unlabeled_metrics(baseline_samples)
    results['no_language'] = baseline_metrics
    
    # Generate samples for each prompt type
    for prompt_type, prompt_text in prompts.items():
        print(f"Generating samples for prompt: {prompt_type}")
        samples = _generate_conditioned(
            base_generator,
            text_encoder,
            preference_net,
            predictors,
            prompt_text,
            num_samples
        )
        metrics = _compute_unlabeled_metrics(samples, prompt_type)
        results[prompt_type] = metrics
    
    return results


def _generate_baseline(
    base_generator,
    predictors: Dict,
    num_samples: int
) -> List[str]:
    """Generate samples without language conditioning."""
    samples = []
    for _ in range(num_samples):
        peptide = sample_peptide(
            base_generator,
            prompt=None,
            predictors=predictors,
            constraints=None,
            text_encoder=None,
            preference_net=None
        )
        samples.append(peptide)
    return samples


def _generate_conditioned(
    base_generator,
    text_encoder,
    preference_net,
    predictors: Dict,
    prompt: str,
    num_samples: int
) -> List[str]:
    """Generate samples with language conditioning."""
    samples = []
    for _ in range(num_samples):
        peptide = sample_peptide(
            base_generator,
            prompt=prompt,
            predictors=predictors,
            constraints=None,
            text_encoder=text_encoder,
            preference_net=preference_net
        )
        samples.append(peptide)
    return samples


def _compute_unlabeled_metrics(
    samples: List[str],
    prompt_type: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute metrics for unlabeled objectives based on structural patterns.
    
    Metrics include:
    - Motif enrichment: frequency of relevant motifs
    - Modification rate: frequency of modification sites
    - Heuristic score: composite score based on literature patterns
    """
    metrics = {}
    
    # Determine relevant motifs based on prompt type
    if prompt_type and 'protease' in prompt_type.lower():
        motifs = PROTEASE_RESISTANCE_MOTIFS
        modification_patterns = []
    elif prompt_type and 'stability' in prompt_type.lower():
        motifs = STABILITY_MOTIFS
        modification_patterns = MODIFICATION_PATTERNS
    else:
        motifs = PROTEASE_RESISTANCE_MOTIFS + STABILITY_MOTIFS
        modification_patterns = MODIFICATION_PATTERNS
    
    # Compute motif enrichment
    motif_counts = _count_motifs(samples, motifs)
    total_motifs = sum(motif_counts.values())
    motif_enrichment = total_motifs / len(samples) if samples else 0.0
    metrics['motif_enrichment'] = motif_enrichment
    
    # Compute modification rate
    mod_counts = _count_modification_sites(samples, modification_patterns)
    total_mods = sum(mod_counts.values())
    modification_rate = total_mods / len(samples) if samples else 0.0
    metrics['modification_rate'] = modification_rate
    
    # Compute heuristic score (composite metric)
    heuristic_score = _compute_heuristic_score(samples, prompt_type)
    metrics['heuristic_score'] = heuristic_score
    
    # Additional detailed metrics
    metrics['avg_sequence_length'] = np.mean([len(s) for s in samples])
    metrics['unique_sequences'] = len(set(samples)) / len(samples) if samples else 0.0
    
    return metrics


def _count_motifs(
    samples: List[str],
    motifs: List[str]
) -> Dict[str, int]:
    """Count occurrences of motifs in samples."""
    counts = Counter()
    
    # Convert SMILES to amino acid sequences for motif analysis
    sequences = _smiles_to_sequences(samples)
    
    for seq in sequences:
        if seq:  # Only process if conversion succeeded
            for motif_pattern in motifs:
                matches = re.findall(motif_pattern, seq)
                counts[motif_pattern] += len(matches)
    
    return counts


def _count_modification_sites(
    samples: List[str],
    patterns: List[str]
) -> Dict[str, int]:
    """Count modification sites in samples."""
    counts = Counter()
    
    # Convert SMILES to amino acid sequences for modification site analysis
    sequences = _smiles_to_sequences(samples)
    
    for seq in sequences:
        if seq:  # Only process if conversion succeeded
            for pattern in patterns:
                matches = re.findall(pattern, seq)
                counts[pattern] += len(matches)
    
    return counts


def _compute_heuristic_score(
    samples: List[str],
    prompt_type: Optional[str]
) -> float:
    """
    Compute a composite heuristic score based on literature patterns.
    
    This is a proxy metric for properties that lack labeled predictors.
    """
    scores = []
    
    # Convert SMILES to amino acid sequences
    sequences = _smiles_to_sequences(samples)
    
    for seq in sequences:
        if not seq:  # Skip if conversion failed
            scores.append(0.0)
            continue
            
        score = 0.0
        
        # Length-based heuristics (on amino acid sequence length)
        if 8 <= len(seq) <= 30:  # Typical therapeutic peptide length
            score += 0.2
        
        # Charge-based (simple approximation)
        positive_charges = seq.count('K') + seq.count('R')
        negative_charges = seq.count('D') + seq.count('E')
        net_charge = positive_charges - negative_charges
        if -2 <= net_charge <= 2:  # Neutral to slightly charged
            score += 0.2
        
        # Hydrophobicity (simple approximation)
        hydrophobic = seq.count('A') + seq.count('V') + seq.count('I') + \
                     seq.count('L') + seq.count('M') + seq.count('F')
        hydrophobic_ratio = hydrophobic / len(seq) if seq else 0
        if 0.2 <= hydrophobic_ratio <= 0.5:  # Moderate hydrophobicity
            score += 0.2
        
        # Proline content (stability)
        proline_ratio = seq.count('P') / len(seq) if seq else 0
        if 0.05 <= proline_ratio <= 0.15:
            score += 0.2
        
        # Cysteine content (potential for disulfide bonds)
        cys_ratio = seq.count('C') / len(seq) if seq else 0
        if cys_ratio > 0 and cys_ratio <= 0.1:
            score += 0.2
        
        scores.append(score)
    
    return np.mean(scores) if scores else 0.0


def _smiles_to_sequences(samples: List[str]) -> List[str]:
    """
    Convert SMILES strings to amino acid sequences.
    
    Returns list of one-letter amino acid sequences, or empty strings if conversion fails.
    """
    sequences = []
    
    # Try to import PeptideAnalyzer
    try:
        import sys
        from pathlib import Path
        lapep_root = Path(__file__).parent.parent
        tr2d2_utils_path = lapep_root / "lapep" / "tr2d2" / "utils"
        
        if (tr2d2_utils_path / "app.py").exists():
            sys.path.insert(0, str(tr2d2_utils_path.parent))
            from utils.app import PeptideAnalyzer
            analyzer = PeptideAnalyzer()
            
            for smiles in samples:
                try:
                    # Use return_sequence to get amino acid sequence
                    seq_list = analyzer.return_sequence(smiles)
                    # Convert three-letter codes to one-letter codes
                    one_letter = ''.join(
                        analyzer.three_to_one.get(aa.split('(')[0], 'X') 
                        for aa in seq_list
                    )
                    sequences.append(one_letter)
                except Exception as e:
                    # If conversion fails, return empty string
                    sequences.append("")
        else:
            # If analyzer not available, return empty sequences
            sequences = [""] * len(samples)
    except Exception as e:
        # If import fails, return empty sequences
        sequences = [""] * len(samples)
    
    return sequences


def compare_to_baseline(
    baseline_metrics: Dict[str, float],
    conditioned_metrics: Dict[str, float]
) -> Dict[str, Dict[str, float]]:
    """
    Compare conditioned samples to baseline.
    
    Returns:
        Dict with relative improvements and effect sizes
    """
    comparison = {}
    
    for metric_name in baseline_metrics:
        if metric_name in conditioned_metrics:
            baseline_val = baseline_metrics[metric_name]
            conditioned_val = conditioned_metrics[metric_name]
            
            if baseline_val > 0:
                relative_change = (conditioned_val - baseline_val) / baseline_val
            else:
                relative_change = 0.0 if conditioned_val == 0 else float('inf')
            
            comparison[metric_name] = {
                'baseline': baseline_val,
                'conditioned': conditioned_val,
                'relative_change': relative_change,
                'absolute_change': conditioned_val - baseline_val
            }
    
    return comparison


def format_unlabeled_table(
    results: Dict[str, Dict[str, float]]
) -> str:
    """
    Format unlabeled objective results as LaTeX table matching Table 3 in the paper.
    """
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Evaluation of language-guided control for unlabeled objectives.}",
        "\\label{tab:unlabeled_control}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Prompt Type} & \\textbf{Motif Enrichment} & ",
        "\\textbf{Modification Rate} & \\textbf{Heuristic Score} \\\\",
        "\\midrule"
    ]
    
    prompt_display_names = {
        'no_language': 'No Language Conditioning',
        'protease_resistance': 'Protease-Resistance Prompt',
        'stability_oriented': 'Stability-Oriented Prompt'
    }
    
    for prompt_key, prompt_display in prompt_display_names.items():
        if prompt_key in results:
            metrics = results[prompt_key]
            motif = metrics.get('motif_enrichment', 0.0)
            mod_rate = metrics.get('modification_rate', 0.0)
            heuristic = metrics.get('heuristic_score', 0.0)
            
            lines.append(
                f"{prompt_display} & {motif:.3f} & {mod_rate:.3f} & {heuristic:.3f} \\\\"
            )
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)

