"""
Evaluation: Stability and Path Independence of Conditioned Dynamics

This module implements experiments from Section 4.2 to test whether LaPep's conservative
conditioning yields path-independent behavior and avoids unstable dynamics.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import torch
from collections import defaultdict

from ..lapep.kernel import compute_transition_kernel
from ..lapep.potential import compute_potential


def evaluate_path_independence(
    base_generator,
    text_encoder,
    preference_net,
    predictors: Dict,
    prompt: str,
    num_cycles: int = 1000,
    cycle_length: int = 4,
    seed: int = 42
) -> Dict[str, float]:
    """
    Evaluate path independence by measuring cycle circulation.
    
    Tests Theorem 4.2: circulation should be zero (or equal to base generator circulation)
    for conservative conditioning.
    
    Args:
        base_generator: Base generator with proposal kernel
        text_encoder: Frozen text encoder
        preference_net: Trained preference network
        predictors: Dict of predictor objects
        prompt: Natural language prompt
        num_cycles: Number of random cycles to sample
        cycle_length: Length of cycles to test
        seed: Random seed
        
    Returns:
        Dict with circulation statistics
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    circulations = []
    
    # Sample random cycles from the edit graph
    for _ in range(num_cycles):
        cycle = _sample_random_cycle(base_generator, cycle_length)
        if cycle is None:
            continue
        
        # Compute circulation for LaPep kernel
        circ = _compute_cycle_circulation(
            cycle,
            base_generator,
            text_encoder,
            preference_net,
            predictors,
            prompt
        )
        circulations.append(circ)
    
    circulations = np.array(circulations)
    
    return {
        'mean_circulation': float(np.mean(circulations)),
        'std_circulation': float(np.std(circulations)),
        'max_circulation': float(np.max(np.abs(circulations))),
        'min_circulation': float(np.min(circulations)),
        'zero_circulation_ratio': float(np.mean(np.abs(circulations) < 1e-6)),
        'circulations': circulations.tolist()
    }


def _sample_random_cycle(
    base_generator,
    length: int
) -> Optional[List[str]]:
    """
    Sample a random cycle of specified length from the edit graph.
    
    Returns a list of states [x0, x1, ..., x_{n-1}, x0] forming a cycle.
    """
    # Start from a random state
    x0 = base_generator.sample_initial_state()
    cycle = [x0]
    current = x0
    
    for _ in range(length):
        # Get neighbors
        neighbors = base_generator.get_neighbors(current)
        if len(neighbors) == 0:
            return None
        
        # Random walk to next state
        next_state = neighbors[np.random.randint(len(neighbors))]
        cycle.append(next_state)
        current = next_state
    
    # Try to close the cycle (return to x0 or a neighbor of x0)
    if current == x0:
        return cycle
    
    # Check if we can close in one step
    neighbors = base_generator.get_neighbors(current)
    if x0 in neighbors:
        cycle.append(x0)
        return cycle
    
    # If not, try to find a path back
    # For simplicity, we'll just check if current is a neighbor of x0
    # In practice, might need BFS for longer cycles
    return None


def _compute_cycle_circulation(
    cycle: List[str],
    base_generator,
    text_encoder,
    preference_net,
    predictors: Dict,
    prompt: str
) -> float:
    """
    Compute circulation around a cycle using the LaPep edge flow.
    
    Circulation = sum of edge flows F(x_i, x_{i+1}) around the cycle.
    For conservative flows, this should be zero.
    """
    from ..lapep.kernel import compute_edge_flow
    
    total_flow = 0.0
    
    for i in range(len(cycle) - 1):
        x = cycle[i]
        x_next = cycle[i + 1]
        
        # Compute edge flow F(x, x')
        flow = compute_edge_flow(
            x, x_next,
            base_generator,
            text_encoder,
            preference_net,
            predictors,
            prompt,
            tau=0  # Use a fixed time step for consistency
        )
        total_flow += flow
    
    return total_flow


def evaluate_path_variance(
    base_generator,
    text_encoder,
    preference_net,
    predictors: Dict,
    prompt: str,
    num_trajectories: int = 100,
    trajectory_length: int = 50,
    initial_state: Optional[str] = None,
    seed: int = 42
) -> Dict[str, float]:
    """
    Evaluate path variance by sampling multiple trajectories from the same initial state.
    
    For path-independent dynamics, trajectories should converge to similar distributions
    regardless of the specific path taken.
    
    Args:
        base_generator: Base generator
        text_encoder: Frozen text encoder
        preference_net: Trained preference network
        predictors: Dict of predictors
        prompt: Natural language prompt
        num_trajectories: Number of independent trajectories
        trajectory_length: Length of each trajectory
        initial_state: Starting state (if None, sample from prior)
        seed: Random seed
        
    Returns:
        Dict with path variance statistics
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if initial_state is None:
        initial_state = base_generator.sample_initial_state()
    
    trajectories = []
    
    for _ in range(num_trajectories):
        trajectory = _sample_trajectory(
            base_generator,
            text_encoder,
            preference_net,
            predictors,
            prompt,
            initial_state,
            trajectory_length
        )
        trajectories.append(trajectory)
    
    # Compute variance in final states
    final_states = [traj[-1] for traj in trajectories]
    
    # Convert to predictor coordinates for comparison
    final_coords = []
    for state in final_states:
        coords = _state_to_predictor_coords(state, predictors)
        final_coords.append(coords)
    
    final_coords = np.array(final_coords)
    
    # Compute variance across trajectories
    coord_variance = np.var(final_coords, axis=0)
    
    return {
        'mean_final_variance': float(np.mean(coord_variance)),
        'max_final_variance': float(np.max(coord_variance)),
        'trajectory_diversity': float(_compute_trajectory_diversity(trajectories)),
        'final_state_coords': final_coords.tolist()
    }


def _sample_trajectory(
    base_generator,
    text_encoder,
    preference_net,
    predictors: Dict,
    prompt: str,
    initial_state: str,
    length: int
) -> List[str]:
    """Sample a single trajectory using LaPep conditioning."""
    from ..lapep.sampler import sample_step
    
    trajectory = [initial_state]
    current = initial_state
    
    for step in range(length):
        current = sample_step(
            current,
            base_generator,
            text_encoder,
            preference_net,
            predictors,
            prompt,
            tau=step
        )
        trajectory.append(current)
    
    return trajectory


def _state_to_predictor_coords(
    state: str,
    predictors: Dict
) -> np.ndarray:
    """Convert a state to normalized predictor coordinates."""
    coords = []
    for pred_name, predictor in predictors.items():
        raw = predictor.predict(state)
        normalized = predictor.normalize(raw)
        coords.append(normalized)
    return np.array(coords)


def _compute_trajectory_diversity(
    trajectories: List[List[str]]
) -> float:
    """
    Compute diversity metric across trajectories.
    
    Uses average pairwise edit distance between final states.
    """
    final_states = [traj[-1] for traj in trajectories]
    
    if len(final_states) < 2:
        return 0.0
    
    distances = []
    for i in range(len(final_states)):
        for j in range(i + 1, len(final_states)):
            dist = _edit_distance(final_states[i], final_states[j])
            distances.append(dist)
    
    return np.mean(distances) if distances else 0.0


def _edit_distance(s1: str, s2: str) -> int:
    """Compute edit distance between two sequences."""
    # Simple token-level edit distance
    tokens1 = s1.split() if isinstance(s1, str) else list(s1)
    tokens2 = s2.split() if isinstance(s2, str) else list(s2)
    
    m, n = len(tokens1), len(tokens2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens1[i-1] == tokens2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]


def compare_conditioning_methods(
    base_generator,
    text_encoder,
    preference_net,
    predictors: Dict,
    prompt: str,
    num_cycles: int = 1000
) -> Dict[str, Dict[str, float]]:
    """
    Compare LaPep to non-conservative baselines.
    
    Returns:
        Dict mapping method names to circulation statistics
    """
    results = {}
    
    # Unconditioned generator
    print("Evaluating unconditioned generator...")
    results['unconditioned'] = evaluate_path_independence(
        base_generator,
        None, None, {}, None,
        num_cycles=num_cycles
    )
    
    # Predictor-only guidance (conservative)
    print("Evaluating predictor-only guidance...")
    results['predictor_only'] = evaluate_path_independence(
        base_generator,
        None, None, predictors, None,
        num_cycles=num_cycles
    )
    
    # Naive language guidance (non-conservative baseline)
    print("Evaluating naive language guidance...")
    results['naive_language'] = _evaluate_naive_guidance(
        base_generator,
        text_encoder,
        preference_net,
        predictors,
        prompt,
        num_cycles
    )
    
    # LaPep (conservative)
    print("Evaluating LaPep...")
    results['lapep'] = evaluate_path_independence(
        base_generator,
        text_encoder,
        preference_net,
        predictors,
        prompt,
        num_cycles=num_cycles
    )
    
    return results


def _evaluate_naive_guidance(
    base_generator,
    text_encoder,
    preference_net,
    predictors: Dict,
    prompt: str,
    num_cycles: int
) -> Dict[str, float]:
    """
    Evaluate a non-conservative baseline that applies language guidance directly
    to token logits without the conservative reweighting.
    """
    # This would implement a naive guidance scheme
    # For now, return placeholder
    return {
        'mean_circulation': 0.0,
        'std_circulation': 0.0,
        'max_circulation': 0.0,
        'min_circulation': 0.0,
        'zero_circulation_ratio': 0.0
    }


def format_circulation_table(
    results: Dict[str, Dict[str, float]]
) -> str:
    """
    Format circulation results as LaTeX table matching Table 2 in the paper.
    """
    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\caption{Path dependence diagnostics for conditioned generators.}",
        "\\label{tab:path_independence}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{Conditioning Method} & \\textbf{Mean Circulation} & ",
        "\\textbf{Max Circulation} & \\textbf{Path Variance} \\\\",
        "\\midrule"
    ]
    
    method_names = {
        'unconditioned': 'Unconditioned Generator',
        'predictor_only': 'Predictor-only Guidance',
        'naive_language': 'Naive Language Guidance',
        'lapep': '\\textbf{LaPep (Ours)}'
    }
    
    for method_key, method_display in method_names.items():
        if method_key in results:
            stats = results[method_key]
            mean_circ = stats.get('mean_circulation', 0.0)
            max_circ = stats.get('max_circulation', 0.0)
            path_var = stats.get('mean_final_variance', 0.0)
            
            lines.append(
                f"{method_display} & {mean_circ:.6f} & {max_circ:.6f} & {path_var:.6f} \\\\"
            )
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)

