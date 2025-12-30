"training script for LaPep preference network"

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import json
import sys
from pathlib import Path
import time
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        if desc:
            print(desc)
        return iterable

from language.text_encoder import load_text_encoder
from language.preference_net import PreferenceNet
from predictors.loader import load_predictors


def sample_peptide_batch(
    base_generator,
    batch_size: int,
    seed: Optional[int] = None
) -> List[str]:
    """sample batch of peptides from frozen generator."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    print(f"Sampling {batch_size} peptides...")
    peptides = base_generator.sample_unconditioned(num_samples=batch_size)
    
    # If we got fewer peptides than requested, try sampling more
    if len(peptides) < batch_size:
        print(f"  Warning: Only got {len(peptides)} peptides, sampling additional ones...")
        max_retries = 10
        retry_count = 0
        while len(peptides) < batch_size and retry_count < max_retries:
            additional = base_generator.sample_unconditioned(num_samples=batch_size - len(peptides))
            peptides.extend(additional)
            retry_count += 1
        
        # If still not enough, truncate to what we have (better than invalid placeholders)
        if len(peptides) < batch_size:
            print(f"  Warning: Could only sample {len(peptides)} peptides (requested {batch_size})")
    elif len(peptides) > batch_size:
        peptides = peptides[:batch_size]
    
    print(f"Successfully sampled {len(peptides)} peptides")
    return peptides


def score_peptides_with_predictors(
    peptides: List[str],
    predictors: Dict
) -> np.ndarray:
    """score each peptide with frozen predictors."""
    scores = []
    for peptide in peptides:
        peptide_scores = []
        for pred_name, predictor in predictors.items():
            # Get raw prediction
            raw_value = predictor.predict(peptide)
            # Normalize to [0, 1] using empirical CDF
            normalized = predictor.normalize(raw_value)
            peptide_scores.append(normalized)
        scores.append(peptide_scores)
    return np.array(scores, dtype=np.float32)


def create_pairwise_comparisons_rule_based(
    peptides: List[str],
    predictor_scores: np.ndarray,
    prompt: str,
    predictors: Dict
) -> List[Tuple[str, str, int]]:
    """
    Create pairwise comparisons using simple rules in predictor space.
    
    Args:
        peptides: List of peptide SMILES strings
        predictor_scores: Array of shape (batch_size, num_predictors)
        prompt: Natural language prompt
        predictors: Dict of predictor objects (for names)
        
    Returns:
        List of (peptide_a, peptide_b, y_ab) tuples where y_ab=1 if a preferred to b
    """
    comparisons = []
    pred_names = list(predictors.keys())
    # Simple rule-based preferences based on prompt keywords
    prompt_lower = prompt.lower()
    # Determine which predictors to prioritize based on prompt
    weights = {}
    if 'binding' in prompt_lower or 'affinity' in prompt_lower:
        weights['binding'] = 1.0
    if 'toxicity' in prompt_lower or 'toxic' in prompt_lower:
        weights['toxicity'] = -1.0  # Negative because lower is better
    if 'stability' in prompt_lower or 'half' in prompt_lower or 'half-life' in prompt_lower:
        weights['halflife'] = 1.0
    
    # Default: prefer higher binding, lower toxicity, higher half-life
    if not weights:
        weights = {'binding': 1.0, 'toxicity': -1.0, 'halflife': 1.0}
    
    # Create all pairs
    for i in range(len(peptides)):
        for j in range(i + 1, len(peptides)):
            x_a, x_b = peptides[i], peptides[j]
            u_a = predictor_scores[i]
            u_b = predictor_scores[j]
            
            # Compute weighted score
            score_a = 0.0
            score_b = 0.0
            
            for idx, pred_name in enumerate(pred_names):
                weight = weights.get(pred_name, 0.0)
                score_a += weight * u_a[idx]
                score_b += weight * u_b[idx]
            
            # Determine preference
            if abs(score_a - score_b) < 1e-6:
                # Scores are equal - randomly assign to ensure comparisons are generated
                if np.random.random() > 0.5:
                    comparisons.append((x_a, x_b, 1))
                else:
                    comparisons.append((x_b, x_a, 1))
            elif score_a > score_b:
                comparisons.append((x_a, x_b, 1))
            else:
                comparisons.append((x_b, x_a, 1))
    
    return comparisons


def create_pairwise_comparisons_llm_judge(
    peptides: List[str],
    predictor_scores: np.ndarray,
    prompt: str,
    llm_judge: Optional[Callable] = None
) -> List[Tuple[str, str, int]]:
    """
    Create pairwise comparisons using an LLM-based judge.
    
    Args:
        peptides: List of peptide SMILES strings
        predictor_scores: Array of shape (batch_size, num_predictors)
        prompt: Natural language prompt
        llm_judge: Optional LLM judge function (peptide_a, peptide_b, prompt) -> int
        
    Returns:
        List of (peptide_a, peptide_b, y_ab) tuples
    """
    if llm_judge is None:
        return []
    
    comparisons = []

    # Create all pairs
    for i in range(len(peptides)):
        for j in range(i + 1, len(peptides)):
            x_a, x_b = peptides[i], peptides[j]
            
            # Use LLM judge to determine preference
            preference = llm_judge(x_a, x_b, prompt)
            # preference should be 1 if x_a preferred, -1 if x_b preferred, 0 if equal
            
            if preference > 0:
                comparisons.append((x_a, x_b, 1))
            elif preference < 0:
                comparisons.append((x_b, x_a, 1))
            # If equal, skip
    
    return comparisons


def compute_pairwise_ranking_loss(
    preference_net: PreferenceNet,
    text_encoder,
    u_a: torch.Tensor,
    u_b: torch.Tensor,
    prompt_embedding: torch.Tensor,
    y_ab: torch.Tensor,
    use_linear: bool = False,
    eta: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute pairwise ranking loss.
    
    L = -y_ab * log(σ(R(x_a;t) - R(x_b;t))) - (1-y_ab) * log(σ(R(x_b;t) - R(x_a;t)))
    
    where R(x;t) = G_η(t)(u(x)) and η(t) = g_ψ(E_text(t))
    
    Args:
        preference_net: MLP g_ψ that maps text embeddings to preference parameters
        text_encoder: Frozen text encoder
        u_a: Predictor coordinates for peptide a, shape (batch_size, num_predictors)
        u_b: Predictor coordinates for peptide b, shape (batch_size, num_predictors)
        prompt_embedding: Text embeddings from frozen encoder, shape (batch_size, embedding_dim)
        y_ab: Preference labels, shape (batch_size,) with values in {0, 1}
        use_linear: If True, use linear preference functional R(x;t) = η^T u
        eta: Optional pre-computed preference parameters η(i) = g_ψ(e(i))
    
    Returns:
        Scalar loss value
    """
    # Pass embedding through MLP to get preference parameters η(i) = g_ψ(e(i))
    if eta is None:
        eta = preference_net(prompt_embedding)  # (batch_size, output_dim)
    
    # Compute preference scores R(x;t) = G_η(t)(u(x))
    if use_linear:
        # Linear: R(x;t) = η^T u
        if eta.shape[-1] == u_a.shape[-1]:
            R_a = torch.sum(eta * u_a, dim=-1)  # (batch_size,)
            R_b = torch.sum(eta * u_b, dim=-1)  # (batch_size,)
        else:
            # Project to match predictor dimension
            min_dim = min(eta.shape[-1], u_a.shape[-1])
            R_a = torch.sum(eta[:, :min_dim] * u_a[:, :min_dim], dim=-1)
            R_b = torch.sum(eta[:, :min_dim] * u_b[:, :min_dim], dim=-1)
    else:
        # In practice, this could be a learned neural network over u but using simple MLP-like transformation for now
        R_a = apply_nonlinear_preference(u_a, eta)
        R_b = apply_nonlinear_preference(u_b, eta)
    
    # Compute score difference
    diff = R_a - R_b  # (batch_size,)
    
    # Algorithm Line 28: Compute pairwise preference loss
    # L_pref = -y_ab log σ(R(x_a;t) – R(x_b;t)) – (1-y_ab) log σ(R(x_b;t) – R(x_a;t))
    loss = -y_ab * torch.log(torch.sigmoid(diff) + 1e-8) - \
           (1 - y_ab) * torch.log(torch.sigmoid(-diff) + 1e-8)
    return loss.mean()


def apply_nonlinear_preference(
    u: torch.Tensor,
    eta: torch.Tensor
) -> torch.Tensor:
    """
    Apply nonlinear preference functional G_η(u) -> right now this is a quadratic formed with learned parameters - in practice, this could be a learned neural network.

    Args:
        u: Predictor coordinates, shape (batch_size, num_predictors)
        eta: Preference parameters, shape (batch_size, output_dim)
    
    Returns:
        Preference scores, shape (batch_size,)
    """
    # Simple nonlinear transformation
    # Use first half of eta for quadratic terms, second half for linear
    num_preds = u.shape[-1]
    eta_dim = eta.shape[-1]
    
    if eta_dim >= 2 * num_preds:
        # Quadratic form: u^T diag(eta_quad) u + eta_linear^T u
        eta_quad = eta[:, :num_preds]
        eta_linear = eta[:, num_preds:2*num_preds]
        quadratic = torch.sum(eta_quad * u * u, dim=-1)
        linear = torch.sum(eta_linear * u, dim=-1)
        return quadratic + linear
    else:
        # fallback to linear
        print("falling back to linear")
        min_dim = min(eta_dim, num_preds)
        return torch.sum(eta[:, :min_dim] * u[:, :min_dim], dim=-1)

def create_comprehensive_plots(training_metrics: Dict, output_dir: Path, predictors: Dict):
    """visualization of training progress and distribution changes."""
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. epoch loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(training_metrics["epochs"], training_metrics["losses"], 'b-', linewidth=2, marker='o')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Average Loss', fontsize=11)
    ax1.set_title('Training Loss (Epoch-level)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 2. batch loss (smoothed)
    ax2 = fig.add_subplot(gs[0, 1])
    if len(training_metrics["batch_losses"]) > 0:
        batch_losses = training_metrics["batch_losses"]
        window = max(1, len(batch_losses) // 100)
        if window > 1:
            smoothed = pd.Series(batch_losses).rolling(window=window, center=True).mean()
            ax2.plot(range(len(batch_losses)), batch_losses, 'lightblue', alpha=0.2, label='Raw')
            ax2.plot(range(len(smoothed)), smoothed, 'b-', linewidth=2, label='Smoothed')
        else:
            ax2.plot(range(len(batch_losses)), batch_losses, 'b-', linewidth=1, alpha=0.5)
        ax2.set_xlabel('Batch', fontsize=11)
        ax2.set_ylabel('Loss', fontsize=11)
        ax2.set_title('Training Loss (Batch-level)', fontsize=12)
        if window > 1:
            ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. epoch times
    ax3 = fig.add_subplot(gs[0, 2])
    if len(training_metrics["epoch_times"]) > 0:
        ax3.bar(training_metrics["epochs"], training_metrics["epoch_times"], alpha=0.7, color='green')
        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('Time (seconds)', fontsize=11)
        ax3.set_title('Epoch Duration', fontsize=12)
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. predictor score distributions over time
    ax4 = fig.add_subplot(gs[1, 0])
    if len(training_metrics["predictor_score_distributions"]) > 0:
        for pred_name in predictors.keys():
            epochs = [d["epoch"] for d in training_metrics["predictor_score_distributions"]]
            means = [d["stats"][pred_name]["mean"] for d in training_metrics["predictor_score_distributions"]]
            stds = [d["stats"][pred_name]["std"] for d in training_metrics["predictor_score_distributions"]]
            ax4.plot(epochs, means, marker='o', label=pred_name, linewidth=2)
            ax4.fill_between(epochs, 
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            alpha=0.2)
        ax4.set_xlabel('Epoch', fontsize=11)
        ax4.set_ylabel('Mean Score', fontsize=11)
        ax4.set_title('Predictor Score Distributions Over Time', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. preference score distribution changes
    ax5 = fig.add_subplot(gs[1, 1])
    if len(training_metrics["preference_score_distributions"]) > 0:
        pref_epochs = [d["epoch"] for d in training_metrics["preference_score_distributions"]]
        pref_means = [d["mean"] for d in training_metrics["preference_score_distributions"]]
        pref_stds = [d["std"] for d in training_metrics["preference_score_distributions"]]
        pref_mins = [d["min"] for d in training_metrics["preference_score_distributions"]]
        pref_maxs = [d["max"] for d in training_metrics["preference_score_distributions"]]
        ax5.plot(pref_epochs, pref_means, 'g-', marker='o', linewidth=2, label='Mean')
        ax5.fill_between(pref_epochs, pref_mins, pref_maxs, alpha=0.2, label='Min-Max')
        ax5.fill_between(pref_epochs, 
                        [m - s for m, s in zip(pref_means, pref_stds)],
                        [m + s for m, s in zip(pref_means, pref_stds)],
                        alpha=0.3, label='±1 std')
        ax5.set_xlabel('Epoch', fontsize=11)
        ax5.set_ylabel('Preference Score', fontsize=11)
        ax5.set_title('Preference Score Distribution Changes', fontsize=12)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. predictor score ranges over time
    ax6 = fig.add_subplot(gs[1, 2])
    if len(training_metrics["predictor_score_distributions"]) > 0:
        for pred_name in predictors.keys():
            epochs = [d["epoch"] for d in training_metrics["predictor_score_distributions"]]
            mins = [d["stats"][pred_name]["min"] for d in training_metrics["predictor_score_distributions"]]
            maxs = [d["stats"][pred_name]["max"] for d in training_metrics["predictor_score_distributions"]]
            ranges = [mx - mn for mx, mn in zip(maxs, mins)]
            ax6.plot(epochs, ranges, marker='s', label=pred_name, linewidth=2)
        ax6.set_xlabel('Epoch', fontsize=11)
        ax6.set_ylabel('Score Range (Max - Min)', fontsize=11)
        ax6.set_title('Predictor Score Ranges Over Time', fontsize=12)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # 7. loss distribution per epoch (box plot)
    ax7 = fig.add_subplot(gs[2, 0])
    if len(training_metrics["batch_losses"]) > 0 and len(training_metrics["epochs"]) > 1:
        # group batch losses by epoch
        batch_idx = 0
        epoch_batch_losses = []
        for epoch_num in training_metrics["epochs"]:
            epoch_losses = []
            # estimate batches per epoch (rough)
            batches_per_epoch = len(training_metrics["batch_losses"]) // len(training_metrics["epochs"])
            start_idx = (epoch_num - 1) * batches_per_epoch
            end_idx = min(epoch_num * batches_per_epoch, len(training_metrics["batch_losses"]))
            if start_idx < len(training_metrics["batch_losses"]):
                epoch_losses = training_metrics["batch_losses"][start_idx:end_idx]
            epoch_batch_losses.append(epoch_losses)
        
        if epoch_batch_losses:
            ax7.boxplot(epoch_batch_losses, labels=training_metrics["epochs"])
            ax7.set_xlabel('Epoch', fontsize=11)
            ax7.set_ylabel('Batch Loss', fontsize=11)
            ax7.set_title('Batch Loss Distribution Per Epoch', fontsize=12)
            ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. cumulative training time
    ax8 = fig.add_subplot(gs[2, 1])
    if len(training_metrics["epoch_times"]) > 0:
        cumulative_times = np.cumsum(training_metrics["epoch_times"])
        ax8.plot(training_metrics["epochs"], cumulative_times / 60, 'purple', marker='o', linewidth=2)
        ax8.set_xlabel('Epoch', fontsize=11)
        ax8.set_ylabel('Cumulative Time (minutes)', fontsize=11)
        ax8.set_title('Cumulative Training Time', fontsize=12)
        ax8.grid(True, alpha=0.3)
    
    # 9. predictor score statistics summary
    ax9 = fig.add_subplot(gs[2, 2])
    if len(training_metrics["predictor_score_distributions"]) > 0:
        # show final epoch statistics
        final_stats = training_metrics["predictor_score_distributions"][-1]["stats"]
        pred_names = list(final_stats.keys())
        means = [final_stats[p]["mean"] for p in pred_names]
        stds = [final_stats[p]["std"] for p in pred_names]
        x_pos = np.arange(len(pred_names))
        ax9.bar(x_pos, means, yerr=stds, alpha=0.7, capsize=5, color=['blue', 'red', 'green'][:len(pred_names)])
        ax9.set_xticks(x_pos)
        ax9.set_xticklabels(pred_names, fontsize=10)
        ax9.set_ylabel('Score', fontsize=11)
        ax9.set_title('Final Predictor Score Statistics', fontsize=12)
        ax9.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Comprehensive Training Analysis', fontsize=16, y=0.995)
    plt.savefig(output_dir / "figures" / "comprehensive_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_model(
    preference_net: PreferenceNet,
    text_encoder,
    base_generator,
    predictors: Dict,
    prompts: List[str],
    device: str,
    output_dir: Path,
    log_func=None
) -> Dict:
    """evaluate model on test prompts and save results."""
    if log_func is None:
        def log_func(msg):
            print(msg)
    
    preference_net.eval()
    results = {}
    
    for prompt in prompts:
        log_func(f"Evaluating on prompt: {prompt}")
        test_peptides = base_generator.sample_unconditioned(num_samples=20)
        predictor_scores = score_peptides_with_predictors(test_peptides, predictors)
        # encode prompt
        with torch.no_grad():
            prompt_embedding = text_encoder.encode([prompt])
            if len(prompt_embedding.shape) == 1:
                prompt_embedding = prompt_embedding.unsqueeze(0)
            prompt_embedding = prompt_embedding.to(device)
            
            # get preference parameters
            eta = preference_net(prompt_embedding)
            
            # compute preference scores
            u_tensor = torch.tensor(predictor_scores, dtype=torch.float32).to(device)
            if eta.shape[-1] < u_tensor.shape[-1]:
                u_tensor = u_tensor[:, :eta.shape[-1]]
            elif eta.shape[-1] > u_tensor.shape[-1]:
                eta = eta[:, :u_tensor.shape[-1]]
            
            preference_scores = torch.sum(eta * u_tensor, dim=-1).cpu().numpy()
        
        # save results for this prompt
        prompt_results = {
            "peptides": test_peptides,
            "predictor_scores": predictor_scores.tolist(),
            "preference_scores": preference_scores.tolist(),
            "top_5_peptides": []
        }
        
        # get top 5 peptides by preference score
        top_indices = np.argsort(preference_scores)[::-1][:5]
        prompt_results["top_5_peptides"] = [
            {
                "peptide": test_peptides[i],
                "preference_score": float(preference_scores[i]),
                "predictor_scores": predictor_scores[i].tolist()
            }
            for i in top_indices
        ]
        
        results[prompt] = prompt_results
        
        # save peptide results to CSV
        df = pd.DataFrame({
            'peptide': test_peptides,
            'preference_score': preference_scores,
            **{f'{pred_name}': predictor_scores[:, i] for i, pred_name in enumerate(predictors.keys())}
        })
        df = df.sort_values('preference_score', ascending=False)
        csv_path = output_dir / "logs" / f"results_{prompt[:30].replace(' ', '_')}.csv"
        df.to_csv(csv_path, index=False)
    
    # create summary figure
    fig, axes = plt.subplots(len(prompts), 1, figsize=(12, 4*len(prompts)))
    if len(prompts) == 1:
        axes = [axes]
    
    for idx, prompt in enumerate(prompts):
        scores = results[prompt]["preference_scores"]
        axes[idx].hist(scores, bins=20, alpha=0.7, edgecolor='black')
        axes[idx].set_xlabel('Preference Score', fontsize=10)
        axes[idx].set_ylabel('Frequency', fontsize=10)
        axes[idx].set_title(f'Preference Score Distribution: {prompt[:50]}...', fontsize=11)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "preference_score_distributions.png", dpi=150)
    plt.close()
    
    return results


def train_preference_network(
    preference_net: PreferenceNet,
    text_encoder,
    base_generator,
    predictors: Dict,
    prompts: List[str],
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = 'cuda',
    use_linear: bool = False,
    reg_weight: float = 0.01,
    peptides_per_batch: int = 16,
    judge_method: str = 'rule_based',
    llm_judge: Optional[Callable] = None,
    output_dir: Optional[str] = None,
    use_lr_scheduler: bool = True,
    scheduler_patience: int = 5,
    scheduler_factor: float = 0.5,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.001
):
    """
    Train preference network using the specified procedure.
    
    For each epoch:
    1. Sample a batch of peptides from frozen generator
    2. Score each peptide with frozen predictors
    3. For each prompt, create pairwise comparisons
    4. Feed prompt through frozen text encoder
    5. Pass embedding into MLP to produce preference parameters
    6. Optimize MLP with pairwise ranking loss
    
    Args:
        preference_net: MLP g_ψ to train
        text_encoder: Frozen text encoder E_text
        base_generator: Frozen base generator
        predictors: Dict of frozen predictor objects
        prompts: List of natural language prompts
        num_epochs: Number of training epochs
        batch_size: Batch size for training pairs
        learning_rate: Learning rate
        device: Device to train on
        use_linear: Whether to use linear preference functional
        reg_weight: L2 regularization weight
        peptides_per_batch: Number of peptides to sample per batch
        judge_method: 'rule_based' or 'llm_judge'
        llm_judge: Optional LLM judge function
    """
    preference_net = preference_net.to(device)
    preference_net.train()
    
    optimizer = optim.Adam(preference_net.parameters(), lr=learning_rate)
    
    # set up logging and results directory first (needed for log function)
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/training_{timestamp}")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # create subdirectories
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    
    # initialize logging
    log_file = output_dir / "logs" / "training.log"
    metrics_file = output_dir / "logs" / "metrics.json"
    
    def log(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(log_file, "a") as f:
            f.write(log_msg + "\n")
    
    # Initialize training metrics
    training_metrics = {
        "epochs": [],
        "losses": [],
        "batch_losses": [],
        "batch_indices": [],  # track which batch each loss corresponds to
        "epoch_times": [],
        "predictor_score_distributions": [],  # track distribution changes over epochs
        "preference_score_distributions": [],  # track preference scores over epochs
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "device": device,
        "prompts": prompts,
        "predictors": list(predictors.keys()),
        "start_time": datetime.now().isoformat()
    }
    
    global_batch_idx = 0
    
    # Learning rate scheduler: reduce LR on plateau
    scheduler = None
    if use_lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=scheduler_factor, 
            patience=scheduler_patience,
            min_lr=1e-6
        )
        log(f"Using learning rate scheduler: ReduceLROnPlateau (patience={scheduler_patience}, factor={scheduler_factor})")
    
    # Early stopping
    best_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopped = False
    
    log(f"Starting training with {num_epochs} epochs")
    log(f"Output directory: {output_dir}")
    log(f"Device: {device}")
    log(f"Learning rate: {learning_rate}")
    log(f"Batch size: {batch_size}")
    log(f"Prompts: {prompts}")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0
        batch_losses = []
        
        log(f"Epoch {epoch+1}/{num_epochs}: Sampling peptides...")
        peptides = sample_peptide_batch(
            base_generator,
            batch_size=peptides_per_batch * len(prompts),
            seed=epoch
        )
        
        log("Scoring peptides with predictors...")
        predictor_scores = score_peptides_with_predictors(peptides, predictors)
        
        # track predictor score distributions for this epoch
        epoch_predictor_stats = {}
        for pred_idx, pred_name in enumerate(predictors.keys()):
            pred_scores = predictor_scores[:, pred_idx]
            epoch_predictor_stats[pred_name] = {
                "mean": float(np.mean(pred_scores)),
                "std": float(np.std(pred_scores)),
                "min": float(np.min(pred_scores)),
                "max": float(np.max(pred_scores)),
                "median": float(np.median(pred_scores))
            }
        training_metrics["predictor_score_distributions"].append({
            "epoch": epoch + 1,
            "stats": epoch_predictor_stats
        })
        
        # Algorithm Line 9: Sample prompts {t(i)}^B_{i=1} from T (prompt distribution)
        # The prompt distribution T is the list of prompts provided in config.
        # We sample uniformly from this list by cycling through all prompts each epoch.
        # For each prompt, we create pairwise comparisons using the weak supervision source W.
        all_comparisons = []
        for prompt_idx, prompt in enumerate(prompts):
            # Assign peptides to this prompt (uniform distribution over prompts)
            prompt_peptides = peptides[prompt_idx * peptides_per_batch:(prompt_idx + 1) * peptides_per_batch]
            prompt_scores = predictor_scores[prompt_idx * peptides_per_batch:(prompt_idx + 1) * peptides_per_batch]
            
            if judge_method == 'rule_based':
                comparisons = create_pairwise_comparisons_rule_based(
                    prompt_peptides, prompt_scores, prompt, predictors
                )
            elif judge_method == 'llm_judge':
                comparisons = create_pairwise_comparisons_llm_judge(
                    prompt_peptides, prompt_scores, prompt, llm_judge
                )
            else:
                raise ValueError(f"Unknown judge method: {judge_method}")
            
            # Add prompt to each comparison
            all_comparisons.extend([(prompt, x_a, x_b, y_ab) for x_a, x_b, y_ab in comparisons])
        
        if len(all_comparisons) == 0:
            log("Warning: No comparisons generated, skipping epoch")
            continue
        
        np.random.shuffle(all_comparisons)
        num_batches_epoch = (len(all_comparisons) + batch_size - 1) // batch_size
        
        log(f"Training on {len(all_comparisons)} comparisons in {num_batches_epoch} batches...")
        
        for batch_idx in tqdm(range(num_batches_epoch), desc=f"Epoch {epoch+1}"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(all_comparisons))
            batch_comparisons = all_comparisons[start_idx:end_idx]
            
            # Extract batch data
            batch_prompts = [comp[0] for comp in batch_comparisons]
            batch_peptides_a = [comp[1] for comp in batch_comparisons]
            batch_peptides_b = [comp[2] for comp in batch_comparisons]
            batch_labels = torch.tensor([comp[3] for comp in batch_comparisons], dtype=torch.float32).to(device)
            
            # Algorithm Lines 15-17: Compute normalized predictor coordinates u_k(x_j) = F_k(f_k(x_j))
            # Get predictor scores for batch peptides
            batch_scores_a = []
            batch_scores_b = []
            for x_a, x_b in zip(batch_peptides_a, batch_peptides_b):
                scores_a = []
                scores_b = []
                for pred_name, predictor in predictors.items():
                    # u_k(x) = F_k(f_k(x)) - normalized predictor coordinates
                    scores_a.append(predictor.normalize(predictor.predict(x_a)))
                    scores_b.append(predictor.normalize(predictor.predict(x_b)))
                batch_scores_a.append(scores_a)
                batch_scores_b.append(scores_b)
            
            u_a = torch.tensor(batch_scores_a, dtype=torch.float32).to(device)  # u(x_a)
            u_b = torch.tensor(batch_scores_b, dtype=torch.float32).to(device)  # u(x_b)
            
            # Algorithm Line 10: Compute text embeddings e(i) ← E_text(t(i))
            # Feed prompts through frozen text encoder
            with torch.no_grad():
                prompt_embeddings = text_encoder.encode(batch_prompts)  # e(i) = E_text(t(i))
                if len(prompt_embeddings.shape) == 1:
                    prompt_embeddings = prompt_embeddings.unsqueeze(0)
                # Ensure embeddings are on the correct device
                prompt_embeddings = prompt_embeddings.to(device)
            
            # Forward pass and loss computation
            optimizer.zero_grad()
            
            try:
                # Get preference parameters η(i) = g_ψ(e(i)) as per algorithm Line 11
                eta = preference_net(prompt_embeddings)  # (batch_size, output_dim)
                
                # Compute pairwise preference loss L_pref as per algorithm Line 28
                loss = compute_pairwise_ranking_loss(
                    preference_net,
                    text_encoder,
                    u_a,
                    u_b,
                    prompt_embeddings,
                    batch_labels,
                    use_linear,
                    eta=eta
                )
                
                # Add regularization L_reg(ψ) on preference parameters η(i) as per algorithm Line 30
                if reg_weight > 0:
                    l2_reg = torch.mean(eta.pow(2.0).sum(dim=-1))
                    loss = loss + reg_weight * l2_reg
                
                # Update ψ ← ψ – α∇ψ (L_pref + λ_reg L_reg) as per algorithm Line 31
                
                # Backward pass
                loss.backward()
                optimizer.step()
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                print(f"u_a device: {u_a.device}, u_b device: {u_b.device}")
                print(f"prompt_embeddings device: {prompt_embeddings.device}")
                print(f"batch_labels device: {batch_labels.device}")
                print(f"preference_net device: {next(preference_net.parameters()).device}")
                raise
            
            batch_loss = loss.item()
            epoch_loss += batch_loss
            batch_losses.append(batch_loss)
            training_metrics["batch_indices"].append(global_batch_idx)
            global_batch_idx += 1
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        epoch_time = time.time() - epoch_start_time
        training_metrics["epoch_times"].append(float(epoch_time))
        
        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(avg_loss)
            current_lr = optimizer.param_groups[0]['lr']
            log(f"Epoch {epoch+1}/{num_epochs} completed: Average Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")
        else:
            log(f"Epoch {epoch+1}/{num_epochs} completed: Average Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
        
        # Early stopping check
        if avg_loss < best_loss - early_stopping_min_delta:
            best_loss = avg_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            log(f"Early stopping triggered: no improvement for {early_stopping_patience} epochs")
            log(f"Best loss: {best_loss:.4f} at epoch {best_epoch}")
            early_stopped = True
            break
        
        # save metrics
        training_metrics["epochs"].append(epoch + 1)
        training_metrics["losses"].append(float(avg_loss))
        training_metrics["batch_losses"].extend([float(l) for l in batch_losses])
        
        # evaluate preference scores at end of epoch to track distribution changes
        if epoch % max(1, num_epochs // 5) == 0 or epoch == num_epochs - 1:
            log(f"Computing preference score distribution at epoch {epoch+1}...")
            preference_net.eval()
            with torch.no_grad():
                test_peptides = base_generator.sample_unconditioned(num_samples=50)
                test_scores = score_peptides_with_predictors(test_peptides, predictors)
                
                # compute preference scores for first prompt
                test_prompt = prompts[0]
                prompt_embedding = text_encoder.encode([test_prompt])
                if len(prompt_embedding.shape) == 1:
                    prompt_embedding = prompt_embedding.unsqueeze(0)
                prompt_embedding = prompt_embedding.to(device)
                
                eta = preference_net(prompt_embedding)
                u_tensor = torch.tensor(test_scores, dtype=torch.float32).to(device)
                if eta.shape[-1] < u_tensor.shape[-1]:
                    u_tensor = u_tensor[:, :eta.shape[-1]]
                elif eta.shape[-1] > u_tensor.shape[-1]:
                    eta = eta[:, :u_tensor.shape[-1]]
                
                pref_scores = torch.sum(eta * u_tensor, dim=-1).cpu().numpy()
                
                training_metrics["preference_score_distributions"].append({
                    "epoch": epoch + 1,
                    "mean": float(np.mean(pref_scores)),
                    "std": float(np.std(pref_scores)),
                    "min": float(np.min(pref_scores)),
                    "max": float(np.max(pref_scores)),
                    "median": float(np.median(pref_scores))
                })
            preference_net.train()
        
        # save checkpoint every epoch
        checkpoint_path = output_dir / "checkpoints" / f"checkpoint_epoch_{epoch+1}.ckpt"
        checkpoint_data = {
            'epoch': epoch + 1,
            'state_dict': preference_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'input_dim': preference_net.input_dim,
            'hidden_dim': preference_net.hidden_dim,
            'output_dim': preference_net.output_dim,
            'num_predictors': preference_net.num_predictors
        }
        if scheduler is not None:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save best model checkpoint
        if epoch + 1 == best_epoch:
            best_checkpoint_path = output_dir / "checkpoints" / "best_model.ckpt"
            torch.save(checkpoint_data, best_checkpoint_path)
            log(f"Saved best model checkpoint (loss: {best_loss:.4f}) to {best_checkpoint_path}")
        
        # save metrics to file
        with open(metrics_file, 'w') as f:
            json.dump(training_metrics, f, indent=2)
        
        # plot comprehensive training curves
        if epoch > 0:
            # 1. epoch-level loss
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # epoch loss
            axes[0, 0].plot(training_metrics["epochs"], training_metrics["losses"], 'b-', linewidth=2, marker='o')
            axes[0, 0].set_xlabel('Epoch', fontsize=12)
            axes[0, 0].set_ylabel('Average Loss', fontsize=12)
            axes[0, 0].set_title('Training Loss (Epoch-level)', fontsize=14)
            axes[0, 0].grid(True, alpha=0.3)
            
            # batch-level loss (smoothed)
            if len(training_metrics["batch_losses"]) > 0:
                batch_losses = training_metrics["batch_losses"]
                # smooth with moving average
                window = max(1, len(batch_losses) // 100)
                if window > 1:
                    smoothed = pd.Series(batch_losses).rolling(window=window, center=True).mean()
                    axes[0, 1].plot(range(len(batch_losses)), batch_losses, 'lightblue', alpha=0.3, label='Raw')
                    axes[0, 1].plot(range(len(smoothed)), smoothed, 'b-', linewidth=2, label='Smoothed')
                else:
                    axes[0, 1].plot(range(len(batch_losses)), batch_losses, 'b-', linewidth=1, alpha=0.5)
                axes[0, 1].set_xlabel('Batch', fontsize=12)
                axes[0, 1].set_ylabel('Loss', fontsize=12)
                axes[0, 1].set_title('Training Loss (Batch-level)', fontsize=14)
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # predictor score distributions over epochs
            if len(training_metrics["predictor_score_distributions"]) > 0:
                for pred_name in predictors.keys():
                    epochs = [d["epoch"] for d in training_metrics["predictor_score_distributions"]]
                    means = [d["stats"][pred_name]["mean"] for d in training_metrics["predictor_score_distributions"]]
                    axes[1, 0].plot(epochs, means, marker='o', label=pred_name, linewidth=2)
                axes[1, 0].set_xlabel('Epoch', fontsize=12)
                axes[1, 0].set_ylabel('Mean Score', fontsize=12)
                axes[1, 0].set_title('Predictor Score Distributions Over Time', fontsize=14)
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # preference score distribution changes
            if len(training_metrics["preference_score_distributions"]) > 0:
                pref_epochs = [d["epoch"] for d in training_metrics["preference_score_distributions"]]
                pref_means = [d["mean"] for d in training_metrics["preference_score_distributions"]]
                pref_stds = [d["std"] for d in training_metrics["preference_score_distributions"]]
                axes[1, 1].plot(pref_epochs, pref_means, 'g-', marker='o', linewidth=2, label='Mean')
                axes[1, 1].fill_between(pref_epochs, 
                                       [m - s for m, s in zip(pref_means, pref_stds)],
                                       [m + s for m, s in zip(pref_means, pref_stds)],
                                       alpha=0.3, label='±1 std')
                axes[1, 1].set_xlabel('Epoch', fontsize=12)
                axes[1, 1].set_ylabel('Preference Score', fontsize=12)
                axes[1, 1].set_title('Preference Score Distribution Changes', fontsize=14)
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / "figures" / "training_curves.png", dpi=150)
            plt.close()
            
            # also save individual epoch loss plot for quick reference
            plt.figure(figsize=(10, 6))
            plt.plot(training_metrics["epochs"], training_metrics["losses"], 'b-', linewidth=2, marker='o')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Average Loss', fontsize=12)
            plt.title('Training Loss', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / "figures" / "training_loss.png", dpi=150)
            plt.close()
    
    total_time = time.time() - start_time
    training_metrics["end_time"] = datetime.now().isoformat()
    training_metrics["total_time_seconds"] = total_time
    
    log(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
    
    # create final comprehensive visualization
    log("Creating comprehensive training visualization...")
    create_comprehensive_plots(training_metrics, output_dir, predictors)
    
    # evaluate model on test prompts
    log("Evaluating model on test prompts...")
    eval_results = evaluate_model(
        preference_net, text_encoder, base_generator, predictors, 
        prompts, device, output_dir, log_func=log
    )
    training_metrics["evaluation"] = eval_results
    
    # save final metrics
    with open(metrics_file, 'w') as f:
        json.dump(training_metrics, f, indent=2)
    
    # save losses to CSV for easy analysis
    losses_df = pd.DataFrame({
        'epoch': training_metrics["epochs"],
        'epoch_loss': training_metrics["losses"],
        'epoch_time_seconds': training_metrics["epoch_times"]
    })
    losses_df.to_csv(output_dir / "logs" / "epoch_losses.csv", index=False)
    
    if len(training_metrics["batch_losses"]) > 0:
        batch_losses_df = pd.DataFrame({
            'batch_index': training_metrics["batch_indices"],
            'batch_loss': training_metrics["batch_losses"]
        })
        batch_losses_df.to_csv(output_dir / "logs" / "batch_losses.csv", index=False)
    
    # save distribution changes to CSV
    if len(training_metrics["predictor_score_distributions"]) > 0:
        dist_data = []
        for dist_entry in training_metrics["predictor_score_distributions"]:
            row = {"epoch": dist_entry["epoch"]}
            for pred_name, stats in dist_entry["stats"].items():
                for stat_name, stat_value in stats.items():
                    row[f"{pred_name}_{stat_name}"] = stat_value
            dist_data.append(row)
        dist_df = pd.DataFrame(dist_data)
        dist_df.to_csv(output_dir / "logs" / "predictor_distributions.csv", index=False)
    
    if len(training_metrics["preference_score_distributions"]) > 0:
        pref_dist_data = []
        for dist_entry in training_metrics["preference_score_distributions"]:
            pref_dist_data.append({
                "epoch": dist_entry["epoch"],
                "mean": dist_entry["mean"],
                "std": dist_entry["std"],
                "min": dist_entry["min"],
                "max": dist_entry["max"],
                "median": dist_entry["median"]
            })
        pref_dist_df = pd.DataFrame(pref_dist_data)
        pref_dist_df.to_csv(output_dir / "logs" / "preference_distributions.csv", index=False)
    
    preference_net.eval()
    return preference_net


def main():
    parser = argparse.ArgumentParser(
        description="Train LaPep preference network"
    )
    parser.add_argument('--config', type=str, required=True, help='Config file')
    parser.add_argument('--output', type=str, default='preference_net.ckpt', help='Output path')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training pairs')
    parser.add_argument('--peptides_per_batch', type=int, default=16, help='Peptides to sample per batch')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--use_linear', action='store_true', help='Use linear preferences')
    parser.add_argument('--judge_method', type=str, default='rule_based', 
                       choices=['rule_based', 'llm_judge'], 
                       help='Method for creating pairwise comparisons')
    parser.add_argument('--reg_weight', type=float, default=0.01, help='L2 regularization weight')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results (default: results/training_TIMESTAMP)')
    parser.add_argument('--use_lr_scheduler', action='store_true', help='Use learning rate scheduler')
    parser.add_argument('--no_lr_scheduler', dest='use_lr_scheduler', action='store_false', help='Disable learning rate scheduler')
    parser.set_defaults(use_lr_scheduler=True)  # Default to True
    parser.add_argument('--scheduler_patience', type=int, default=5, help='Patience for LR scheduler')
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help='Factor for LR reduction')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.001, help='Minimum delta for early stopping')
    
    args = parser.parse_args()
    
    # Handle use_lr_scheduler default (argparse quirk with store_true/store_false)
    # If neither flag is provided, default to True
    if not hasattr(args, 'use_lr_scheduler'):
        args.use_lr_scheduler = True
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    print("=" * 80)
    print("LOADING MODELS FOR TRAINING")
    print("=" * 80)
    
    # Auto-detect device: use CPU if CUDA not available, even if --device cuda is specified
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU instead")
        actual_device = 'cpu'
    else:
        actual_device = args.device
    
    print(f"\n[Device] Using device: {actual_device}")
    if actual_device.startswith('cuda'):
        print(f"[Device] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[Device] CUDA device: {torch.cuda.get_device_name(0)}")
    
    print(f"\n[Text Encoder] Loading text encoder: {config['text_encoder_name']}")
    text_encoder = load_text_encoder(config['text_encoder_name'], device=actual_device)
    print(f"[Text Encoder] Loaded successfully on {text_encoder.device}")
    print(f"\n[Predictors] Loading predictors...")
    # Determine format from generator type
    generator_type = config.get('generator_type', config.get('base_generator_type', 'pepmdlm'))
    format_type = 'wt' if generator_type == 'pepdfm' else 'smiles'
    
    predictors = load_predictors(
        config,
        format_type=format_type,
        device=actual_device,
        protein_seq=config.get('protein_seq')
    )
    
    # Load frozen base generator
    print(f"\n[Base Generator] Loading {generator_type.upper()} generator...")
    from generators.peptune_wrapper import load_peptune_generator
    from generators.dfm_wrapper import load_dfm_model
    
    if generator_type == 'pepdfm':
        generator_path = config.get('dfm_model_path')
        if generator_path is None:
            raise ValueError("dfm_model_path not specified in config for generator_type='pepdfm'")
        base_generator = load_dfm_model(generator_path, device=actual_device)
        if base_generator is None:
            raise RuntimeError(f"Failed to load PepDFM model from {generator_path}")
    else:
        generator_path = config.get('base_generator_path')
        if generator_path is None:
            raise ValueError("base_generator_path not specified in config")
        base_generator = load_peptune_generator(generator_path, device=actual_device)
        if base_generator.model is None:
            raise RuntimeError(f"Failed to load PepMDLM model from {generator_path}")
    print(f"[Base Generator] Loaded successfully")
    
    # here are defaults but can also be defined in the config file 
    prompts = config.get('training_prompts', [
        "Generate a peptide with high binding affinity and low toxicity",
        "Generate a stable peptide with long half-life",
        "Generate a balanced peptide optimizing binding, toxicity, and half-life"
    ])
    
    print(f"\n[Prompt Distribution T] Using {len(prompts)} training prompt(s):")
    for i, prompt in enumerate(prompts, 1):
        print(f"  {i}. {prompt}")
    print(f"\n[Preference Network] Detecting embedding dimension...")
    print("(This may take a moment to load the Qwen model on first run...)")
    try:
        import time
        start_time = time.time()
        test_embedding = text_encoder.encode("test")
        elapsed = time.time() - start_time
        embedding_dim = test_embedding.shape[-1]
        print(f"[Preference Network] Detected embedding dimension: {embedding_dim} (took {elapsed:.2f}s)")
    except Exception as e:
        print(f"[Preference Network] Warning: Could not detect embedding dimension: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to default for Qwen3-Embedding-0.6B
        embedding_dim = 1024
        print(f"[Preference Network] Using default embedding dimension: {embedding_dim}")
    
    hidden_dim = config.get('hidden_dim', 256)
    output_dim = config.get('output_dim', 64)
    num_predictors = len(predictors)
    
    print(f"\n[Preference Network] Initializing preference network:")
    print(f"  - Input dimension (embedding): {embedding_dim}")
    print(f"  - Hidden dimension: {hidden_dim}")
    print(f"  - Output dimension: {output_dim}")
    print(f"  - Number of predictors: {num_predictors}")
    
    preference_net = PreferenceNet(
        input_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_predictors=num_predictors
    )
    preference_net = preference_net.to(actual_device)
    print(f"[Preference Network] Preference network created and moved to {actual_device}")
    print("=" * 80)
    
    # Optional LLM judge
    llm_judge = None
    if args.judge_method == 'llm_judge':
        try:
            from language.llm_judge import create_llm_judge
            llm_api_key = config.get('llm_api_key', None)
            llm_model = config.get('llm_model', 'gpt-4')
            llm_judge = create_llm_judge(model_name=llm_model, api_key=llm_api_key)
            print(f"Using LLM judge with model: {llm_model}")
        except ImportError:
            print("Warning: LLM judge dependencies not available, falling back to rule-based")
            args.judge_method = 'rule_based'
    
    # Train
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"[Training Config]")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Peptides per batch: {args.peptides_per_batch}")
    print(f"  - Judge method: {args.judge_method}")
    print(f"  - Device: {actual_device}")
    print(f"  - Number of training prompts: {len(prompts)}")
    print(f"  - Output directory: {args.output_dir}")
    print("=" * 80)
    print("\nTraining preference network...")
    trained_net = train_preference_network(
        preference_net,
        text_encoder,
        base_generator,
        predictors,
        prompts,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=actual_device,
        use_linear=args.use_linear,
        reg_weight=args.reg_weight,
        peptides_per_batch=args.peptides_per_batch,
        judge_method=args.judge_method,
        llm_judge=llm_judge,
        output_dir=args.output_dir,
        use_lr_scheduler=args.use_lr_scheduler,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta
    )
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"results/training_{timestamp}")
    
    # save final model
    final_model_path = output_dir / "final_model.ckpt"
    torch.save({
        'state_dict': trained_net.state_dict(),
        'input_dim': trained_net.input_dim,
        'hidden_dim': trained_net.hidden_dim,
        'output_dim': trained_net.output_dim,
        'num_predictors': trained_net.num_predictors
    }, final_model_path)
    
    # also save to user-specified path if different
    if args.output != str(final_model_path):
        torch.save({
            'state_dict': trained_net.state_dict(),
            'input_dim': trained_net.input_dim,
            'hidden_dim': trained_net.hidden_dim,
            'output_dim': trained_net.output_dim,
            'num_predictors': trained_net.num_predictors
        }, args.output)
    
    print(f"Saved trained model to {final_model_path}")
    print(f"All results saved to {output_dir}")
    print(f"  - Training logs: {output_dir / 'logs'}")
    print(f"  - Checkpoints: {output_dir / 'checkpoints'}")
    print(f"  - Figures: {output_dir / 'figures'}")


if __name__ == '__main__':
    main()
