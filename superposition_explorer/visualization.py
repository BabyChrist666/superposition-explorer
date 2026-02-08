"""
Visualization tools for superposition analysis.

Provides ASCII-based visualizations for terminal display
(no matplotlib dependency).
"""

import torch
from typing import Optional, List, Dict, Any


def plot_feature_geometry(
    W: torch.Tensor,
    n_features: int = 20,
    width: int = 60,
) -> str:
    """
    Create ASCII visualization of feature geometry.

    Shows feature norms as a bar chart.
    """
    if W.shape[0] > W.shape[1]:
        W = W.T

    with torch.no_grad():
        norms = W.norm(dim=0)[:n_features].cpu().numpy()

    max_norm = max(norms) if len(norms) > 0 else 1
    bar_width = width - 15  # Leave room for labels

    lines = ["Feature Norms:"]
    lines.append("-" * width)

    for i, norm in enumerate(norms):
        bar_len = int((norm / max_norm) * bar_width)
        bar = "#" * bar_len
        lines.append(f"  F{i:02d} |{bar} {norm:.3f}")

    lines.append("-" * width)
    lines.append(f"Mean: {norms.mean():.4f}, Std: {norms.std():.4f}")

    return "\n".join(lines)


def plot_interference_heatmap(
    interference: torch.Tensor,
    n_features: int = 10,
) -> str:
    """
    Create ASCII heatmap of feature interference.

    Uses characters to represent interference levels:
    . = low, o = medium, O = high, # = very high
    """
    interference = interference[:n_features, :n_features].cpu().numpy()

    chars = " .oO#"  # Increasing intensity

    lines = ["Interference Heatmap:"]

    # Header
    header = "     " + "".join(f"{i%10}" for i in range(n_features))
    lines.append(header)
    lines.append("    +" + "-" * n_features)

    for i in range(n_features):
        row = f" F{i:02d}|"
        for j in range(n_features):
            if i == j:
                row += "x"  # Self-interference
            else:
                val = interference[i, j]
                idx = min(int(val * len(chars)), len(chars) - 1)
                row += chars[idx]
        lines.append(row)

    lines.append("")
    lines.append("Legend: ' '=0, '.'<0.25, 'o'<0.5, 'O'<0.75, '#'>=0.75, 'x'=self")

    return "\n".join(lines)


def plot_dimensionality_analysis(
    d_eff: torch.Tensor,
    n_features: int = 20,
    width: int = 60,
) -> str:
    """
    Visualize effective dimensionality per feature.
    """
    d_eff = d_eff[:n_features].cpu().numpy()

    max_d = max(d_eff) if len(d_eff) > 0 else 1
    bar_width = width - 15

    lines = ["Effective Dimensionality:"]
    lines.append("-" * width)

    for i, d in enumerate(d_eff):
        bar_len = int((d / max_d) * bar_width)
        bar = "=" * bar_len
        lines.append(f"  F{i:02d} |{bar} {d:.2f}")

    lines.append("-" * width)
    lines.append(f"Mean d_eff: {d_eff.mean():.3f}")
    lines.append(f"Min d_eff: {d_eff.min():.3f}, Max d_eff: {d_eff.max():.3f}")

    return "\n".join(lines)


def plot_sparsity_transition(
    results: Dict[str, List],
    width: int = 60,
) -> str:
    """
    Visualize how metrics change with sparsity.
    """
    lines = ["Sparsity-Superposition Transition:"]
    lines.append("=" * width)

    sparsities = results.get("sparsity", [])
    interferences = results.get("interference", [])

    if not sparsities or not interferences:
        return "No data to plot"

    max_int = max(interferences) if interferences else 1
    bar_width = width - 25

    for s, i in zip(sparsities, interferences):
        bar_len = int((i / max_int) * bar_width)
        bar = "*" * bar_len
        lines.append(f"  S={s:.2f} |{bar} int={i:.4f}")

    lines.append("=" * width)

    return "\n".join(lines)


def plot_feature_similarity_matrix(
    cosines: torch.Tensor,
    n_features: int = 10,
) -> str:
    """
    Visualize cosine similarity between features.
    """
    cosines = cosines[:n_features, :n_features].cpu().numpy()

    lines = ["Feature Cosine Similarities:"]

    # Map cosines [-1, 1] to characters
    def cosine_to_char(c):
        if c > 0.8:
            return "+"
        elif c > 0.3:
            return "o"
        elif c > -0.3:
            return "."
        elif c > -0.8:
            return "-"
        else:
            return "X"

    header = "     " + "".join(f"{i%10}" for i in range(n_features))
    lines.append(header)
    lines.append("    +" + "-" * n_features)

    for i in range(n_features):
        row = f" F{i:02d}|"
        for j in range(n_features):
            if i == j:
                row += "1"
            else:
                row += cosine_to_char(cosines[i, j])
        lines.append(row)

    lines.append("")
    lines.append("Legend: '+'>.8, 'o'>.3, '.'~0, '-'<-.3, 'X'<-.8")

    return "\n".join(lines)


def plot_training_progress(
    losses: List[float],
    width: int = 60,
    height: int = 10,
) -> str:
    """
    ASCII plot of training loss over time.
    """
    if not losses:
        return "No training data"

    lines = ["Training Progress:"]
    lines.append("-" * width)

    # Sample losses if too many
    if len(losses) > width - 5:
        step = len(losses) // (width - 5)
        sampled = losses[::step]
    else:
        sampled = losses

    min_loss = min(sampled)
    max_loss = max(sampled)
    loss_range = max_loss - min_loss + 1e-8

    # Create grid
    grid = [[" " for _ in range(len(sampled))] for _ in range(height)]

    for x, loss in enumerate(sampled):
        y = int((1 - (loss - min_loss) / loss_range) * (height - 1))
        y = max(0, min(height - 1, y))
        grid[y][x] = "*"

    # Add Y-axis labels
    for i, row in enumerate(grid):
        loss_val = max_loss - (i / (height - 1)) * loss_range
        label = f"{loss_val:.4f}" if i in [0, height // 2, height - 1] else "      "
        lines.append(f"{label} |{''.join(row)}")

    lines.append("       +" + "-" * len(sampled))
    lines.append(f"       Step 0 -> {len(losses)}")
    lines.append("")
    lines.append(f"Final loss: {losses[-1]:.6f}")

    return "\n".join(lines)


def plot_feature_clusters(
    clusters: List[List[int]],
    n_features: int,
) -> str:
    """
    Visualize feature clustering.
    """
    lines = ["Feature Clusters:"]
    lines.append("-" * 50)

    # Create feature -> cluster mapping
    feature_to_cluster = {}
    for i, cluster in enumerate(clusters):
        for f in cluster:
            feature_to_cluster[f] = i

    # Display clusters
    for i, cluster in enumerate(clusters):
        if len(cluster) > 1:
            features = ", ".join(f"F{f}" for f in cluster[:10])
            if len(cluster) > 10:
                features += f", ... (+{len(cluster) - 10} more)"
            lines.append(f"  Cluster {i}: {features}")

    # Summary
    singleton_count = sum(1 for c in clusters if len(c) == 1)
    multi_count = sum(1 for c in clusters if len(c) > 1)

    lines.append("-" * 50)
    lines.append(f"Singletons: {singleton_count}, Multi-feature clusters: {multi_count}")

    return "\n".join(lines)


def create_summary_report(
    model,
    losses: Optional[List[float]] = None,
) -> str:
    """
    Create a comprehensive summary report.
    """
    from superposition_explorer.analysis import SuperpositionAnalyzer

    lines = ["=" * 60]
    lines.append("SUPERPOSITION ANALYSIS REPORT")
    lines.append("=" * 60)

    analyzer = SuperpositionAnalyzer(model)

    if hasattr(model, 'W'):
        W = model.W
    else:
        return "Could not find weight matrix"

    geometry = analyzer.analyze_weight_matrix(W)
    interference = analyzer.compute_interference(W)
    d_eff = analyzer.compute_effective_dimensionality(W)

    # Model info
    lines.append("\n--- Model Configuration ---")
    if hasattr(model, 'config'):
        lines.append(f"Features: {model.config.n_features}")
        lines.append(f"Hidden: {model.config.n_hidden}")
        lines.append(f"Ratio: {model.config.n_features / model.config.n_hidden:.2f}x")
        lines.append(f"Sparsity: {model.config.sparsity}")

    # Geometry summary
    lines.append("\n--- Feature Geometry ---")
    lines.append(f"Mean norm: {geometry.norms.mean():.4f}")
    lines.append(f"Std norm: {geometry.norms.std():.4f}")
    lines.append(f"Max norm: {geometry.norms.max():.4f}")
    lines.append(f"Min norm: {geometry.norms.min():.4f}")

    # Interference summary
    lines.append("\n--- Interference ---")
    lines.append(f"Total interference: {interference.total_interference:.4f}")
    lines.append(f"Mean per-feature: {interference.feature_interference.mean():.4f}")
    lines.append(f"Max per-feature: {interference.feature_interference.max():.4f}")

    # Dimensionality
    lines.append("\n--- Effective Dimensionality ---")
    lines.append(f"Mean d_eff: {d_eff.mean():.3f}")
    lines.append(f"Min d_eff: {d_eff.min():.3f}")
    lines.append(f"Max d_eff: {d_eff.max():.3f}")

    # Training info
    if losses:
        lines.append("\n--- Training ---")
        lines.append(f"Final loss: {losses[-1]:.6f}")
        lines.append(f"Initial loss: {losses[0]:.6f}")
        lines.append(f"Improvement: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)
