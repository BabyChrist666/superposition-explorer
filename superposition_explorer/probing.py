"""
Probing tools for understanding feature representations.

Linear probes to test whether features are linearly decodable
and measure polysemanticity (neurons responding to multiple features).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple


@dataclass
class ProbeResult:
    """Results from a linear probe."""
    accuracy: float
    loss: float
    weights: torch.Tensor
    feature_idx: int
    r_squared: float = 0.0

    def to_dict(self) -> dict:
        return {
            "feature_idx": self.feature_idx,
            "accuracy": self.accuracy,
            "loss": self.loss,
            "r_squared": self.r_squared,
        }


class LinearProbe(nn.Module):
    """
    Linear probe for testing feature decodability.
    """

    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def get_direction(self) -> torch.Tensor:
        """Get the probe's weight vector (feature direction)."""
        return self.linear.weight.squeeze()


def probe_features(
    activations: torch.Tensor,
    targets: torch.Tensor,
    feature_idx: int,
    n_steps: int = 1000,
    learning_rate: float = 1e-2,
    device: str = "cpu",
) -> ProbeResult:
    """
    Train a linear probe to decode a specific feature.

    Args:
        activations: Hidden activations [batch, hidden_dim]
        targets: Target feature values [batch, n_features]
        feature_idx: Which feature to probe for
        n_steps: Training steps
        learning_rate: Learning rate
        device: Device to use

    Returns:
        ProbeResult with accuracy and weights.
    """
    activations = activations.to(device)
    targets = targets[:, feature_idx].to(device)

    hidden_dim = activations.shape[1]
    probe = LinearProbe(hidden_dim, 1).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)

    for _ in range(n_steps):
        optimizer.zero_grad()
        preds = probe(activations).squeeze()
        loss = F.mse_loss(preds, targets)
        loss.backward()
        optimizer.step()

    # Evaluate
    with torch.no_grad():
        preds = probe(activations).squeeze()
        final_loss = F.mse_loss(preds, targets).item()

        # R-squared
        ss_res = ((targets - preds) ** 2).sum()
        ss_tot = ((targets - targets.mean()) ** 2).sum() + 1e-8
        r_squared = 1 - ss_res / ss_tot

        # Accuracy (for binary features)
        if targets.max() <= 1:
            accuracy = ((preds > 0.5) == (targets > 0.5)).float().mean().item()
        else:
            # Correlation for continuous features
            accuracy = r_squared.item()

    return ProbeResult(
        accuracy=accuracy,
        loss=final_loss,
        weights=probe.get_direction().cpu(),
        feature_idx=feature_idx,
        r_squared=r_squared.item(),
    )


def probe_all_features(
    activations: torch.Tensor,
    targets: torch.Tensor,
    n_steps: int = 500,
    device: str = "cpu",
) -> List[ProbeResult]:
    """
    Probe for all features.

    Returns a list of ProbeResult for each feature.
    """
    n_features = targets.shape[1]
    results = []

    for i in range(n_features):
        result = probe_features(
            activations, targets, i,
            n_steps=n_steps, device=device
        )
        results.append(result)

    return results


def measure_polysemanticity(
    neuron_activations: torch.Tensor,
    feature_labels: torch.Tensor,
    threshold: float = 0.1,
) -> Dict[str, Any]:
    """
    Measure how polysemantic neurons are.

    A neuron is polysemantic if it responds to multiple unrelated features.

    Args:
        neuron_activations: [batch, n_neurons]
        feature_labels: [batch, n_features] (binary indicators)
        threshold: Correlation threshold for "responding to" a feature

    Returns:
        Dictionary with polysemanticity metrics.
    """
    n_neurons = neuron_activations.shape[1]
    n_features = feature_labels.shape[1]

    # Compute correlation between each neuron and each feature
    correlations = torch.zeros(n_neurons, n_features)

    for i in range(n_neurons):
        neuron = neuron_activations[:, i]
        for j in range(n_features):
            feature = feature_labels[:, j].float()

            # Pearson correlation
            neuron_centered = neuron - neuron.mean()
            feature_centered = feature - feature.mean()

            corr = (neuron_centered * feature_centered).sum() / (
                (neuron_centered.norm() * feature_centered.norm()) + 1e-8
            )
            correlations[i, j] = corr.abs()

    # Count how many features each neuron responds to
    responds_to = (correlations > threshold).sum(dim=1)

    # Polysemanticity score: neurons responding to >1 feature
    polysemantic_neurons = (responds_to > 1).sum().item()
    polysemanticity_ratio = polysemantic_neurons / n_neurons

    # Average number of features per neuron
    avg_features_per_neuron = responds_to.float().mean().item()

    # Maximum correlation per neuron (selectivity)
    max_correlations = correlations.max(dim=1)[0]
    avg_selectivity = max_correlations.mean().item()

    return {
        "polysemantic_neurons": polysemantic_neurons,
        "polysemanticity_ratio": polysemanticity_ratio,
        "avg_features_per_neuron": avg_features_per_neuron,
        "avg_selectivity": avg_selectivity,
        "correlation_matrix": correlations,
        "feature_counts": responds_to,
    }


def find_monosemantic_neurons(
    neuron_activations: torch.Tensor,
    feature_labels: torch.Tensor,
    selectivity_threshold: float = 0.5,
    polysemanticity_threshold: float = 0.2,
) -> Dict[int, int]:
    """
    Find neurons that are monosemantic (respond to exactly one feature).

    Returns:
        Dictionary mapping neuron_idx -> feature_idx for monosemantic neurons.
    """
    metrics = measure_polysemanticity(
        neuron_activations, feature_labels, polysemanticity_threshold
    )
    correlations = metrics["correlation_matrix"]

    monosemantic = {}

    for neuron_idx in range(correlations.shape[0]):
        correlations_for_neuron = correlations[neuron_idx]

        # Check if responds to exactly one feature strongly
        max_corr = correlations_for_neuron.max()
        second_max = correlations_for_neuron.topk(2)[0][1] if correlations_for_neuron.shape[0] > 1 else 0

        if max_corr > selectivity_threshold and second_max < polysemanticity_threshold:
            feature_idx = correlations_for_neuron.argmax().item()
            monosemantic[neuron_idx] = feature_idx

    return monosemantic


def compute_feature_coverage(
    model_weights: torch.Tensor,
    probe_results: List[ProbeResult],
    r_squared_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute how well the model's representation covers features.

    Args:
        model_weights: Weight matrix from model
        probe_results: Results from probing all features
        r_squared_threshold: Threshold for "well-represented"

    Returns:
        Coverage metrics.
    """
    n_features = len(probe_results)

    # Count well-represented features
    well_represented = sum(1 for r in probe_results if r.r_squared > r_squared_threshold)

    # Average R-squared
    avg_r_squared = sum(r.r_squared for r in probe_results) / n_features

    # Feature representation quality distribution
    r_squared_values = [r.r_squared for r in probe_results]

    return {
        "coverage_ratio": well_represented / n_features,
        "well_represented_count": well_represented,
        "avg_r_squared": avg_r_squared,
        "min_r_squared": min(r_squared_values),
        "max_r_squared": max(r_squared_values),
        "r_squared_values": r_squared_values,
    }


def compare_probe_to_model_weights(
    probe_weights: torch.Tensor,
    model_weights: torch.Tensor,
    feature_idx: int,
) -> Dict[str, float]:
    """
    Compare a probe's learned direction to the model's weight for a feature.

    This tests whether the model learns the same direction as an optimal probe.
    """
    # Get model's direction for this feature
    if model_weights.shape[0] > model_weights.shape[1]:
        model_direction = model_weights[:, feature_idx]
    else:
        model_direction = model_weights[feature_idx]

    # Normalize both
    probe_norm = probe_weights / (probe_weights.norm() + 1e-8)
    model_norm = model_direction / (model_direction.norm() + 1e-8)

    # Cosine similarity
    cosine_sim = (probe_norm * model_norm).sum().item()

    # Angle in degrees
    angle_rad = torch.acos(torch.clamp(torch.tensor(cosine_sim), -1, 1))
    angle_deg = angle_rad.item() * 180 / 3.14159

    return {
        "cosine_similarity": cosine_sim,
        "angle_degrees": angle_deg,
        "probe_norm": probe_weights.norm().item(),
        "model_norm": model_direction.norm().item(),
    }
