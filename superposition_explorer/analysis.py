"""
Analysis tools for studying superposition.

Provides metrics and utilities for understanding how features
are represented in superposition.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple


@dataclass
class FeatureGeometry:
    """Geometric properties of feature representations."""
    directions: torch.Tensor  # Unit vectors for each feature
    norms: torch.Tensor       # Magnitude of each feature
    cosine_similarities: torch.Tensor  # Pairwise cosines
    interference_matrix: torch.Tensor  # Pairwise interference

    def to_dict(self) -> dict:
        return {
            "mean_norm": self.norms.mean().item(),
            "std_norm": self.norms.std().item(),
            "mean_interference": self.interference_matrix.mean().item(),
            "max_interference": self.interference_matrix.max().item(),
        }


@dataclass
class InterferenceMetrics:
    """Metrics quantifying feature interference."""
    pairwise_interference: torch.Tensor
    feature_interference: torch.Tensor  # Per-feature average
    total_interference: float
    interference_rank: torch.Tensor  # Features ranked by interference

    def to_dict(self) -> dict:
        return {
            "total_interference": self.total_interference,
            "mean_per_feature": self.feature_interference.mean().item(),
            "max_per_feature": self.feature_interference.max().item(),
            "min_per_feature": self.feature_interference.min().item(),
        }


class SuperpositionAnalyzer:
    """
    Analyzer for studying superposition in neural networks.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def analyze_weight_matrix(
        self,
        W: torch.Tensor,
    ) -> FeatureGeometry:
        """
        Analyze the geometry of a weight matrix.

        Args:
            W: Weight matrix of shape [hidden_dim, n_features] or [n_features, hidden_dim]

        Returns:
            FeatureGeometry with directions, norms, and interference.
        """
        # Ensure W is [hidden_dim, n_features]
        if W.shape[0] > W.shape[1]:
            W = W.T

        with torch.no_grad():
            # Feature norms
            norms = W.norm(dim=0)

            # Normalized directions
            directions = W / (norms.unsqueeze(0) + 1e-8)

            # Cosine similarities
            cosines = directions.T @ directions

            # Interference (squared cosines)
            interference = cosines ** 2

        return FeatureGeometry(
            directions=directions.T,
            norms=norms,
            cosine_similarities=cosines,
            interference_matrix=interference,
        )

    def compute_interference(
        self,
        W: torch.Tensor,
    ) -> InterferenceMetrics:
        """
        Compute detailed interference metrics.

        Args:
            W: Weight matrix.

        Returns:
            InterferenceMetrics with pairwise and per-feature interference.
        """
        geometry = self.analyze_weight_matrix(W)
        interference = geometry.interference_matrix

        n_features = interference.shape[0]

        # Mask out self-interference
        mask = 1 - torch.eye(n_features, device=interference.device)
        masked_interference = interference * mask

        # Per-feature interference (sum of interference with other features)
        feature_interference = masked_interference.sum(dim=1) / (n_features - 1)

        # Total interference
        total = masked_interference.sum() / (n_features * (n_features - 1))

        # Rank features by interference
        _, rank = feature_interference.sort(descending=True)

        return InterferenceMetrics(
            pairwise_interference=interference,
            feature_interference=feature_interference,
            total_interference=total.item(),
            interference_rank=rank,
        )

    def compute_effective_dimensionality(
        self,
        W: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute effective dimensionality for each feature.

        d_eff(i) = ||w_i||^4 / sum_j (w_i . w_j)^2

        This measures how many "effective dimensions" each feature uses.
        """
        if W.shape[0] > W.shape[1]:
            W = W.T

        with torch.no_grad():
            norms = W.norm(dim=0)
            dot_products = W.T @ W
            dot_products_sq = dot_products ** 2

            d_eff = norms ** 4 / (dot_products_sq.sum(dim=1) + 1e-8)

        return d_eff

    def compute_superposition_score(
        self,
        W: torch.Tensor,
        sparsity: float,
    ) -> float:
        """
        Compute a superposition score based on theory.

        Higher scores indicate more features than would be possible
        without superposition.
        """
        if W.shape[0] > W.shape[1]:
            W = W.T

        hidden_dim, n_features = W.shape

        # Without superposition, can only represent hidden_dim features
        # With superposition, can represent more based on sparsity

        # Theoretical capacity with superposition
        # (from "Toy Models of Superposition" paper)
        theoretical_capacity = hidden_dim / (sparsity ** 2 + 1e-8)

        # Actual number of features with significant norm
        norms = W.norm(dim=0)
        threshold = norms.max() * 0.1
        active_features = (norms > threshold).sum().item()

        return active_features / hidden_dim

    def find_feature_clusters(
        self,
        W: torch.Tensor,
        threshold: float = 0.5,
    ) -> List[List[int]]:
        """
        Find clusters of interfering features.

        Features are clustered if their interference exceeds threshold.
        """
        geometry = self.analyze_weight_matrix(W)
        interference = geometry.interference_matrix

        n_features = interference.shape[0]

        # Build adjacency graph
        adjacent = (interference > threshold).cpu().numpy()
        np.fill_diagonal(adjacent, False)

        # Find connected components (simple BFS)
        visited = set()
        clusters = []

        for i in range(n_features):
            if i in visited:
                continue

            cluster = []
            queue = [i]

            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue

                visited.add(node)
                cluster.append(node)

                for j in range(n_features):
                    if adjacent[node, j] and j not in visited:
                        queue.append(j)

            if cluster:
                clusters.append(sorted(cluster))

        return clusters

    def analyze_layer(
        self,
        layer: nn.Module,
    ) -> Dict[str, Any]:
        """
        Analyze a single layer for superposition.
        """
        if isinstance(layer, nn.Linear):
            W = layer.weight
        else:
            raise ValueError(f"Unsupported layer type: {type(layer)}")

        geometry = self.analyze_weight_matrix(W)
        interference = self.compute_interference(W)
        d_eff = self.compute_effective_dimensionality(W)

        return {
            "geometry": geometry,
            "interference": interference,
            "effective_dimensionality": d_eff,
            "input_dim": W.shape[1],
            "output_dim": W.shape[0],
        }


def compute_feature_sparsity(
    activations: torch.Tensor,
    threshold: float = 1e-6,
) -> torch.Tensor:
    """
    Compute sparsity of feature activations.

    Returns fraction of samples where each feature is inactive.
    """
    active = (activations.abs() > threshold).float()
    sparsity = 1 - active.mean(dim=0)
    return sparsity


def compute_mutual_information(
    feature_i: torch.Tensor,
    feature_j: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """
    Estimate mutual information between two features.

    Uses binned histogram estimation.
    """
    # Discretize
    bins_i = torch.linspace(feature_i.min(), feature_i.max() + 1e-6, n_bins + 1)
    bins_j = torch.linspace(feature_j.min(), feature_j.max() + 1e-6, n_bins + 1)

    # Joint histogram
    joint = torch.zeros(n_bins, n_bins)
    for bi in range(n_bins):
        for bj in range(n_bins):
            mask_i = (feature_i >= bins_i[bi]) & (feature_i < bins_i[bi + 1])
            mask_j = (feature_j >= bins_j[bj]) & (feature_j < bins_j[bj + 1])
            joint[bi, bj] = (mask_i & mask_j).sum()

    # Normalize to get probability
    joint = joint / joint.sum()
    joint = joint + 1e-10  # Avoid log(0)

    # Marginals
    p_i = joint.sum(dim=1)
    p_j = joint.sum(dim=0)

    # Mutual information
    mi = 0.0
    for bi in range(n_bins):
        for bj in range(n_bins):
            if joint[bi, bj] > 1e-10:
                mi += joint[bi, bj] * torch.log(
                    joint[bi, bj] / (p_i[bi] * p_j[bj] + 1e-10)
                )

    return mi.item()


def compute_representation_similarity(
    model1_weights: torch.Tensor,
    model2_weights: torch.Tensor,
) -> float:
    """
    Compute similarity between two weight matrices.

    Uses centered kernel alignment (CKA).
    """
    def center_gram(K):
        n = K.shape[0]
        H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
        return H @ K @ H

    def hsic(K, L):
        n = K.shape[0]
        KL = center_gram(K) @ center_gram(L)
        return KL.trace() / ((n - 1) ** 2)

    K1 = model1_weights @ model1_weights.T
    K2 = model2_weights @ model2_weights.T

    cka = hsic(K1, K2) / (
        torch.sqrt(hsic(K1, K1) * hsic(K2, K2)) + 1e-8
    )

    return cka.item()


def analyze_sparsity_superposition_tradeoff(
    model_factory,
    sparsity_levels: List[float],
    n_steps: int = 5000,
    device: str = "cpu",
) -> Dict[str, List]:
    """
    Analyze how superposition varies with sparsity.

    Args:
        model_factory: Function that creates model given sparsity.
        sparsity_levels: List of sparsity values to test.
        n_steps: Training steps per model.
        device: Device to use.

    Returns:
        Dictionary with metrics at each sparsity level.
    """
    results = {
        "sparsity": [],
        "interference": [],
        "effective_dim": [],
        "loss": [],
    }

    analyzer = None

    for sparsity in sparsity_levels:
        model, losses = model_factory(sparsity, n_steps, device)

        if analyzer is None:
            analyzer = SuperpositionAnalyzer(model)
        else:
            analyzer.model = model

        # Get the weight matrix
        if hasattr(model, 'W'):
            W = model.W
        elif hasattr(model, 'encoder'):
            W = model.encoder.weight
        else:
            raise ValueError("Could not find weight matrix")

        interference = analyzer.compute_interference(W)
        d_eff = analyzer.compute_effective_dimensionality(W)

        results["sparsity"].append(sparsity)
        results["interference"].append(interference.total_interference)
        results["effective_dim"].append(d_eff.mean().item())
        results["loss"].append(losses[-1] if losses else 0)

    return results
