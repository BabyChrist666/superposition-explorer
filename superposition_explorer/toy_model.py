"""
Toy models for studying superposition.

Implements the bottleneck autoencoder from Anthropic's superposition paper
and utilities for generating sparse feature data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import numpy as np


@dataclass
class ToyModelConfig:
    """Configuration for the toy superposition model."""
    n_features: int = 100  # Number of input features
    n_hidden: int = 20     # Hidden dimension (bottleneck)
    sparsity: float = 0.1  # Feature sparsity (prob of being active)
    feature_importance: Optional[torch.Tensor] = None  # Importance weights

    # Training config
    learning_rate: float = 1e-3
    batch_size: int = 256
    n_steps: int = 10000

    def __post_init__(self):
        if self.feature_importance is None:
            # Exponentially decaying importance (like in the paper)
            self.feature_importance = torch.tensor([
                0.7 ** i for i in range(self.n_features)
            ])


class ToyModel(nn.Module):
    """
    Toy model for studying superposition.

    A simple bottleneck autoencoder that learns to compress n_features
    into n_hidden dimensions, demonstrating superposition when n_features > n_hidden.

    Based on Anthropic's "Toy Models of Superposition" paper.
    """

    def __init__(self, config: ToyModelConfig):
        super().__init__()
        self.config = config

        # Encoder and decoder (tied weights)
        self.W = nn.Parameter(torch.randn(config.n_hidden, config.n_features) * 0.1)
        self.b = nn.Parameter(torch.zeros(config.n_features))

    @property
    def encoder(self) -> torch.Tensor:
        """W encodes features to hidden space."""
        return self.W

    @property
    def decoder(self) -> torch.Tensor:
        """W^T decodes hidden space back to features."""
        return self.W.T

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input features to hidden representation."""
        return x @ self.W.T

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Decode hidden representation back to features."""
        return F.relu(h @ self.W + self.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode then decode."""
        h = self.encode(x)
        return self.decode(h)

    def get_feature_directions(self) -> torch.Tensor:
        """Get the direction each feature uses in hidden space."""
        # Normalize each column of W to get unit directions
        W_norm = self.W / (self.W.norm(dim=0, keepdim=True) + 1e-8)
        return W_norm.T  # Shape: [n_features, n_hidden]

    def get_feature_norms(self) -> torch.Tensor:
        """Get the magnitude each feature has in hidden space."""
        return self.W.norm(dim=0)

    def get_interference_matrix(self) -> torch.Tensor:
        """
        Compute pairwise interference between features.

        Returns matrix I where I[i,j] = (w_i . w_j)^2 / (||w_i|| ||w_j||)^2
        This measures how much feature i interferes with feature j.
        """
        W_norm = self.W / (self.W.norm(dim=0, keepdim=True) + 1e-8)
        cosines = W_norm.T @ W_norm  # Cosine similarities
        return cosines ** 2  # Squared cosines = interference

    def compute_loss(
        self,
        x: torch.Tensor,
        importance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute weighted reconstruction loss.

        Loss = sum_i importance_i * (x_i - x_hat_i)^2
        """
        x_hat = self(x)

        if importance is None:
            importance = self.config.feature_importance.to(x.device)

        # Weighted MSE
        diff = (x - x_hat) ** 2
        weighted_diff = diff * importance.unsqueeze(0)
        return weighted_diff.mean()

    def analyze_superposition(self) -> Dict[str, Any]:
        """Analyze how much superposition the model exhibits."""
        with torch.no_grad():
            # Feature norms (how much capacity each feature uses)
            norms = self.get_feature_norms()

            # Interference matrix
            interference = self.get_interference_matrix()

            # Average interference per feature (excluding self)
            mask = 1 - torch.eye(self.config.n_features, device=interference.device)
            avg_interference = (interference * mask).sum(dim=1) / (self.config.n_features - 1)

            # Effective dimensionality used by each feature
            # d_eff = ||w||^2 / sum_j (w.w_j)^2
            dot_products_sq = (self.W.T @ self.W) ** 2
            d_eff = norms ** 2 / (dot_products_sq.sum(dim=1) + 1e-8)

            return {
                "feature_norms": norms,
                "interference_matrix": interference,
                "avg_interference": avg_interference,
                "effective_dimensionality": d_eff,
                "total_capacity_used": norms.sum().item(),
                "superposition_ratio": self.config.n_features / self.config.n_hidden,
            }


def generate_sparse_features(
    n_samples: int,
    n_features: int,
    sparsity: float,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate sparse feature activations.

    Each feature is active with probability (1 - sparsity), and when active
    has a value uniformly sampled from [0, 1].
    """
    # Generate mask (which features are active)
    mask = (torch.rand(n_samples, n_features, device=device) > sparsity).float()

    # Generate values (uniform [0, 1])
    values = torch.rand(n_samples, n_features, device=device)

    return mask * values


def train_toy_model(
    config: ToyModelConfig,
    device: str = "cpu",
    verbose: bool = True,
    log_every: int = 1000,
) -> Tuple[ToyModel, List[float]]:
    """
    Train a toy superposition model.

    Args:
        config: Model configuration.
        device: Device to train on.
        verbose: Whether to print progress.
        log_every: How often to log.

    Returns:
        Trained model and list of losses.
    """
    model = ToyModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    losses = []
    importance = config.feature_importance.to(device)

    for step in range(config.n_steps):
        # Generate batch
        x = generate_sparse_features(
            config.batch_size,
            config.n_features,
            config.sparsity,
            device=device,
        )

        # Forward and backward
        optimizer.zero_grad()
        loss = model.compute_loss(x, importance)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if verbose and (step + 1) % log_every == 0:
            avg_loss = sum(losses[-log_every:]) / log_every
            print(f"Step {step + 1}/{config.n_steps}, Loss: {avg_loss:.6f}")

    return model, losses


class SparseAutoencoder(nn.Module):
    """
    Sparse autoencoder for feature extraction.

    Used to find interpretable directions in the hidden space
    that correspond to individual features.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        sparsity_coef: float = 1e-3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_coef = sparsity_coef

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning reconstruction and hidden activations.
        """
        h = F.relu(self.encoder(x))
        x_hat = self.decoder(h)
        return x_hat, h

    def compute_loss(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute reconstruction + sparsity loss."""
        x_hat, h = self(x)

        recon_loss = F.mse_loss(x_hat, x)
        sparsity_loss = h.abs().mean()  # L1 sparsity

        total_loss = recon_loss + self.sparsity_coef * sparsity_loss

        return {
            "total": total_loss,
            "reconstruction": recon_loss,
            "sparsity": sparsity_loss,
        }


class SuperpositionTracker:
    """
    Track superposition metrics during training.
    """

    def __init__(self):
        self.history = {
            "loss": [],
            "feature_norms": [],
            "interference": [],
            "effective_dim": [],
        }

    def log(self, model: ToyModel, loss: float):
        """Log metrics from current model state."""
        self.history["loss"].append(loss)

        with torch.no_grad():
            analysis = model.analyze_superposition()
            self.history["feature_norms"].append(
                analysis["feature_norms"].cpu().numpy()
            )
            self.history["interference"].append(
                analysis["avg_interference"].mean().item()
            )
            self.history["effective_dim"].append(
                analysis["effective_dimensionality"].mean().item()
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of tracked metrics."""
        return {
            "final_loss": self.history["loss"][-1] if self.history["loss"] else None,
            "final_interference": self.history["interference"][-1] if self.history["interference"] else None,
            "final_effective_dim": self.history["effective_dim"][-1] if self.history["effective_dim"] else None,
            "n_steps": len(self.history["loss"]),
        }
