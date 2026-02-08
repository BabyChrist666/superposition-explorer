"""Tests for superposition_explorer.toy_model module."""

import pytest
import torch

from superposition_explorer.toy_model import (
    ToyModel,
    ToyModelConfig,
    generate_sparse_features,
    train_toy_model,
    SparseAutoencoder,
    SuperpositionTracker,
)


class TestToyModelConfig:
    def test_defaults(self):
        config = ToyModelConfig()
        assert config.n_features == 100
        assert config.n_hidden == 20
        assert config.sparsity == 0.1
        assert config.feature_importance is not None

    def test_custom_config(self):
        config = ToyModelConfig(n_features=50, n_hidden=10)
        assert config.n_features == 50
        assert config.n_hidden == 10

    def test_feature_importance_shape(self):
        config = ToyModelConfig(n_features=20)
        assert config.feature_importance.shape == (20,)


class TestToyModel:
    @pytest.fixture
    def config(self):
        return ToyModelConfig(n_features=20, n_hidden=5, sparsity=0.5)

    @pytest.fixture
    def model(self, config):
        return ToyModel(config)

    def test_create(self, model, config):
        assert model.W.shape == (config.n_hidden, config.n_features)
        assert model.b.shape == (config.n_features,)

    def test_encode(self, model):
        x = torch.randn(4, 20)
        h = model.encode(x)
        assert h.shape == (4, 5)

    def test_decode(self, model):
        h = torch.randn(4, 5)
        x_hat = model.decode(h)
        assert x_hat.shape == (4, 20)
        assert (x_hat >= 0).all()  # ReLU output

    def test_forward(self, model):
        x = torch.randn(4, 20).abs()  # Non-negative input
        x_hat = model(x)
        assert x_hat.shape == x.shape

    def test_get_feature_directions(self, model):
        directions = model.get_feature_directions()
        assert directions.shape == (20, 5)
        # Check normalization
        norms = directions.norm(dim=1)
        assert torch.allclose(norms, torch.ones(20), atol=1e-5)

    def test_get_feature_norms(self, model):
        norms = model.get_feature_norms()
        assert norms.shape == (20,)
        assert (norms >= 0).all()

    def test_get_interference_matrix(self, model):
        interference = model.get_interference_matrix()
        assert interference.shape == (20, 20)
        # Diagonal should be 1 (self-interference)
        assert torch.allclose(interference.diag(), torch.ones(20), atol=1e-5)
        # Off-diagonal should be between 0 and 1
        mask = 1 - torch.eye(20)
        assert (interference * mask >= 0).all()
        assert (interference * mask <= 1).all()

    def test_compute_loss(self, model, config):
        x = generate_sparse_features(32, config.n_features, config.sparsity)
        loss = model.compute_loss(x)
        assert loss.item() >= 0
        assert loss.requires_grad

    def test_analyze_superposition(self, model):
        analysis = model.analyze_superposition()
        assert "feature_norms" in analysis
        assert "interference_matrix" in analysis
        assert "avg_interference" in analysis
        assert "effective_dimensionality" in analysis
        assert analysis["superposition_ratio"] == 4.0  # 20/5


class TestGenerateSparseFeatures:
    def test_shape(self):
        x = generate_sparse_features(100, 20, 0.5)
        assert x.shape == (100, 20)

    def test_values_in_range(self):
        x = generate_sparse_features(100, 20, 0.5)
        assert (x >= 0).all()
        assert (x <= 1).all()

    def test_sparsity(self):
        x = generate_sparse_features(10000, 20, 0.9)
        # About 10% should be active
        active_ratio = (x > 0).float().mean().item()
        assert 0.05 < active_ratio < 0.15  # Allow some variance

    def test_device(self):
        x = generate_sparse_features(10, 5, 0.5, device="cpu")
        assert x.device == torch.device("cpu")


class TestTrainToyModel:
    def test_training_reduces_loss(self):
        config = ToyModelConfig(
            n_features=10, n_hidden=3,
            n_steps=100, batch_size=32
        )
        model, losses = train_toy_model(config, verbose=False)

        assert len(losses) == 100
        assert losses[-1] < losses[0]  # Loss should decrease

    def test_returns_trained_model(self):
        config = ToyModelConfig(
            n_features=10, n_hidden=3,
            n_steps=50, batch_size=16
        )
        model, _ = train_toy_model(config, verbose=False)

        assert isinstance(model, ToyModel)
        # Model should have learned something
        x = generate_sparse_features(32, 10, 0.5)
        loss = model.compute_loss(x)
        assert loss.item() < 0.5  # Reasonable reconstruction


class TestSparseAutoencoder:
    def test_create(self):
        sae = SparseAutoencoder(input_dim=100, hidden_dim=50)
        assert sae.input_dim == 100
        assert sae.hidden_dim == 50

    def test_forward(self):
        sae = SparseAutoencoder(input_dim=20, hidden_dim=10)
        x = torch.randn(4, 20)
        x_hat, h = sae(x)
        assert x_hat.shape == (4, 20)
        assert h.shape == (4, 10)
        assert (h >= 0).all()  # ReLU output

    def test_compute_loss(self):
        sae = SparseAutoencoder(input_dim=20, hidden_dim=10)
        x = torch.randn(4, 20)
        losses = sae.compute_loss(x)
        assert "total" in losses
        assert "reconstruction" in losses
        assert "sparsity" in losses


class TestSuperpositionTracker:
    def test_log(self):
        config = ToyModelConfig(n_features=10, n_hidden=3)
        model = ToyModel(config)
        tracker = SuperpositionTracker()

        tracker.log(model, 0.5)
        tracker.log(model, 0.3)

        assert len(tracker.history["loss"]) == 2
        assert tracker.history["loss"] == [0.5, 0.3]

    def test_get_summary(self):
        config = ToyModelConfig(n_features=10, n_hidden=3)
        model = ToyModel(config)
        tracker = SuperpositionTracker()

        tracker.log(model, 0.5)
        summary = tracker.get_summary()

        assert summary["final_loss"] == 0.5
        assert summary["n_steps"] == 1
