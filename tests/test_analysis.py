"""Tests for superposition_explorer.analysis module."""

import pytest
import torch
import torch.nn as nn

from superposition_explorer.analysis import (
    SuperpositionAnalyzer,
    FeatureGeometry,
    InterferenceMetrics,
    compute_feature_sparsity,
    compute_mutual_information,
    compute_representation_similarity,
)
from superposition_explorer.toy_model import ToyModel, ToyModelConfig


class TestFeatureGeometry:
    def test_create(self):
        n_features = 10
        n_hidden = 5
        geometry = FeatureGeometry(
            directions=torch.randn(n_features, n_hidden),
            norms=torch.ones(n_features),
            cosine_similarities=torch.eye(n_features),
            interference_matrix=torch.eye(n_features),
        )
        assert geometry.directions.shape == (n_features, n_hidden)

    def test_to_dict(self):
        geometry = FeatureGeometry(
            directions=torch.randn(10, 5),
            norms=torch.ones(10) * 0.5,
            cosine_similarities=torch.eye(10),
            interference_matrix=torch.eye(10),
        )
        d = geometry.to_dict()
        assert "mean_norm" in d
        assert "mean_interference" in d


class TestInterferenceMetrics:
    def test_create(self):
        n_features = 10
        metrics = InterferenceMetrics(
            pairwise_interference=torch.eye(n_features),
            feature_interference=torch.zeros(n_features),
            total_interference=0.0,
            interference_rank=torch.arange(n_features),
        )
        assert metrics.total_interference == 0.0

    def test_to_dict(self):
        metrics = InterferenceMetrics(
            pairwise_interference=torch.eye(5),
            feature_interference=torch.ones(5) * 0.1,
            total_interference=0.5,
            interference_rank=torch.arange(5),
        )
        d = metrics.to_dict()
        assert d["total_interference"] == 0.5


class TestSuperpositionAnalyzer:
    @pytest.fixture
    def model(self):
        config = ToyModelConfig(n_features=20, n_hidden=5)
        return ToyModel(config)

    @pytest.fixture
    def analyzer(self, model):
        return SuperpositionAnalyzer(model)

    def test_analyze_weight_matrix(self, analyzer, model):
        geometry = analyzer.analyze_weight_matrix(model.W)

        assert geometry.directions.shape == (20, 5)
        assert geometry.norms.shape == (20,)
        assert geometry.cosine_similarities.shape == (20, 20)
        assert geometry.interference_matrix.shape == (20, 20)

    def test_compute_interference(self, analyzer, model):
        metrics = analyzer.compute_interference(model.W)

        assert metrics.pairwise_interference.shape == (20, 20)
        assert metrics.feature_interference.shape == (20,)
        assert 0 <= metrics.total_interference <= 1
        assert metrics.interference_rank.shape == (20,)

    def test_compute_effective_dimensionality(self, analyzer, model):
        d_eff = analyzer.compute_effective_dimensionality(model.W)

        assert d_eff.shape == (20,)
        assert (d_eff > 0).all()
        # For a model with more features than hidden dims, d_eff should be < hidden_dim
        assert d_eff.mean() <= 5  # hidden_dim

    def test_compute_superposition_score(self, analyzer, model):
        score = analyzer.compute_superposition_score(model.W, sparsity=0.5)
        assert score > 0

    def test_find_feature_clusters(self, analyzer, model):
        clusters = analyzer.find_feature_clusters(model.W, threshold=0.5)

        assert isinstance(clusters, list)
        # All features should be in exactly one cluster
        all_features = set()
        for cluster in clusters:
            for f in cluster:
                assert f not in all_features
                all_features.add(f)
        assert len(all_features) == 20


class TestComputeFeatureSparsity:
    def test_all_active(self):
        activations = torch.ones(100, 10)
        sparsity = compute_feature_sparsity(activations)
        assert sparsity.shape == (10,)
        assert torch.allclose(sparsity, torch.zeros(10))

    def test_all_inactive(self):
        activations = torch.zeros(100, 10)
        sparsity = compute_feature_sparsity(activations)
        assert torch.allclose(sparsity, torch.ones(10))

    def test_half_active(self):
        activations = torch.zeros(100, 10)
        activations[:50] = 1.0
        sparsity = compute_feature_sparsity(activations)
        assert torch.allclose(sparsity, torch.ones(10) * 0.5)


class TestComputeMutualInformation:
    def test_independent_features(self):
        # Random features should have low MI
        feature_i = torch.randn(1000)
        feature_j = torch.randn(1000)
        mi = compute_mutual_information(feature_i, feature_j)
        assert mi < 0.5  # Should be low for independent

    def test_identical_features(self):
        # Identical features should have high MI
        feature = torch.randn(1000)
        mi = compute_mutual_information(feature, feature)
        assert mi > 1.0  # Should be high for identical


class TestComputeRepresentationSimilarity:
    def test_identical_weights(self):
        W = torch.randn(10, 5)
        sim = compute_representation_similarity(W, W)
        assert abs(sim - 1.0) < 0.01  # Should be ~1.0

    def test_orthogonal_weights(self):
        # Create orthogonal matrices
        W1 = torch.eye(5)
        W2 = torch.zeros(5, 5)
        W2[0, 4] = 1
        W2[1, 3] = 1
        W2[2, 2] = 1
        W2[3, 1] = 1
        W2[4, 0] = 1

        sim = compute_representation_similarity(W1, W2)
        # Permuted identity should still have some similarity
        assert 0 <= sim <= 1


class TestAnalyzerWithLinearLayer:
    def test_analyze_linear_layer(self):
        layer = nn.Linear(10, 5)
        analyzer = SuperpositionAnalyzer(layer)

        result = analyzer.analyze_layer(layer)

        assert "geometry" in result
        assert "interference" in result
        assert "effective_dimensionality" in result
        assert result["input_dim"] == 10
        assert result["output_dim"] == 5
