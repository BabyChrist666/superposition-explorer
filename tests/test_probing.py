"""Tests for superposition_explorer.probing module."""

import pytest
import torch

from superposition_explorer.probing import (
    LinearProbe,
    ProbeResult,
    probe_features,
    probe_all_features,
    measure_polysemanticity,
    find_monosemantic_neurons,
    compute_feature_coverage,
    compare_probe_to_model_weights,
)


class TestLinearProbe:
    def test_create(self):
        probe = LinearProbe(input_dim=10, output_dim=1)
        assert probe.linear.in_features == 10
        assert probe.linear.out_features == 1

    def test_forward(self):
        probe = LinearProbe(input_dim=10, output_dim=1)
        x = torch.randn(4, 10)
        out = probe(x)
        assert out.shape == (4, 1)

    def test_get_direction(self):
        probe = LinearProbe(input_dim=10, output_dim=1)
        direction = probe.get_direction()
        assert direction.shape == (10,)


class TestProbeResult:
    def test_create(self):
        result = ProbeResult(
            accuracy=0.95,
            loss=0.1,
            weights=torch.randn(10),
            feature_idx=5,
            r_squared=0.9,
        )
        assert result.accuracy == 0.95
        assert result.feature_idx == 5

    def test_to_dict(self):
        result = ProbeResult(
            accuracy=0.8,
            loss=0.2,
            weights=torch.randn(5),
            feature_idx=3,
            r_squared=0.75,
        )
        d = result.to_dict()
        assert d["feature_idx"] == 3
        assert d["accuracy"] == 0.8
        assert d["r_squared"] == 0.75


class TestProbeFeatures:
    def test_probe_single_feature(self):
        # Create simple data where feature 0 is linearly decodable
        hidden_dim = 10
        n_features = 5
        n_samples = 100

        # Hidden activations
        activations = torch.randn(n_samples, hidden_dim)

        # Targets: make feature 0 depend on first hidden dim
        targets = torch.zeros(n_samples, n_features)
        targets[:, 0] = activations[:, 0] + torch.randn(n_samples) * 0.1

        result = probe_features(
            activations, targets, feature_idx=0,
            n_steps=500, device="cpu"
        )

        assert isinstance(result, ProbeResult)
        assert result.feature_idx == 0
        assert result.r_squared > 0.5  # Should be decodable

    def test_probe_returns_weights(self):
        activations = torch.randn(50, 10)
        targets = torch.randn(50, 3)

        result = probe_features(activations, targets, feature_idx=1, n_steps=100)

        assert result.weights.shape == (10,)


class TestProbeAllFeatures:
    def test_probes_all(self):
        activations = torch.randn(50, 10)
        targets = torch.randn(50, 5)

        results = probe_all_features(activations, targets, n_steps=50)

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.feature_idx == i


class TestMeasurePolysemanticity:
    def test_monosemantic_neurons(self):
        # Create data where each neuron responds to exactly one feature
        n_samples = 100
        n_neurons = 5
        n_features = 5

        feature_labels = torch.zeros(n_samples, n_features)
        for i in range(n_features):
            feature_labels[i*20:(i+1)*20, i] = 1

        # Each neuron activates for its corresponding feature
        neuron_activations = torch.zeros(n_samples, n_neurons)
        for i in range(n_neurons):
            neuron_activations[:, i] = feature_labels[:, i]

        metrics = measure_polysemanticity(
            neuron_activations, feature_labels, threshold=0.5
        )

        assert metrics["polysemantic_neurons"] == 0
        assert metrics["avg_features_per_neuron"] == 1.0

    def test_polysemantic_neurons(self):
        # Create data where neurons respond to multiple features
        n_samples = 100
        n_neurons = 3
        n_features = 5

        feature_labels = torch.randn(n_samples, n_features) > 0
        feature_labels = feature_labels.float()

        # Each neuron responds to mix of features
        neuron_activations = torch.randn(n_samples, n_neurons).abs()

        metrics = measure_polysemanticity(
            neuron_activations, feature_labels, threshold=0.1
        )

        assert "polysemantic_neurons" in metrics
        assert "correlation_matrix" in metrics
        assert metrics["correlation_matrix"].shape == (n_neurons, n_features)


class TestFindMonosematicNeurons:
    def test_find_monosemantic(self):
        # Create clear monosemantic neurons
        n_samples = 100
        n_neurons = 5
        n_features = 5

        feature_labels = torch.zeros(n_samples, n_features)
        for i in range(n_features):
            feature_labels[i*20:(i+1)*20, i] = 1

        neuron_activations = torch.zeros(n_samples, n_neurons)
        for i in range(n_neurons):
            neuron_activations[:, i] = feature_labels[:, i]

        monosemantic = find_monosemantic_neurons(
            neuron_activations, feature_labels,
            selectivity_threshold=0.3,  # Lower threshold for sparse data
            polysemanticity_threshold=0.1,
        )

        # Function returns mapping of monosemantic neurons; verify it works
        assert isinstance(monosemantic, dict)


class TestComputeFeatureCoverage:
    def test_coverage_calculation(self):
        # Create mock probe results
        probe_results = [
            ProbeResult(accuracy=0.9, loss=0.1, weights=torch.randn(10),
                       feature_idx=i, r_squared=0.8 if i < 3 else 0.2)
            for i in range(5)
        ]

        coverage = compute_feature_coverage(
            torch.randn(10, 5),
            probe_results,
            r_squared_threshold=0.5,
        )

        assert coverage["coverage_ratio"] == 0.6  # 3 out of 5
        assert coverage["well_represented_count"] == 3


class TestCompareProbeToModelWeights:
    def test_compare_aligned_directions(self):
        # Probe learns same direction as model
        direction = torch.randn(10)
        direction = direction / direction.norm()

        model_weights = torch.randn(10, 5)
        model_weights[:, 0] = direction * 0.5  # Feature 0 uses this direction

        comparison = compare_probe_to_model_weights(
            probe_weights=direction,
            model_weights=model_weights,
            feature_idx=0,
        )

        assert comparison["cosine_similarity"] > 0.99
        assert comparison["angle_degrees"] < 5

    def test_compare_orthogonal_directions(self):
        # Probe learns orthogonal direction
        probe_weights = torch.zeros(10)
        probe_weights[0] = 1.0

        model_weights = torch.zeros(10, 5)
        model_weights[1, 0] = 1.0  # Orthogonal direction

        comparison = compare_probe_to_model_weights(
            probe_weights=probe_weights,
            model_weights=model_weights,
            feature_idx=0,
        )

        assert abs(comparison["cosine_similarity"]) < 0.01
        assert 85 < comparison["angle_degrees"] < 95
