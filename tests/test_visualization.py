"""Tests for superposition_explorer.visualization module."""

import pytest
import torch

from superposition_explorer.visualization import (
    plot_feature_geometry,
    plot_interference_heatmap,
    plot_dimensionality_analysis,
    plot_sparsity_transition,
    plot_feature_similarity_matrix,
    plot_training_progress,
    plot_feature_clusters,
    create_summary_report,
)
from superposition_explorer.toy_model import ToyModel, ToyModelConfig


class TestPlotFeatureGeometry:
    def test_basic_plot(self):
        W = torch.randn(5, 10)  # 5 hidden, 10 features
        output = plot_feature_geometry(W, n_features=10)

        assert isinstance(output, str)
        assert "Feature Norms:" in output
        assert "F00" in output
        assert "Mean:" in output

    def test_truncates_features(self):
        W = torch.randn(5, 100)
        output = plot_feature_geometry(W, n_features=5)

        assert "F04" in output
        assert "F05" not in output


class TestPlotInterferenceHeatmap:
    def test_basic_heatmap(self):
        interference = torch.eye(10)
        output = plot_interference_heatmap(interference, n_features=10)

        assert isinstance(output, str)
        assert "Interference Heatmap:" in output
        assert "Legend:" in output

    def test_shows_self_interference(self):
        interference = torch.eye(5)
        output = plot_interference_heatmap(interference, n_features=5)

        # Diagonal should show 'x' for self-interference
        assert "x" in output


class TestPlotDimensionalityAnalysis:
    def test_basic_plot(self):
        d_eff = torch.ones(10) * 2.5
        output = plot_dimensionality_analysis(d_eff, n_features=10)

        assert isinstance(output, str)
        assert "Effective Dimensionality:" in output
        assert "Mean d_eff:" in output


class TestPlotSparsityTransition:
    def test_basic_plot(self):
        results = {
            "sparsity": [0.1, 0.3, 0.5, 0.7, 0.9],
            "interference": [0.5, 0.4, 0.3, 0.2, 0.1],
        }
        output = plot_sparsity_transition(results)

        assert isinstance(output, str)
        assert "Sparsity-Superposition Transition:" in output
        assert "S=0.10" in output

    def test_empty_data(self):
        results = {"sparsity": [], "interference": []}
        output = plot_sparsity_transition(results)
        assert "No data" in output


class TestPlotFeatureSimilarityMatrix:
    def test_basic_matrix(self):
        cosines = torch.eye(10)
        output = plot_feature_similarity_matrix(cosines, n_features=10)

        assert isinstance(output, str)
        assert "Feature Cosine Similarities:" in output
        assert "Legend:" in output


class TestPlotTrainingProgress:
    def test_basic_plot(self):
        losses = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.12, 0.1, 0.09]
        output = plot_training_progress(losses)

        assert isinstance(output, str)
        assert "Training Progress:" in output
        assert "Final loss:" in output

    def test_empty_losses(self):
        output = plot_training_progress([])
        assert "No training data" in output


class TestPlotFeatureClusters:
    def test_basic_clusters(self):
        clusters = [[0], [1, 2, 3], [4, 5], [6], [7], [8], [9]]
        output = plot_feature_clusters(clusters, n_features=10)

        assert isinstance(output, str)
        assert "Feature Clusters:" in output
        assert "Cluster" in output

    def test_shows_multi_feature_clusters(self):
        clusters = [[0, 1, 2, 3, 4]]  # One big cluster
        output = plot_feature_clusters(clusters, n_features=5)

        assert "F0, F1, F2, F3, F4" in output


class TestCreateSummaryReport:
    def test_with_model(self):
        config = ToyModelConfig(n_features=10, n_hidden=3)
        model = ToyModel(config)

        report = create_summary_report(model)

        assert isinstance(report, str)
        assert "SUPERPOSITION ANALYSIS REPORT" in report
        assert "Model Configuration" in report
        assert "Feature Geometry" in report
        assert "Interference" in report

    def test_with_losses(self):
        config = ToyModelConfig(n_features=10, n_hidden=3)
        model = ToyModel(config)
        losses = [0.5, 0.3, 0.2, 0.1]

        report = create_summary_report(model, losses=losses)

        assert "Training" in report
        assert "Final loss:" in report
        assert "Improvement:" in report
