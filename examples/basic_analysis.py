#!/usr/bin/env python3
"""
Basic Superposition Analysis Example

This example demonstrates how to investigate polysemanticity
and feature superposition in neural network representations.
"""

import numpy as np
from superposition_explorer import (
    SuperpositionAnalyzer,
    FeatureProbe,
    ActivationCollector,
    SuperpositionMetrics,
)


def main():
    print("=" * 60)
    print("Superposition Explorer - Basic Analysis")
    print("=" * 60)

    # Create mock activations (in practice, extract from real model)
    n_samples = 1000
    n_features = 512
    activations = np.random.randn(n_samples, n_features)

    # Initialize analyzer
    analyzer = SuperpositionAnalyzer(
        n_features=n_features,
        n_concepts=100,  # Estimated number of concepts
    )

    # Example 1: Compute superposition metrics
    print("\n1. Computing superposition metrics...")
    metrics = analyzer.compute_metrics(activations)

    print(f"   Dimensionality: {metrics.effective_dimensionality:.2f}")
    print(f"   Sparsity: {metrics.sparsity:.4f}")
    print(f"   Feature interference: {metrics.interference:.4f}")
    print(f"   Superposition ratio: {metrics.superposition_ratio:.4f}")

    # Example 2: Find polysemantic neurons
    print("\n2. Finding polysemantic neurons...")

    # Create synthetic concept labels
    concept_labels = np.random.randint(0, 10, n_samples)

    polysemantic = analyzer.find_polysemantic_neurons(
        activations,
        concept_labels,
        threshold=0.3,
    )

    print(f"   Found {len(polysemantic)} polysemantic neurons")
    print(f"   Top 5 polysemantic: {polysemantic[:5]}")

    # Example 3: Compute feature geometry
    print("\n3. Analyzing feature geometry...")

    geometry = analyzer.compute_feature_geometry(activations)

    print(f"   Average cosine similarity: {geometry.avg_cosine_sim:.4f}")
    print(f"   Feature clustering coefficient: {geometry.clustering:.4f}")
    print(f"   Nearest neighbor distance: {geometry.nn_distance:.4f}")

    # Example 4: Visualize superposition
    print("\n4. Creating visualizations...")

    # Save feature space visualization
    analyzer.plot_feature_space(
        activations,
        save_path="feature_space.png",
        method="pca",
    )
    print("   Saved feature_space.png")

    # Save interference matrix
    analyzer.plot_interference_matrix(
        activations,
        save_path="interference_matrix.png",
    )
    print("   Saved interference_matrix.png")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
