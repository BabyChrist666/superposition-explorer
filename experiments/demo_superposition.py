"""
Demonstration of Superposition Explorer.

Shows how to train toy models and analyze superposition.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from superposition_explorer import (
    ToyModel, ToyModelConfig,
    train_toy_model, generate_sparse_features,
    SuperpositionAnalyzer,
)
from superposition_explorer.visualization import (
    plot_feature_geometry,
    plot_interference_heatmap,
    plot_dimensionality_analysis,
    plot_training_progress,
    create_summary_report,
)
from superposition_explorer.probing import (
    probe_features,
    measure_polysemanticity,
)


def demo_toy_model_training():
    """Train a toy model and observe superposition."""
    print("=" * 60)
    print("DEMO 1: Training Toy Model")
    print("=" * 60)

    config = ToyModelConfig(
        n_features=50,
        n_hidden=10,
        sparsity=0.8,
        n_steps=2000,
        batch_size=256,
    )

    print(f"\nConfiguration:")
    print(f"  Features: {config.n_features}")
    print(f"  Hidden: {config.n_hidden}")
    print(f"  Superposition ratio: {config.n_features / config.n_hidden:.1f}x")
    print(f"  Sparsity: {config.sparsity}")

    print("\nTraining...")
    model, losses = train_toy_model(config, verbose=False)

    print(f"\nTraining complete!")
    print(f"  Initial loss: {losses[0]:.6f}")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Improvement: {(1 - losses[-1]/losses[0]) * 100:.1f}%")

    return model, losses, config


def demo_analysis(model):
    """Analyze superposition in the trained model."""
    print("\n" + "=" * 60)
    print("DEMO 2: Superposition Analysis")
    print("=" * 60)

    analyzer = SuperpositionAnalyzer(model)

    print("\n2.1 Feature Geometry:")
    geometry = analyzer.analyze_weight_matrix(model.W)
    print(f"  Mean feature norm: {geometry.norms.mean():.4f}")
    print(f"  Std feature norm: {geometry.norms.std():.4f}")
    print(f"  Max norm: {geometry.norms.max():.4f}")
    print(f"  Min norm: {geometry.norms.min():.4f}")

    print("\n2.2 Interference Analysis:")
    interference = analyzer.compute_interference(model.W)
    print(f"  Total interference: {interference.total_interference:.4f}")
    print(f"  Mean per-feature: {interference.feature_interference.mean():.4f}")
    print(f"  Max per-feature: {interference.feature_interference.max():.4f}")

    print("\n2.3 Effective Dimensionality:")
    d_eff = analyzer.compute_effective_dimensionality(model.W)
    print(f"  Mean d_eff: {d_eff.mean():.3f}")
    print(f"  Min d_eff: {d_eff.min():.3f}")
    print(f"  Max d_eff: {d_eff.max():.3f}")

    print("\n2.4 Feature Clusters:")
    clusters = analyzer.find_feature_clusters(model.W, threshold=0.3)
    multi_clusters = [c for c in clusters if len(c) > 1]
    print(f"  Total clusters: {len(clusters)}")
    print(f"  Multi-feature clusters: {len(multi_clusters)}")
    if multi_clusters:
        print(f"  Largest cluster size: {max(len(c) for c in multi_clusters)}")

    return geometry, interference, d_eff


def demo_visualization(model, losses):
    """Show ASCII visualizations."""
    print("\n" + "=" * 60)
    print("DEMO 3: Visualizations")
    print("=" * 60)

    print("\n3.1 Training Progress:")
    print(plot_training_progress(losses[:100]))  # Sample for brevity

    print("\n3.2 Feature Norms (first 15 features):")
    print(plot_feature_geometry(model.W, n_features=15, width=50))

    analyzer = SuperpositionAnalyzer(model)
    geometry = analyzer.analyze_weight_matrix(model.W)

    print("\n3.3 Interference Heatmap (10x10):")
    print(plot_interference_heatmap(geometry.interference_matrix, n_features=10))

    d_eff = analyzer.compute_effective_dimensionality(model.W)
    print("\n3.4 Effective Dimensionality (first 15 features):")
    print(plot_dimensionality_analysis(d_eff, n_features=15, width=50))


def demo_probing(model, config):
    """Demonstrate linear probing."""
    print("\n" + "=" * 60)
    print("DEMO 4: Linear Probing")
    print("=" * 60)

    # Generate test data
    x = generate_sparse_features(500, config.n_features, config.sparsity)
    h = model.encode(x)

    print("\n4.1 Probing for individual features:")
    for i in [0, 10, 20]:
        result = probe_features(
            h.detach(), x.detach(),
            feature_idx=i,
            n_steps=200,
        )
        print(f"  Feature {i}: R^2 = {result.r_squared:.3f}, Loss = {result.loss:.4f}")

    print("\n4.2 Polysemanticity Analysis:")
    # Create binary feature labels
    feature_active = (x > 0.1).float()

    metrics = measure_polysemanticity(
        h.detach(), feature_active.detach(),
        threshold=0.2,
    )

    print(f"  Polysemantic neurons: {metrics['polysemantic_neurons']}")
    print(f"  Polysemanticity ratio: {metrics['polysemanticity_ratio']:.2%}")
    print(f"  Avg features per neuron: {metrics['avg_features_per_neuron']:.2f}")
    print(f"  Avg selectivity: {metrics['avg_selectivity']:.3f}")


def demo_sparsity_comparison():
    """Compare superposition at different sparsity levels."""
    print("\n" + "=" * 60)
    print("DEMO 5: Sparsity vs Superposition")
    print("=" * 60)

    sparsity_levels = [0.5, 0.7, 0.9, 0.95]

    print("\nTraining models at different sparsity levels...")
    results = []

    for sparsity in sparsity_levels:
        config = ToyModelConfig(
            n_features=30,
            n_hidden=10,
            sparsity=sparsity,
            n_steps=1000,
        )
        model, losses = train_toy_model(config, verbose=False)

        analyzer = SuperpositionAnalyzer(model)
        interference = analyzer.compute_interference(model.W)
        d_eff = analyzer.compute_effective_dimensionality(model.W)

        results.append({
            "sparsity": sparsity,
            "interference": interference.total_interference,
            "d_eff": d_eff.mean().item(),
            "loss": losses[-1],
        })

    print("\n        Sparsity  Interference  Eff.Dim  Loss")
    print("        " + "-" * 42)
    for r in results:
        print(f"        {r['sparsity']:.2f}      {r['interference']:.4f}       {r['d_eff']:.3f}    {r['loss']:.5f}")

    print("\n  -> Higher sparsity = lower interference = more features packed")


def main():
    print("\n" + "#" * 60)
    print("#" + " " * 18 + "SUPERPOSITION EXPLORER" + " " * 16 + "#")
    print("#" + " " * 11 + "Feature Superposition in Neural Networks" + " " * 5 + "#")
    print("#" * 60)

    model, losses, config = demo_toy_model_training()
    demo_analysis(model)
    demo_visualization(model, losses)
    demo_probing(model, config)
    demo_sparsity_comparison()

    print("\n" + "=" * 60)
    print("Summary Report:")
    print("=" * 60)
    print(create_summary_report(model, losses))

    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
