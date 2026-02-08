"""
Superposition Explorer - Investigate polysemanticity and feature superposition.

Tools for studying how neural networks encode more features than dimensions,
and how to disentangle overlapping feature representations.
"""

from superposition_explorer.toy_model import (
    ToyModel,
    ToyModelConfig,
    generate_sparse_features,
    train_toy_model,
)
from superposition_explorer.analysis import (
    SuperpositionAnalyzer,
    FeatureGeometry,
    InterferenceMetrics,
    compute_feature_sparsity,
)
from superposition_explorer.visualization import (
    plot_feature_geometry,
    plot_interference_heatmap,
    plot_dimensionality_analysis,
    plot_sparsity_transition,
)
from superposition_explorer.probing import (
    LinearProbe,
    ProbeResult,
    probe_features,
    measure_polysemanticity,
)

__version__ = "0.1.0"
__all__ = [
    # Toy Model
    "ToyModel",
    "ToyModelConfig",
    "generate_sparse_features",
    "train_toy_model",
    # Analysis
    "SuperpositionAnalyzer",
    "FeatureGeometry",
    "InterferenceMetrics",
    "compute_feature_sparsity",
    # Visualization
    "plot_feature_geometry",
    "plot_interference_heatmap",
    "plot_dimensionality_analysis",
    "plot_sparsity_transition",
    # Probing
    "LinearProbe",
    "ProbeResult",
    "probe_features",
    "measure_polysemanticity",
]
