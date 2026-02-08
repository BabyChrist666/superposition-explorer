# Superposition Explorer

Investigate polysemanticity and feature superposition in neural networks. Tools for studying how networks encode more features than dimensions.

[![Tests](https://github.com/BabyChrist666/superposition-explorer/actions/workflows/tests.yml/badge.svg)](https://github.com/BabyChrist666/superposition-explorer/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/BabyChrist666/superposition-explorer/branch/master/graph/badge.svg)](https://codecov.io/gh/BabyChrist666/superposition-explorer)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Neural networks often learn to represent more features than they have dimensions - a phenomenon called **superposition**. This library provides tools to:

- Train toy models that exhibit superposition
- Analyze interference between features
- Visualize feature geometry and clustering
- Probe for polysemantic neurons
- Measure effective dimensionality

Based on Anthropic's "Toy Models of Superposition" paper.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from superposition_explorer import (
    ToyModel, ToyModelConfig,
    train_toy_model, generate_sparse_features,
    SuperpositionAnalyzer,
)

# Configure model: 100 features in 20 dimensions (5x superposition)
config = ToyModelConfig(
    n_features=100,
    n_hidden=20,
    sparsity=0.9,  # 90% of features inactive
    n_steps=10000,
)

# Train the model
model, losses = train_toy_model(config, verbose=True)

# Analyze superposition
analyzer = SuperpositionAnalyzer(model)
interference = analyzer.compute_interference(model.W)

print(f"Total interference: {interference.total_interference:.4f}")
print(f"Mean per-feature: {interference.feature_interference.mean():.4f}")
```

## Core Concepts

### Superposition

When features are sparse, networks can represent more features than dimensions by using overlapping (non-orthogonal) directions:

```
Without superposition: n_features <= n_hidden
With superposition:    n_features >> n_hidden (if features are sparse)
```

### Interference Matrix

Measures how much features interfere with each other:

```python
from superposition_explorer import SuperpositionAnalyzer

analyzer = SuperpositionAnalyzer(model)
geometry = analyzer.analyze_weight_matrix(model.W)

# I[i,j] = (w_i . w_j)^2 / (||w_i|| ||w_j||)^2
interference = geometry.interference_matrix
```

### Effective Dimensionality

How many "effective dimensions" each feature uses:

```python
d_eff = analyzer.compute_effective_dimensionality(model.W)
# d_eff < 1 means feature shares space with others
```

## Toy Model

The toy model is a simple bottleneck autoencoder:

```python
from superposition_explorer import ToyModel, ToyModelConfig

config = ToyModelConfig(
    n_features=100,     # Input/output features
    n_hidden=20,        # Bottleneck dimension
    sparsity=0.9,       # Feature sparsity
    feature_importance=torch.tensor([0.7**i for i in range(100)]),
)

model = ToyModel(config)

# Forward pass: encode then decode
x_hat = model(x)

# Get feature directions in hidden space
directions = model.get_feature_directions()  # [n_features, n_hidden]
norms = model.get_feature_norms()            # [n_features]
```

## Analysis Tools

### Feature Geometry

```python
from superposition_explorer import SuperpositionAnalyzer

analyzer = SuperpositionAnalyzer(model)
geometry = analyzer.analyze_weight_matrix(model.W)

print(f"Mean norm: {geometry.norms.mean():.4f}")
print(f"Mean interference: {geometry.interference_matrix.mean():.4f}")
```

### Feature Clustering

Find features that interfere with each other:

```python
clusters = analyzer.find_feature_clusters(model.W, threshold=0.5)
# [[0, 5, 12], [1, 3], [2], ...]  # Features that share space
```

### Polysemanticity Measurement

```python
from superposition_explorer import measure_polysemanticity

metrics = measure_polysemanticity(
    neuron_activations=hidden_acts,
    feature_labels=input_features,
    threshold=0.1,
)

print(f"Polysemantic neurons: {metrics['polysemantic_neurons']}")
print(f"Avg features per neuron: {metrics['avg_features_per_neuron']:.2f}")
```

## Visualization (ASCII)

```python
from superposition_explorer.visualization import (
    plot_feature_geometry,
    plot_interference_heatmap,
    create_summary_report,
)

# Feature norms as bar chart
print(plot_feature_geometry(model.W))

# Interference as ASCII heatmap
print(plot_interference_heatmap(geometry.interference_matrix))

# Full analysis report
print(create_summary_report(model, losses))
```

## Linear Probing

Test whether features are linearly decodable:

```python
from superposition_explorer import probe_features, probe_all_features

# Probe for feature 5
result = probe_features(
    activations=hidden_acts,
    targets=input_features,
    feature_idx=5,
)

print(f"R-squared: {result.r_squared:.3f}")
print(f"Probe direction: {result.weights}")
```

## Key Findings from the Paper

1. **Sparsity enables superposition**: More sparse features = more superposition possible
2. **Importance matters**: More important features get more orthogonal directions
3. **Phase transitions**: Sudden transitions between monosemantic and polysemantic representations
4. **Feature geometry**: Features form regular polytopes in weight space

## References

- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) - Anthropic
- [Superposition, Memorization, and Double Descent](https://arxiv.org/abs/2210.01891)

## License

MIT
