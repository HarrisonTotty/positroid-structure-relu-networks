# Positroid Structure in ReLU Networks

Experiment code for a series of blog posts investigating the combinatorial geometry of trained neural networks:

1. [Positroid Structure in ReLU Networks](https://harrison.totty.dev/p/positroid-structure-relu-networks)
2. [The Hidden Geometry of ReLU Networks](https://harrison.totty.dev/p/hidden-geometry-relu-networks)
3. [Matroid-Guided Pruning: Removing 70% of Neurons for Free](https://harrison.totty.dev/p/matroid-guided-pruning)
4. [Chasing Geometric Structure in Transformers](https://harrison.totty.dev/p/chasing-geometric-structure-in-transformers)

## What this is about

A single-hidden-layer ReLU network with weight matrix $W$ and bias vector $b$ defines a hyperplane arrangement in input space. The combinatorial structure of that arrangement — which subsets of hyperplanes can meet at a point — is captured by an affine matroid.

Positroids are a special class of matroids arising from totally nonnegative matrices. They have rich structure (Grassmann necklaces, decorated permutations, plabic graphs) and show up throughout algebraic combinatorics.

**Key finding:** When ReLU networks are trained by gradient descent, the affine matroid of the first layer is *always* a positroid in our experiments (800+ trials across multiple datasets and architectures, 100% positroid rate). This holds even for unconstrained weights — training itself appears to impose positroid structure.

**But not always:** For arbitrary (non-trained) biases paired with totally positive weight matrices, positroid structure can fail. We construct 12,642 explicit counterexamples by deliberately choosing biases that create "crossing" non-basis patterns.

**Theorem (Contiguous-Implies-Positroid):** We prove that if every non-basis of a rank-$k$ matroid on $[n]$ is a cyclic interval $\{j, j{+}1, \ldots, j{+}k{-}1\} \bmod n$, then the matroid is a positroid. This cleanly explains the dichotomy: trained networks only produce cyclic-interval non-bases, while counterexamples require "spread" patterns.

**Pruning application:** The matroid structure tells us which neurons are essential (appear in every basis) vs. tail (never in any basis). Tail neurons can be removed for free — zero accuracy loss. Combining this with importance scoring yields a matroid-guided pruning strategy that removes 70% of neurons while preserving accuracy.

**Transformer extension:** We investigate whether positroid structure extends to transformer attention and MLP layers, using positroid-constrained attention (via Plücker coordinates), tropical MLP (determinant nonlinearity from boundary measurement), and positroid LoRA. The results are mostly negative — positroid constraints don't help transformers — but analysis of pretrained GPT-2 weights reveals approximate total positivity in early layers.

## Setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras
```

## Running experiments

Each experiment can be run via `just` or directly with `uv run python -m positroid.experiments.<name>`.

### Posts 1 & 2: Positroid structure in ReLU networks

#### Activation positroid (baseline)

Random TP weights + random biases. Shows that random biases mostly produce uniform (trivially positroid) matroids.

```bash
just experiment-positroid [ARGS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-dim` | `2` | Input dimension |
| `--hidden-dims` | `3 4 5 6` | Hidden dimensions to test (space-separated) |
| `--num-trials` | `100` | Number of trials per configuration |
| `--seed` | `42` | Random seed |
| `--detailed` | off | Print per-trial details for non-positroid cases |

#### Trained network positroid

Trains networks on real data with both TP-constrained and unconstrained weights, then checks positroid structure.

```bash
just experiment-trained-positroid [ARGS]
just experiment-trained-positroid-digits [ARGS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--datasets` | `moons circles` | Datasets to use (space-separated) |
| `--hidden-dims` | `6 8 10` | Hidden dimensions (space-separated) |
| `--num-trials` | `20` | Trials per configuration |
| `--n-samples` | `200` | Training samples per trial |
| `--epochs` | `200` | Training epochs |
| `--lr` | `0.01` | Learning rate |
| `--tp-kernel` | `exponential` | TP kernel: `exponential` or `cauchy` |
| `--seed` | `42` | Random seed |
| `--track-evolution` | off | Record matroid structure at snapshots during training |
| `--detailed` | off | Print per-trial details for non-uniform/non-positroid cases |

Available datasets: `moons`, `circles`, `spirals`, `xor`, `digits_0v1_pca2`, `digits_0v1_pca3`, `digits_0v1_pca5`, `digits_0v1_pca10`, `digits_3v8_pca2`, `digits_3v8_pca3`, `digits_3v8_pca5`, `digits_3v8_pca10`, `mnist_pca10`, `mnist_pca20`, `mnist_pca50`.

#### Counterexample search

Constructs TP weight matrices and deliberately chooses biases to produce non-positroid affine matroids.

```bash
just experiment-counterexample [ARGS]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--configs` | `2,5 2,6 2,8 2,10` | Configurations as `input_dim,hidden_dim` pairs |
| `--num-matrices` | `10` | TP matrices to generate per config per kernel |
| `--num-random` | `50` | Random bias trials per matrix (for `random` strategy) |
| `--strategies` | `targeted random` | Search strategies (space-separated) |
| `--kernels` | `exponential cauchy` | TP kernels (space-separated) |
| `--seed` | `42` | Random seed |
| `--detailed` | off | Print details for all non-uniform matroids |

#### Supporting experiments

```bash
# TP vs non-TP weight comparison across 8 parameterization modes
uv run python -m positroid.experiments.non_tp_baseline

# Epoch-by-epoch matroid evolution during training
uv run python -m positroid.experiments.matroid_evolution

# Rank-deficiency support analysis at scale
uv run python -m positroid.experiments.scale_evolution

# Multi-hidden-layer positroid analysis
uv run python -m positroid.experiments.multilayer

# MNIST multiclass classification
uv run python -m positroid.experiments.mnist_experiment

# Diagnose CREATED (non-positroid) cases
uv run python -m positroid.experiments.diagnose_created
```

### Post 3: Matroid-guided pruning

```bash
# 6 pruning strategies: matroid-guided, random, magnitude, activation, sensitivity, direction_replacement
uv run python -m positroid.experiments.pruning
```

### Post 4: Transformer experiments

```bash
# Positroid vs standard attention comparison
uv run python -m positroid.experiments.transformer_experiment

# Tropical MLP ablation (5 MLP variants)
uv run python -m positroid.experiments.tropical_mlp_ablation

# Optimization gap diagnosis (gradient norm tracking)
uv run python -m positroid.experiments.optimization_diagnosis

# Positroid cell network experiments (det/plucker_ratio/canonical_residue readouts)
uv run python -m positroid.experiments.positroid_network_experiment

# GPT-2 attention positroid analysis (requires torch, transformers)
uv run python -m positroid.experiments.attention_positroid

# GPT-2 weight matrix analysis (requires torch, transformers)
uv run python -m positroid.experiments.pretrained_analysis
```

### Figure generation

```bash
# Figures for posts 1 & 2
uv run python scripts/generate_blog_figures.py

# Figures for post 3 (pruning)
uv run python scripts/generate_pruning_figures.py

# Figures for post 4 (transformers)
uv run python scripts/generate_transformer_figures.py

# Pruning comparison animation (requires manim)
uv run python scripts/pruning_animation.py
```

## Running tests

```bash
just test            # 466 tests
just coverage        # with coverage report
just check           # lint + typecheck + test
```

## Project structure

```
src/positroid/
├── linalg/          # Minors, totally positive matrix construction & verification
├── matroid/         # Matroid, linear matroid, positroid (Grassmann necklace), plabic graphs
├── arrangement/     # Hyperplane arrangements and their affine matroids
├── network/         # ReLU networks, TP-constrained networks, positroid cell networks
├── positroid_cell/  # Boundary measurement (Marsh-Rietsch), Plücker coordinates
├── transformer/     # Positroid attention, tropical MLP, det MLP, LoRA, MoE, analysis
├── datasets/        # Toy 2D, PCA-reduced digits, MNIST
└── experiments/     # All experiments described above

scripts/
├── generate_blog_figures.py        # Figures for posts 1 & 2
├── generate_pruning_figures.py     # Figures for post 3
├── generate_transformer_figures.py # Figures for post 4
└── pruning_animation.py            # Manim animation for pruning comparison
```

## License

MIT
