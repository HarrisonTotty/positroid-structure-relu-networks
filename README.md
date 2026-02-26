# Positroid Structure in ReLU Networks

Experiment code for two blog posts investigating the combinatorial geometry of trained ReLU networks:

1. [The Hidden Geometry of ReLU Networks](https://harrison.totty.dev/p/hidden-geometry-relu-networks)
2. [Positroid Structure in ReLU Networks](https://harrison.totty.dev/p/positroid-structure-relu-networks)

## What this is about

A single-hidden-layer ReLU network with weight matrix $W$ and bias vector $b$ defines a hyperplane arrangement in input space. The combinatorial structure of that arrangement — which subsets of hyperplanes can meet at a point — is captured by an affine matroid.

Positroids are a special class of matroids arising from totally nonnegative matrices. They have rich structure (Grassmann necklaces, decorated permutations, plabic graphs) and show up throughout algebraic combinatorics.

**Key finding:** When ReLU networks are trained by gradient descent, the affine matroid of the first layer is *always* a positroid in our experiments (800+ trials across multiple datasets and architectures, 100% positroid rate). This holds even for unconstrained weights — training itself appears to impose positroid structure.

**But not always:** For arbitrary (non-trained) biases paired with totally positive weight matrices, positroid structure can fail. We construct 12,642 explicit counterexamples by deliberately choosing biases that create "crossing" non-basis patterns.

**Theorem (Contiguous-Implies-Positroid):** We prove that if every non-basis of a rank-$k$ matroid on $[n]$ is a cyclic interval $\{j, j{+}1, \ldots, j{+}k{-}1\} \bmod n$, then the matroid is a positroid. This cleanly explains the dichotomy: trained networks only produce cyclic-interval non-bases, while counterexamples require "spread" patterns.

## Setup

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras
```

## Running experiments

Each experiment can be run via `just` or directly with `uv run python -m positroid.experiments.<name>`.

### Activation positroid (baseline)

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

### Trained network positroid

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

Available datasets: `moons`, `circles`, `spirals`, `xor`, `digits_0v1_pca2`, `digits_0v1_pca3`, `digits_0v1_pca5`, `digits_0v1_pca10`, `digits_3v8_pca2`, `digits_3v8_pca3`, `digits_3v8_pca5`, `digits_3v8_pca10`.

### Counterexample search

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

## Running tests

```bash
just test            # 176 tests
just coverage        # with coverage report
just check           # lint + typecheck + test
```

## Project structure

```
src/positroid/
├── linalg/          # Minors, totally positive matrix construction & verification
├── matroid/         # Matroid, linear matroid (SVD-based), positroid (Grassmann necklace)
├── arrangement/     # Hyperplane arrangements and their affine matroids
├── network/         # ReLU networks, TP-constrained networks, training loop
├── datasets/        # 2D toy datasets (moons, circles, spirals, XOR) and PCA-reduced digits
└── experiments/     # The three main experiments described above

scripts/
└── generate_blog_figures.py   # Generates all figures used in the blog posts
```

## License

MIT
