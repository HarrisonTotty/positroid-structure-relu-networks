default:
    @just --list

# Install dependencies
install:
    uv sync --all-extras

# Run all tests
test *ARGS:
    uv run pytest {{ ARGS }}

# Run tests with coverage
coverage:
    uv run pytest --cov=positroid --cov-report=term-missing

# Type check
typecheck:
    uv run mypy src/

# Lint
lint:
    uv run ruff check src/ tests/

# Format
fmt:
    uv run ruff format src/ tests/

# Format check (no changes)
fmt-check:
    uv run ruff format --check src/ tests/

# Run all checks
check: lint typecheck test

# Run the activation positroid experiment
experiment-positroid *ARGS:
    uv run python -m positroid.experiments.activation_positroid {{ ARGS }}

# Run the trained network positroid experiment
experiment-trained-positroid *ARGS:
    uv run python -m positroid.experiments.trained_positroid {{ ARGS }}

# Run trained positroid experiment on digit datasets
experiment-trained-positroid-digits *ARGS:
    uv run python -m positroid.experiments.trained_positroid \
        --datasets digits_0v1_pca5 digits_0v1_pca10 digits_3v8_pca5 digits_3v8_pca10 \
        --hidden-dims 8 12 16 \
        --num-trials 10 \
        {{ ARGS }}

# Run counterexample search for the activation positroid conjecture
experiment-counterexample *ARGS:
    uv run python -m positroid.experiments.counterexample_search {{ ARGS }}

# Build paper 1 (positroid structure)
paper1:
    cd papers/1-positroid-structure && pdflatex -interaction=nonstopmode -jobname=positroid-structure-relu-networks main.tex && pdflatex -interaction=nonstopmode -jobname=positroid-structure-relu-networks main.tex

# Build all papers
papers: paper1

# Clean paper build artifacts
papers-clean:
    cd papers/1-positroid-structure && rm -f positroid-structure-relu-networks.{aux,log,out,bbl,blg}
