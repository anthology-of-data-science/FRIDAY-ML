# FRIDAY-ML: AI Agent Quick Reference

**Educational ML workspace using AI-assisted development**

Philosophy: *"Write less, read more, evaluate everything"* - AI agents write boilerplate code while learners focus on understanding principles, auditing generated code, and evaluating models.

## Tech Stack

- **Marimo**: Reactive Python notebooks (`.py` files, not `.ipynb`)
- **Positron**: Recommended IDE for data science workflows
- **scikit-learn**: Standard library for classical ML (linear models, trees, clustering)
- **TensorFlow/Keras**: Deep learning with TFDS integration
- **Polars**: Data manipulation library
- **Altair**: Declarative visualization library
- **uv**: Fast Python package installer

## Quick Commands

```bash
# Setup
uv sync                           # Install dependencies
uv sync --extra tf-apple          # + TensorFlow for Apple Silicon
uv sync --extra tf                # + TensorFlow (standard)

# Run notebooks
uv run marimo edit notebooks/mnist.py        # MNIST deep learning
uv run marimo edit notebooks/ames-housing.py # AutoGluon example
just mnist                                    # Shortcut for MNIST
just ames-housing                             # Shortcut for Ames Housing
just tensorboard                              # View training logs
```

## Project Structure

```
FRIDAY-ML/
├── notebooks/          # Marimo notebooks (mnist.py, ames-housing.py)
├── data/              # Datasets (*.csv files)
├── logs/              # TensorBoard training logs
├── references/        # Reference docs and examples
│   ├── altair/        # 8 comprehensive Altair notebooks
│   └── polars/        # 19 comprehensive Polars notebooks
└── .claude/
    ├── CLAUDE.md      # This file - quick reference
    └── rules/         # Detailed guidelines
        ├── GUIDELINES.md    # Comprehensive AI agent guidance ⭐
        ├── code-style.md    # Python code standards
        ├── marimo.md        # Marimo patterns and best practices
        └── security.md      # Security guidelines
```

## Core Principles (See [GUIDELINES.md](.claude/rules/GUIDELINES.md) for details)

1. **Educational Context**: Code is for ML learners - prioritize clarity over cleverness
2. **Use Marimo Format**: Reactive `.py` notebooks, not `.ipynb`
3. **Minimize Boilerplate**: Leverage AutoGluon to reduce preprocessing code
4. **Reactive Cells**: Self-contained cells with clear dependencies
5. **Visualization Focus**: Always include Altair visualizations and TensorBoard logging
6. **No Magic**: Make data transformations explicit so learners can audit them

## Quick Patterns

**Marimo Notebook**:
```python
import marimo as mo
app = marimo.App()

@app.cell
def load_data():
    df = pl.read_csv("data/dataset.csv")
    return df,

@app.cell
def analyze_data(df):
    result = df.group_by("category").agg(pl.col("value").mean())
    return result,
```

**scikit-learn**:
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression().fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))
```

**AutoGluon**:
```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label="target").fit(train_data)
leaderboard = predictor.leaderboard()
```

**TensorFlow/Keras**:
```python
import tensorflow_datasets as tfds

(train, test), info = tfds.load('mnist', split=['train', 'test'], with_info=True, as_supervised=True)
train = train.map(normalize).cache().shuffle(1000).batch(32).prefetch(1)

# Use TensorBoard callbacks
log_dir = f"logs/fit/{datetime.now():%Y%m%d-%H%M%S}"
model.fit(train, callbacks=[tf.keras.callbacks.TensorBoard(log_dir=log_dir)])
```

## Essential Reading

- **[GUIDELINES.md](.claude/rules/GUIDELINES.md)** ⭐ - Comprehensive AI agent guidance
- **[code-style.md](.claude/rules/code-style.md)** - Python standards and Polars patterns
- **[marimo.md](.claude/rules/marimo.md)** - Marimo notebook best practices
- **[references/polars/](references/polars/)** - 19 comprehensive Polars examples
- **[references/altair/](references/altair/)** - 8 comprehensive Polars examples

## Python Version

Requires Python 3.12+ (specified in pyproject.toml)
