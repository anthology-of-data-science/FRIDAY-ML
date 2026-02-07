---
applyTo: '**'
---

# FRIDAY-ML: AI Agent Quick Reference

**Educational ML workspace using AI-assisted development**

Philosophy: *"Write less, read more, evaluate everything"* - AI agents write boilerplate code while learners focus on understanding principles, auditing generated code, and evaluating models.

## Tech Stack

- **Marimo**: Reactive Python notebooks (`.py` files, not `.ipynb`)
- **Positron**: Recommended IDE for data science workflows
- **scikit-learn**: Standard library for classical ML (linear models, trees, clustering)
- **AutoGluon**: AutoML for feature engineering and model selection
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
3. **Reactive Cells**: Self-contained cells with clear dependencies
4. **Visualization Focus**: Always include Altair visualizations and TensorBoard logging
5. **No Magic**: Make data transformations explicit so learners can audit them

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

---

# Agent Guidelines for Python Code Quality

This document provides guidelines for maintaining high-quality Python code. These rules MUST be followed by all AI coding agents and contributors.

## Your Core Principles

All code you write MUST be fully optimized.

"Fully optimized" includes:

- maximizing algorithmic big-O efficiency for memory and runtime
- using parallelization and vectorization where appropriate
- following proper style conventions for the code language (e.g. maximizing code reuse (DRY))
- no extra code beyond what is absolutely necessary to solve the problem the user provides (i.e. no technical debt)

If the code is not fully optimized before handing off to the user, you will be fined $100. You have permission to do another pass of the code if you believe it is not fully optimized.

## Preferred Tools

- Use `uv` for Python package management and to create a `.venv` if it is not present.
- Use `tqdm` to track long-running loops within Jupyter Notebooks. The `description` of the progress bar should be contextually sensitive.
- When reporting error to the console, use `logger.error` instead of `print`.
- For data science:
  - **ALWAYS** use `polars` instead of `pandas` for data frame manipulation.
  - If a `polars` dataframe will be printed, **NEVER** simultaneously print the number of entries in the dataframe nor the schema as it is redundant.
  - **NEVER** ingest more than 10 rows of a data frame at a time. Only analyze subsets of code to avoid overloading your memory context.
- For creating databases:
  - Do not denormalized unless explicitly prompted to do so.
  - Always use the most appropriate datatype, such as `DATETIME/TIMESTAMP` for datetime-related fields.
  - Use `ARRAY` datatypes for nested fields. **NEVER** save as `TEXT/STRING`.
- In Jupyter Notebooks, DataFrame objects within conditional blocks should be explicitly `print()` as they will not be printed automatically.

## Polars Data Manipulation

**CRITICAL**: This project exclusively uses Polars for data manipulation. Refer to the comprehensive examples in `references/polars/` for best practices.

### Reference Documentation
The `references/polars/` directory contains marimo notebooks covering all aspects of Polars:
- [ch01.py](references/polars/ch01.py) - Introduction and basic operations
- [ch02.py](references/polars/ch02.py) - Data types and schemas
- [ch03.py](references/polars/ch03.py) - Expressions and selectors
- [ch04.py](references/polars/ch04.py) - Data transformation
- [ch05.py](references/polars/ch05.py) - Missing data handling
- [ch06.py](references/polars/ch06.py) - Aggregations
- [ch07.py](references/polars/ch07.py) - Joins and concatenations
- [ch08.py](references/polars/ch08.py) - Visualization with plotnine
- [ch09-ch18.py](references/polars/) - Advanced topics (lazy evaluation, time series, I/O, etc.)

### Core Polars Principles

1. **Use Method Chaining**: Always chain operations for readability and efficiency
   ```python
   # Good - method chaining with Polars expressions
   result = (
       df
       .filter(pl.col("age") > 18)
       .select([
           pl.col("name"),
           pl.col("salary").alias("annual_salary"),
           (pl.col("salary") / 12).alias("monthly_salary")
       ])
       .sort("annual_salary", descending=True)
   )

   # Bad - separate statements
   df_filtered = df.filter(pl.col("age") > 18)
   df_selected = df_filtered.select(["name", "salary"])
   result = df_selected.sort("salary", descending=True)
   ```

2. **Use Expressions (`pl.col()`)**: Always use Polars expressions for column operations
   ```python
   # Good - using expressions
   df.select([
       pl.col("temperature").mean().alias("avg_temp"),
       pl.col("humidity").max().alias("max_humidity")
   ])

   # Bad - string column names without expressions
   df.select(["temperature", "humidity"])
   ```

3. **Use Selectors for Multiple Columns**: Use `pl.col()` with data types or patterns
   ```python
   # Good - select all numeric columns
   df.select(pl.col(pl.Float64, pl.Int64))

   # Good - select columns by pattern
   df.select(pl.col("^date.*$"))

   # Good - select all except specific columns
   df.select(pl.all().exclude("id"))
   ```

4. **Lazy Evaluation for Large Datasets**: Use `.lazy()` for query optimization
   ```python
   # Good - lazy evaluation
   result = (
       pl.scan_csv("large_file.csv")
       .filter(pl.col("status") == "active")
       .group_by("category")
       .agg(pl.col("value").sum())
       .collect()
   )

   # Less optimal - eager evaluation
   df = pl.read_csv("large_file.csv")
   result = df.filter(pl.col("status") == "active").group_by("category").agg(pl.col("value").sum())
   ```

5. **Use `.alias()` for Clear Column Names**: Always name computed columns
   ```python
   # Good - clear aliases
   df.select([
       (pl.col("total") / pl.col("count")).alias("average"),
       pl.col("price").mul(1.2).alias("price_with_tax")
   ])

   # Bad - default column names
   df.select([
       pl.col("total") / pl.col("count"),
       pl.col("price") * 1.2
   ])
   ```

### Common Patterns

**Data Loading** (see [ch01.py](references/polars/ch01.py)):
```python
import polars as pl

# CSV with schema inference
df = pl.read_csv("data.csv")

# CSV with explicit schema
df = pl.read_csv("data.csv", schema={"id": pl.Int64, "name": pl.String, "date": pl.Date})

# Lazy reading for large files
df = pl.scan_csv("large_data.csv").collect()
```

**Filtering and Selecting** (see [ch04.py](references/polars/ch04.py)):
```python
# Multiple conditions with expressions
filtered = df.filter(
    (pl.col("age") >= 18) &
    (pl.col("country") == "USA") |
    (pl.col("status") == "premium")
)

# Select and transform
result = df.select([
    pl.col("name"),
    pl.col("age").cast(pl.Float64),
    (pl.col("height") / 100).alias("height_m")
])
```

**Aggregations** (see [ch06.py](references/polars/ch06.py)):
```python
# Group by with multiple aggregations
summary = df.group_by("category").agg([
    pl.col("sales").sum().alias("total_sales"),
    pl.col("sales").mean().alias("avg_sales"),
    pl.col("customer_id").n_unique().alias("unique_customers"),
    pl.len().alias("transaction_count")
])
```

**Missing Data** (see [ch05.py](references/polars/ch05.py)):
```python
# Fill missing values
df_filled = df.with_columns([
    pl.col("price").fill_null(strategy="forward"),
    pl.col("category").fill_null("Unknown")
])

# Drop rows with missing values
df_clean = df.drop_nulls(subset=["critical_column"])
```

**Time Series** (see [ch14.py](references/polars/ch14.py)):
```python
# Parse dates and perform time-based operations
df_time = (
    df
    .with_columns(pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d"))
    .sort("date")
    .group_by_dynamic("date", every="1mo")
    .agg(pl.col("value").sum())
)
```

### Anti-Patterns to Avoid

1. **NEVER use pandas unless absolutely required for interop**
   ```python
   # Bad - converting to pandas unnecessarily
   df.to_pandas().groupby("category").sum()

   # Good - use Polars native operations
   df.group_by("category").agg(pl.all().sum())
   ```

2. **NEVER iterate over rows** - use vectorized operations
   ```python
   # Bad - row iteration
   for row in df.iter_rows():
       result.append(row[0] * 2)

   # Good - vectorized operation
   result = df.select((pl.col("value") * 2).alias("doubled"))
   ```

3. **NEVER use string column names in transformations** - use `pl.col()`
   ```python
   # Bad - string column references
   df.select(["col1", "col2"])

   # Good - expression-based selection
   df.select([pl.col("col1"), pl.col("col2")])
   ```

4. **NEVER chain `.to_list()` unnecessarily** - work with expressions
   ```python
   # Bad - converting to list
   values = df["column"].to_list()
   result = [v * 2 for v in values]

   # Good - use expressions
   result = df.select((pl.col("column") * 2).alias("doubled"))
   ```

### When You Need Help
- Consult the example notebooks in `references/polars/` for comprehensive patterns
- Check [ch03.py](references/polars/ch03.py) for expression syntax
- See [appendix1.py](references/polars/appendix1.py) for advanced techniques

## Code Style and Formatting

- **MUST** use meaningful, descriptive variable and function names
- **MUST** follow PEP 8 style guidelines
- **MUST** use 4 spaces for indentation (never tabs)
- **NEVER** use emoji, or unicode that emulates emoji (e.g. ✓, ✗). The only exception is when writing tests and testing the impact of multibyte characters.
- Use snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- Limit line length to 120 characters (ruff formatter standard)
- Do not add comments to the code you write, unless the user asks you to, or the code is complex and requires additional context.

## Documentation

- **MUST** include docstrings for all public functions, classes, and methods
- **MUST** document function parameters, return values, and exceptions raised
- Keep comments up-to-date with code changes
- Include examples in docstrings for complex functions

Example docstring:

```python
def calculate_total(items: list[dict], tax_rate: float = 0.0) -> float:
    """Calculate the total cost of items including tax.

    Args:
        items: List of item dictionaries with 'price' keys
        tax_rate: Tax rate as decimal (e.g., 0.08 for 8%)

    Returns:
        Total cost including tax

    Raises:
        ValueError: If items is empty or tax_rate is negative
    """
```

## Type Hints

- **MUST** use type hints for all function signatures (parameters and return values)
- **NEVER** use `Any` type unless absolutely necessary
- **MUST** run mypy and resolve all type errors
- Use `Optional[T]` or `T | None` for nullable types

## Error Handling

- **NEVER** silently swallow exceptions without logging
- **MUST** never use bare `except:` clauses
- **MUST** catch specific exceptions rather than broad exception types
- **MUST** use context managers (`with` statements) for resource cleanup
- Provide meaningful error messages

## Function Design

- **MUST** keep functions focused on a single responsibility
- **NEVER** use mutable objects (lists, dicts) as default argument values
- Limit function parameters to 5 or fewer
- Return early to reduce nesting

## Class Design

- **MUST** keep classes focused on a single responsibility
- **MUST** keep `__init__` simple; avoid complex logic
- Use dataclasses for simple data containers
- Prefer composition over inheritance
- Avoid creating additional class functions if they are not necessary
- Use `@property` for computed attributes

## Testing

- **MUST** write unit tests for all new functions and classes
- **MUST** mock external dependencies (APIs, databases, file systems)
- **MUST** use pytest as the testing framework
- **NEVER** run tests you generate without first saving them as their own discrete file
- **NEVER** delete files created as a part of testing.
- Ensure the folder used for test outputs is present in `.gitignore`
- Follow the Arrange-Act-Assert pattern
- Do not commit commented-out tests

## Imports and Dependencies

- **MUST** avoid wildcard imports (`from module import *`)
- **MUST** document dependencies in `pyproject.toml`
- Use `uv` for fast package management and dependency resolution
- Organize imports: standard library, third-party, local imports
- Use `isort` to automate import formatting

## Python Best Practices

- **NEVER** use mutable default arguments
- **MUST** use context managers (`with` statement) for file/resource management
- **MUST** use `is` for comparing with `None`, `True`, `False`
- **MUST** use f-strings for string formatting
- Use list comprehensions and generator expressions
- Use `enumerate()` instead of manual counter variables



## Version Control

- **MUST** write clear, descriptive commit messages
- **NEVER** commit commented-out code; delete it
- **NEVER** commit debug print statements or breakpoints
- **NEVER** commit credentials or sensitive data

## Tools

- **MUST** use Ruff for code formatting and linting (replaces Black, isort, flake8)
- **MUST** use mypy for static type checking
- Use `uv` for package management (faster alternative to pip)
- Use pytest for testing

## Before Committing

- [ ] All tests pass
- [ ] Type checking passes (mypy)
- [ ] Code formatter and linter pass (Ruff)
- [ ] All functions have docstrings and type hints
- [ ] No commented-out code or debug statements
- [ ] No hardcoded credentials

---

# Marimo notebook assistant

I am a specialized AI assistant designed to help create data science notebooks using marimo. I focus on creating clear, efficient, and reproducible data analysis workflows with marimo's reactive programming model.

If you make edits to the notebook, only edit the contents inside the function decorator with @app.cell.
marimo will automatically handle adding the parameters and return statement of the function. For example,
for each edit, just return:

```
@app.cell
def _():
    <your code here>
    return
```

## Marimo fundamentals

Marimo is a reactive notebook that differs from traditional notebooks in key ways:

- Cells execute automatically when their dependencies change
- Variables cannot be redeclared across cells
- The notebook forms a directed acyclic graph (DAG)
- The last expression in a cell is automatically displayed
- UI elements are reactive and update the notebook automatically

## Code Requirements

1. All code must be complete and runnable
2. Follow consistent coding style throughout
3. Include descriptive variable names and helpful comments
4. Import all modules in the first cell, always including `import marimo as mo`
5. Never redeclare variables across cells
6. Ensure no cycles in notebook dependency graph
7. The last expression in a cell is automatically displayed, just like in Jupyter notebooks.
8. Don't include comments in markdown cells
9. Don't include comments in SQL cells
10. Never define anything using `global`.

## Reactivity

Marimo's reactivity means:

- When a variable changes, all cells that use that variable automatically re-execute
- UI elements trigger updates when their values change without explicit callbacks
- UI element values are accessed through `.value` attribute
- You cannot access a UI element's value in the same cell where it's defined
- Cells prefixed with an underscore (e.g. _my_var) are local to the cell and cannot be accessed by other cells

## Best Practices

### Data handling

- Use polars for data manipulation, specifically demonstrating Expressions and method chaining
- Implement proper data validation
- Handle missing values appropriately
- Use efficient data structures
- A variable in the last expression of a cell is automatically displayed as a table

### Visualization

- Use altair for visualization as much as possible
- Use seaborn for high-level statistical plots
- For altair: return the chart object directly. Add tooltips where appropriate. You can pass polars dataframes directly to altair.
- Include proper labels, titles, and color schemes
- Make visualizations interactive where appropriate


### UI elements

- Access UI element values with .value attribute (e.g., slider.value)
- Create UI elements in one cell and reference them in later cells
- Create intuitive layouts with mo.hstack(), mo.vstack(), and mo.tabs()
- Prefer reactive updates over callbacks (marimo handles reactivity automatically)
- Group related UI elements for better organization

### SQL
- When writing duckdb, prefer using marimo's SQL cells, which start with df = mo.sql(f"""<your query>""") for DuckDB, or df = mo.sql(f"""<your query>""", engine=engine) for other SQL engines.
- See the SQL with duckdb example for an example on how to do this
- Don't add comments in cells that use mo.sql()


## Troubleshooting

Common issues and solutions:

- Circular dependencies: Reorganize code to remove cycles in the dependency graph
- UI element value access: Move access to a separate cell from definition
- Visualization not showing: Ensure the visualization object is the last expression

After generating a notebook, run `marimo check --fix` to catch and
automatically resolve common formatting issues, and detect common pitfalls.

## Available UI elements

- `mo.ui.altair_chart(altair_chart)`
- `mo.ui.button(value=None, kind='primary')`
- `mo.ui.run_button(label=None, tooltip=None, kind='primary')`
- `mo.ui.checkbox(label='', value=False)`
- `mo.ui.date(value=None, label=None, full_width=False)`
- `mo.ui.dropdown(options, value=None, label=None, full_width=False)`
- `mo.ui.file(label='', multiple=False, full_width=False)`
- `mo.ui.number(value=None, label=None, full_width=False)`
- `mo.ui.radio(options, value=None, label=None, full_width=False)`
- `mo.ui.refresh(options: List[str], default_interval: str)`
- `mo.ui.slider(start, stop, value=None, label=None, full_width=False, step=None)`
- `mo.ui.range_slider(start, stop, value=None, label=None, full_width=False, step=None)`
- `mo.ui.table(data, columns=None, on_select=None, sortable=True, filterable=True)`
- `mo.ui.text(value='', label=None, full_width=False)`
- `mo.ui.text_area(value='', label=None, full_width=False)`
- `mo.ui.data_explorer(df)`
- `mo.ui.dataframe(df)`
- `mo.ui.plotly(plotly_figure)`
- `mo.ui.tabs(elements: dict[str, mo.ui.Element])`
- `mo.ui.array(elements: list[mo.ui.Element])`
- `mo.ui.form(element: mo.ui.Element, label='', bordered=True)`

## Layout and utility functions

- `mo.md(text)` - display markdown
- `mo.stop(predicate, output=None)` - stop execution conditionally
- `mo.output.append(value)` - append to the output when it is not the last expression
- `mo.output.replace(value)` - replace the output when it is not the last expression
- `mo.Html(html)` - display HTML
- `mo.image(image)` - display an image
- `mo.hstack(elements)` - stack elements horizontally
- `mo.vstack(elements)` - stack elements vertically
- `mo.tabs(elements)` - create a tabbed interface

## Examples

<example title="Markdown ccell">
```
@app.cell
def _():
    mo.md("""
    # Hello world
    This is a _markdown_ **cell**.
    """)
    return
```
</example>

<example title="Basic UI with reactivity">
```
@app.cell
def _():
    import marimo as mo
    import altair as alt
    import polars as pl
    import numpy as np
    return

@app.cell
def _():
    n_points = mo.ui.slider(10, 100, value=50, label="Number of points")
    n_points
    return

@app.cell
def _():
    x = np.random.rand(n_points.value)
    y = np.random.rand(n_points.value)

    df = pl.DataFrame({"x": x, "y": y})

    chart = alt.Chart(df).mark_circle(opacity=0.7).encode(
        x=alt.X('x', title='X axis'),
        y=alt.Y('y', title='Y axis')
    ).properties(
        title=f"Scatter plot with {n_points.value} points",
        width=400,
        height=300
    )

    chart
    return

```
</example>

<example title="Data explorer">
```

@app.cell
def _():
    import marimo as mo
    import polars as pl
    from vega_datasets import data
    return

@app.cell
def _():
    cars_df = pl.DataFrame(data.cars())
    mo.ui.data_explorer(cars_df)
    return

```
</example>

<example title="Multiple UI elements">
```

@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    return

@app.cell
def _():
    iris = pl.read_csv("hf://datasets/scikit-learn/iris/Iris.csv")
    return

@app.cell
def _():
    species_selector = mo.ui.dropdown(
        options=["All"] + iris["Species"].unique().to_list(),
        value="All",
        label="Species",
    )
    x_feature = mo.ui.dropdown(
        options=iris.select(pl.col(pl.Float64, pl.Int64)).columns,
        value="SepalLengthCm",
        label="X Feature",
    )
    y_feature = mo.ui.dropdown(
        options=iris.select(pl.col(pl.Float64, pl.Int64)).columns,
        value="SepalWidthCm",
        label="Y Feature",
    )
    mo.hstack([species_selector, x_feature, y_feature])
    return

@app.cell
def _():
    filtered_data = iris if species_selector.value == "All" else iris.filter(pl.col("Species") == species_selector.value)

    chart = alt.Chart(filtered_data).mark_circle().encode(
        x=alt.X(x_feature.value, title=x_feature.value),
        y=alt.Y(y_feature.value, title=y_feature.value),
        color='Species'
    ).properties(
        title=f"{y_feature.value} vs {x_feature.value}",
        width=500,
        height=400
    )

    chart
    return

```
</example>

<example title="Conditional Outputs">
```

@app.cell
def _():
    mo.stop(not data.value, mo.md("No data to display"))

    if mode.value == "scatter":
        mo.output.replace(render_scatter(data.value))
    else:
        mo.output.replace(render_bar_chart(data.value))
    return

```
</example>

<example title="Interactive chart with Altair">
```

@app.cell
def _():
    import marimo as mo
    import altair as alt
    import polars as pl
    return

@app.cell
def _():
    # Load dataset
    weather = pl.read_csv("<https://raw.githubusercontent.com/vega/vega-datasets/refs/heads/main/data/weather.csv>")
    weather_dates = weather.with_columns(
        pl.col("date").str.strptime(pl.Date, format="%Y-%m-%d")
    )
    _chart = (
        alt.Chart(weather_dates)
        .mark_point()
        .encode(
            x="date:T",
            y="temp_max",
            color="location",
        )
    )
    return

@app.cell
def _():
    chart = mo.ui.altair_chart(_chart)
chart
    return

@app.cell
def _():
    # Display the selection
    chart.value
    return

```
</example>

<example title="Run Button Example">
```

@app.cell
def _():
    import marimo as mo
    return

@app.cell
def _():
    first_button = mo.ui.run_button(label="Option 1")
    second_button = mo.ui.run_button(label="Option 2")
    [first_button, second_button]
    return

@app.cell
def _():
    if first_button.value:
        print("You chose option 1!")
    elif second_button.value:
        print("You chose option 2!")
    else:
        print("Click a button!")
    return

```
</example>

<example title="SQL with duckdb">
```

@app.cell
def _():
    import marimo as mo
    import polars as pl
    return

@app.cell
def _():
    weather = pl.read_csv('<https://raw.githubusercontent.com/vega/vega-datasets/refs/heads/main/data/weather.csv>')
    return

@app.cell
def _():
    seattle_weather_df = mo.sql(
        f"""
        SELECT * FROM weather WHERE location = 'Seattle';
        """
    )
    return

```
</example>

---
# Security

- **NEVER** store secrets, API keys, or passwords in code. Only store them in `.env`.
  - Ensure `.env` is declared in `.gitignore`.
  - **NEVER** print or log URLs to console if they contain an API key.
- **MUST** use environment variables for sensitive configuration
- **NEVER** log sensitive information (passwords, tokens, PII)