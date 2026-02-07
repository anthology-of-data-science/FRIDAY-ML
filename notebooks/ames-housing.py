import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Ames Housing Price Prediction with scikit-learn

    This notebook demonstrates classical machine learning using scikit-learn to predict house prices
    in Ames, Iowa. We'll build explicit pipelines, compare multiple models, and evaluate results.

    **Learning Goals:**
    - Data exploration and preprocessing
    - Building scikit-learn pipelines
    - Model comparison and selection
    - Cross-validation for robust evaluation
    """)
    return


@app.cell
def _():
    import polars as pl
    import altair as alt
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    return (pl,)


@app.cell
def _(mo):
    mo.md("""
    ## 1. Load and Explore Data
    """)
    return


@app.cell
def _(pl):
    # Load the Ames housing dataset
    df = pl.read_csv("../data/ames-housing.csv")

    print(f"Dataset shape: {df.shape}")
    print(f"Number of features: {len(df.columns) - 1}")
    df.head()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Data Preprocessing
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Build Preprocessing Pipeline

    We'll create a pipeline that:
    1. Imputes missing values with median
    2. Standardizes features (important for linear models)
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Train and Compare Models

    We'll compare three models:
    - **Linear Regression**: Simple baseline
    - **Ridge Regression**: L2 regularization to prevent overfitting
    - **Random Forest**: Non-linear ensemble model
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Cross-Validation Evaluation
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Test Set Evaluation
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 7. Predictions vs Actual

    Visualize how well our best model predicts house prices.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 8. Feature Importance (Random Forest)

    Which features are most important for predicting house prices?
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Next Steps

    **To improve the model, try:**

    1. **Include categorical features** using `OneHotEncoder` or `TargetEncoder`
    2. **Feature engineering**: Create new features like:
       - Total square footage (living + basement)
       - Age of house (current year - year built)
       - Price per square foot
    3. **Hyperparameter tuning** using `GridSearchCV`:
       - Number of trees in Random Forest
       - Maximum depth
       - Minimum samples per leaf
    4. **Try other models**:
       - XGBoost
       - LightGBM
       - Stacking ensemble
    5. **Handle outliers** in target variable (very expensive/cheap houses)

    **Compare with AutoGluon:**
    - Run `notebooks/ames-housing-autogluon.py` to see automated model selection
    - Compare which approach gives better insights for learning
    """)
    return


if __name__ == "__main__":
    app.run()
