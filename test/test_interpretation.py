import os
import numpy as np
import pandas as pd
import xgboost as xgb
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.interpretation import (
    create_shap_plot,
    create_ale_plot,
    create_pdp_plot
)


def create_test_data():
    """Create a small dataset for testing the interpretation functions."""

    X = pd.DataFrame({
        "numeric__Income": [20000, 30000, 40000, 50000, 60000, 70000],
        "numeric__Age": [20, 25, 30, 35, 40, 45],
        "ordinal__Education_Level": [1, 1, 2, 2, 3, 3]
    })

    y = np.array([0, 0, 0, 1, 1, 1])

    return X, y


def create_test_model(X, y):
    """Create and train a small XGBoost model for testing."""

    model = xgb.XGBClassifier(
        n_estimators=20,
        max_depth=2,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X, y)

    return model


def test_shap_plot(tmp_path):

    X, y = create_test_data()
    model = create_test_model(X, y)

    output_path = tmp_path / "shap_summary.png"

    create_shap_plot(
        model=model,
        X=X,
        feature_names=X.columns.tolist(),
        output_path=output_path
    )

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0


def test_ale_plot(tmp_path):

    X, y = create_test_data()
    model = create_test_model(X, y)

    output_path = tmp_path / "ale_income.png"

    create_ale_plot(
        model=model,
        X=X,
        feature="numeric__Income",
        output_path=output_path
    )

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0


def test_pdp_plot(tmp_path):

    X, y = create_test_data()
    model = create_test_model(X, y)

    output_path = tmp_path / "pdp_income.png"

    create_pdp_plot(
        model=model,
        X=X,
        feature="numeric__Income",
        output_path=output_path
    )

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0