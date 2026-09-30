import matplotlib.pyplot as plt
import shap
from PyALE import ale
from sklearn.inspection import PartialDependenceDisplay


class ProbabilityModel:
    """Wrapper for models that return class probabilities."""

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]


def create_shap_plot(model, X, feature_names, output_path):
    """Create and save a SHAP summary plot."""

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)

    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_ale_plot(model, X, feature, output_path):
    """Create and save an ALE plot."""

    prob_model = ProbabilityModel(model)

    ale(X=X, model=prob_model, feature=[feature], grid_size=20, include_CI=True)

    plt.title(f"ALE Plot - {feature}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_pdp_plot(model, X, feature, output_path):
    """Create and save a Partial Dependence Plot."""

    fig, ax = plt.subplots(figsize=(7, 5))

    PartialDependenceDisplay.from_estimator(
        model, X, features=[feature], kind="average", ax=ax
    )

    ax.set_title(f"Partial Dependence Plot - {feature}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
