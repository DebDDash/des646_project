"""
Implements SHAP-based explainability for model predictions.

Supports:
- Tabular models (e.g., sklearn, XGBoost, LightGBM)
- Deep models (PyTorch / TensorFlow)
- Text and image models (via kernel or deep explainers)

Provides:
    - SHAP value computation
    - Summary and force plots
    - Batch explanations
    - Export utilities for downstream visualization
"""

import shap
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union, Optional


class ShapExplainer:
    """
    Wrapper for generating SHAP explanations for different model types.
    """

    def __init__(self, model, data_sample, model_type: str = "tabular"):
        """
        Args:
            model: Trained model object (sklearn, torch, tensorflow, etc.)
            data_sample: Representative subset of the training data.
            model_type: One of ['tabular', 'image', 'text'].
        """
        self.model = model
        self.data_sample = data_sample
        self.model_type = model_type.lower()

        if self.model_type == "tabular":
            self.explainer = shap.Explainer(model, data_sample)
        elif self.model_type == "image":
            self.explainer = shap.GradientExplainer(model, data_sample)
        elif self.model_type == "text":
            self.explainer = shap.Explainer(model)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def explain_instance(self, instance):
        """Return SHAP values for a single instance."""
        shap_values = self.explainer(instance)
        return shap_values

    def explain_batch(self, X_batch):
        """Return SHAP values for a batch of samples."""
        shap_values = self.explainer(X_batch)
        return shap_values

    def summary_plot(self, shap_values, feature_names=None, max_display=20, save_path: Optional[Union[str, Path]] = None):
        """Generate and optionally save SHAP summary plot."""
        shap.summary_plot(
            shap_values.values,
            self.data_sample,
            feature_names=feature_names,
            show=False,
            max_display=max_display
        )
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def force_plot(self, shap_values, instance_index=0, save_path: Optional[Union[str, Path]] = None):
        """Generate SHAP force plot for an individual prediction."""
        force = shap.plots.force(shap_values[instance_index])
        if save_path:
            shap.save_html(save_path, force)
        return force

    def export_values(self, shap_values, path: Union[str, Path]):
        """Export SHAP values as a numpy file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, shap_values.values)
        print(f"[INFO] SHAP values saved to {path}")


if __name__ == "__main__":
    # Example usage for tabular data
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier

    data = load_iris()
    X, y = data.data, data.target
    model = RandomForestClassifier().fit(X, y)

    explainer = ShapExplainer(model, X, model_type="tabular")
    shap_values = explainer.explain_batch(X[:50])

    explainer.summary_plot(
        shap_values,
        feature_names=data.feature_names,
        save_path="outputs/shap_summary.png"
    )
    explainer.force_plot(shap_values, instance_index=0, save_path="outputs/force_plot.html")
