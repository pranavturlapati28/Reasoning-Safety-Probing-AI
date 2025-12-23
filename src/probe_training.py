"""
Linear probe training for detecting hint usage.

Implements:
- Simple linear probes
- Multi-layer probe training
- Evaluation metrics (AUROC, accuracy, calibration)
- Permutation tests for statistical significance
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class LinearProbe:
    """Linear probe for binary classification."""

    def __init__(
        self,
        input_dim: int,
        regularization: float = 1.0,
        max_iter: int = 1000
    ):
        """
        Initialize linear probe.

        Args:
            input_dim: Dimension of input features
            regularization: L2 regularization strength (C parameter)
            max_iter: Maximum iterations for optimization
        """
        self.input_dim = input_dim
        self.regularization = regularization

        self.model = LogisticRegression(
            C=regularization,
            max_iter=max_iter,
            random_state=42
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2
    ) -> Dict:
        """
        Train the probe.

        Args:
            X: Features (n_samples, input_dim)
            y: Labels (n_samples,)
            validation_split: Fraction for validation

        Returns:
            Dictionary with training metrics
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=validation_split,
            random_state=42,
            stratify=y
        )

        # Train
        self.model.fit(X_train, y_train)

        # Evaluate
        train_acc = self.model.score(X_train, y_train)
        val_acc = self.model.score(X_val, y_val)

        # Predictions
        y_val_pred = self.model.predict(X_val)
        y_val_prob = self.model.predict_proba(X_val)[:, 1]

        # Metrics
        val_auroc = roc_auc_score(y_val, y_val_prob)

        return {
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "val_auroc": val_auroc,
            "confusion_matrix": confusion_matrix(y_val, y_val_pred),
            "n_train": len(X_train),
            "n_val": len(X_val)
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate probe on test set.

        Args:
            X: Features
            y: True labels

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)

        accuracy = accuracy_score(y, y_pred)
        auroc = roc_auc_score(y, y_prob)
        cm = confusion_matrix(y, y_pred)

        return {
            "accuracy": accuracy,
            "auroc": auroc,
            "confusion_matrix": cm,
            "predictions": y_pred,
            "probabilities": y_prob
        }


class MultiLayerProbeAnalysis:
    """Train and analyze probes across multiple layers."""

    def __init__(self):
        """Initialize multi-layer probe analysis."""
        self.probes = {}
        self.results = {}

    def train_probes_per_layer(
        self,
        hidden_states_dict: Dict[int, np.ndarray],
        labels: np.ndarray,
        regularization: float = 1.0
    ) -> Dict:
        """
        Train a probe for each layer.

        Args:
            hidden_states_dict: Dict mapping layer_idx -> hidden states array
            labels: Binary labels
            regularization: L2 regularization

        Returns:
            Dictionary of results per layer
        """
        results = {}

        for layer_idx, hidden_states in hidden_states_dict.items():
            print(f"Training probe for layer {layer_idx}...")

            # Create and train probe
            probe = LinearProbe(
                input_dim=hidden_states.shape[1],
                regularization=regularization
            )

            train_results = probe.train(hidden_states, labels)

            # Store probe and results
            self.probes[layer_idx] = probe
            results[layer_idx] = train_results

            print(f"  Val AUROC: {train_results['val_auroc']:.4f}")

        self.results = results
        return results

    def evaluate_all_layers(
        self,
        hidden_states_dict: Dict[int, np.ndarray],
        labels: np.ndarray
    ) -> Dict:
        """
        Evaluate all trained probes.

        Args:
            hidden_states_dict: Test hidden states per layer
            labels: Test labels

        Returns:
            Evaluation results per layer
        """
        eval_results = {}

        for layer_idx, probe in self.probes.items():
            if layer_idx in hidden_states_dict:
                hidden_states = hidden_states_dict[layer_idx]
                eval_results[layer_idx] = probe.evaluate(hidden_states, labels)

        return eval_results

    def plot_layer_performance(self, metric: str = "auroc"):
        """
        Plot probe performance across layers.

        Args:
            metric: Metric to plot (auroc, accuracy)
        """
        if not self.results:
            print("No results to plot. Train probes first.")
            return

        layers = sorted(self.results.keys())
        values = [self.results[layer][f"val_{metric}"] for layer in layers]

        plt.figure(figsize=(10, 6))
        plt.plot(layers, values, marker='o', linewidth=2, markersize=8)
        plt.xlabel("Layer", fontsize=12)
        plt.ylabel(f"Validation {metric.upper()}", fontsize=12)
        plt.title(f"Probe {metric.upper()} Across Layers", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return plt.gcf()


def permutation_test(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_permutations: int = 1000,
    metric_fn=roc_auc_score
) -> Dict:
    """
    Permutation test for statistical significance.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_permutations: Number of permutations
        metric_fn: Metric function to use

    Returns:
        Dictionary with test results
    """
    # Observed metric
    observed_metric = metric_fn(y_true, y_prob)

    # Permutation distribution
    null_distribution = []

    for _ in range(n_permutations):
        y_permuted = np.random.permutation(y_true)
        null_metric = metric_fn(y_permuted, y_prob)
        null_distribution.append(null_metric)

    null_distribution = np.array(null_distribution)

    # P-value (one-sided test: observed > null)
    p_value = np.mean(null_distribution >= observed_metric)

    return {
        "observed_metric": observed_metric,
        "null_mean": np.mean(null_distribution),
        "null_std": np.std(null_distribution),
        "p_value": p_value,
        "null_distribution": null_distribution,
        "significant": p_value < 0.05
    }


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, label: str = ""):
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        label: Label for the curve
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"{label} (AUROC = {auroc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label="Random")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt.gcf()


def plot_confusion_matrix(cm: np.ndarray, labels: List[str] = None):
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix
        labels: Class labels
    """
    if labels is None:
        labels = ["No Hint Influence", "Hint Influenced"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14)
    plt.tight_layout()

    return plt.gcf()


if __name__ == "__main__":
    # Test probe training
    print("Testing probe training...")

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    hidden_dim = 512

    # Create synthetic hidden states with some signal
    X_pos = np.random.randn(n_samples // 2, hidden_dim) + 0.5
    X_neg = np.random.randn(n_samples // 2, hidden_dim) - 0.5
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))

    # Shuffle
    indices = np.random.permutation(n_samples)
    X, y = X[indices], y[indices]

    # Train probe
    probe = LinearProbe(input_dim=hidden_dim)
    results = probe.train(X, y)

    print(f"\nTraining Results:")
    print(f"  Train Accuracy: {results['train_accuracy']:.4f}")
    print(f"  Val Accuracy: {results['val_accuracy']:.4f}")
    print(f"  Val AUROC: {results['val_auroc']:.4f}")

    # Run permutation test
    X_test, y_test = X[:100], y[:100]
    y_prob_test = probe.predict_proba(X_test)

    perm_results = permutation_test(y_test, y_prob_test, n_permutations=100)
    print(f"\nPermutation Test:")
    print(f"  Observed AUROC: {perm_results['observed_metric']:.4f}")
    print(f"  Null Mean: {perm_results['null_mean']:.4f}")
    print(f"  P-value: {perm_results['p_value']:.4f}")
    print(f"  Significant: {perm_results['significant']}")
