# confidence_interval.py
"""
Compute 95% confidence intervals for classification metrics
using bootstrap resampling.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, average_precision_score
)
from typing import Callable, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    metric_func: Callable = None,
    metric_name: str = "metric",
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Compute a confidence interval for a metric via the bootstrap.

    Args:
        y_true: ground-truth labels
        y_pred: predicted labels (0/1)
        y_prob: predicted probabilities (used by AUC-style metrics)
        metric_func: callable that computes the metric
        metric_name: metric name
        n_bootstraps: number of bootstrap resamples
        confidence_level: confidence level (default 0.95)
        random_state: random seed
        
    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)
    
    # Point estimate
    if y_prob is not None and metric_name in ['auc', 'auprc']:
        point_estimate = metric_func(y_true, y_prob)
    else:
        point_estimate = metric_func(y_true, y_pred)
    
    # Bootstrap resampling
    bootstrapped_scores = []
    
    for i in range(n_bootstraps):
        # Resample with replacement
        indices = rng.randint(0, n_samples, n_samples)
        
        # Require at least two classes in the resample
        if len(np.unique(y_true[indices])) < 2:
            continue
        
        try:
            if y_prob is not None and metric_name in ['auc', 'auprc']:
                score = metric_func(y_true[indices], y_prob[indices])
            else:
                score = metric_func(y_true[indices], y_pred[indices])
            bootstrapped_scores.append(score)
        except:
            continue
    
    # Confidence interval
    alpha = (1 - confidence_level) / 2
    ci_lower = np.percentile(bootstrapped_scores, alpha * 100)
    ci_upper = np.percentile(bootstrapped_scores, (1 - alpha) * 100)
    
    return point_estimate, ci_lower, ci_upper


def calculate_all_metrics_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Compute every classification metric with its 95% confidence interval.
    
    Args:
        y_true: ground-truth labels, shape (n_samples,) or (n_samples, n_classes)
        y_pred: predicted labels, shape (n_samples,) or (n_samples, n_classes)
        y_prob: predicted probabilities, shape (n_samples,) or (n_samples, n_classes)
        n_bootstraps: number of bootstrap resamples
        confidence_level: confidence level
        random_state: random seed
        
    Returns:
        Dict holding the value and confidence interval of each metric
    """
    
    # Multi-label input?
    if len(y_true.shape) > 1:
        # Multi-label classification
        return calculate_multilabel_metrics_with_ci(
            y_true, y_pred, y_prob, 
            n_bootstraps, confidence_level, random_state
        )
    
    # Single-label classification
    results = {}
    
    # Metrics to compute
    metrics = {
        'accuracy': (accuracy_score, False),
        'precision': (precision_score, False),
        'recall': (recall_score, False),
        'f1': (f1_score, False),
    }
    
    # Add AUC and AUPRC when probabilities are available
    if y_prob is not None:
        metrics['auc'] = (roc_auc_score, True)
        metrics['auprc'] = (average_precision_score, True)
    
    # Compute each metric
    for metric_name, (metric_func, use_prob) in metrics.items():
        try:
            if use_prob and y_prob is not None:
                point, ci_low, ci_high = bootstrap_ci(
                    y_true, y_pred, y_prob,
                    metric_func, metric_name,
                    n_bootstraps, confidence_level, random_state
                )
            else:
                point, ci_low, ci_high = bootstrap_ci(
                    y_true, y_pred, None,
                    metric_func, metric_name,
                    n_bootstraps, confidence_level, random_state
                )
            
            results[metric_name] = {
                'value': float(point),
                'ci_lower': float(ci_low),
                'ci_upper': float(ci_high),
                'ci': f"[{ci_low:.4f}, {ci_high:.4f}]"
            }
        except Exception as e:
            print(f"Warning: Could not compute {metric_name}: {e}")
            results[metric_name] = {
                'value': 0.0,
                'ci_lower': 0.0,
                'ci_upper': 0.0,
                'ci': "N/A"
            }
    
    return results


def calculate_multilabel_metrics_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Compute multi-label classification metrics with confidence intervals.
    
    Args:
        y_true: ground-truth labels, shape (n_samples, n_classes)
        y_pred: predicted labels, shape (n_samples, n_classes)
        y_prob: predicted probabilities, shape (n_samples, n_classes)
        n_bootstraps: number of bootstrap resamples
        confidence_level: confidence level
        random_state: random seed
        
    Returns:
        Dict holding macro- and micro-averaged metrics with confidence intervals
    """
    results = {}
    
    # Metrics to compute
    metrics_config = {
        'macro': {
            'accuracy': (lambda yt, yp: accuracy_score(yt.ravel(), yp.ravel()), False),
            'precision_macro': (lambda yt, yp: precision_score(yt, yp, average='macro', zero_division=0), False),
            'recall_macro': (lambda yt, yp: recall_score(yt, yp, average='macro', zero_division=0), False),
            'f1_macro': (lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0), False),
        },
        'micro': {
            'precision_micro': (lambda yt, yp: precision_score(yt, yp, average='micro', zero_division=0), False),
            'recall_micro': (lambda yt, yp: recall_score(yt, yp, average='micro', zero_division=0), False),
            'f1_micro': (lambda yt, yp: f1_score(yt, yp, average='micro', zero_division=0), False),
        }
    }
    
    # Add AUC metrics when probabilities are available
    if y_prob is not None:
        metrics_config['macro']['auc_macro'] = (
            lambda yt, yp: roc_auc_score(yt, yp, average='macro'), True
        )
        metrics_config['micro']['auc_micro'] = (
            lambda yt, yp: roc_auc_score(yt, yp, average='micro'), True
        )
    
    # Compute each metric group
    for group_name, metrics in metrics_config.items():
        for metric_name, (metric_func, use_prob) in metrics.items():
            try:
                if use_prob and y_prob is not None:
                    point, ci_low, ci_high = bootstrap_ci(
                        y_true, y_pred, y_prob,
                        metric_func, metric_name,
                        n_bootstraps, confidence_level, random_state
                    )
                else:
                    point, ci_low, ci_high = bootstrap_ci(
                        y_true, y_pred, None,
                        metric_func, metric_name,
                        n_bootstraps, confidence_level, random_state
                    )
                
                results[metric_name] = {
                    'value': float(point),
                    'ci_lower': float(ci_low),
                    'ci_upper': float(ci_high),
                    'ci': f"[{ci_low:.4f}, {ci_high:.4f}]"
                }
            except Exception as e:
                print(f"Warning: Could not compute {metric_name}: {e}")
                results[metric_name] = {
                    'value': 0.0,
                    'ci_lower': 0.0,
                    'ci_upper': 0.0,
                    'ci': "N/A"
                }
    
    return results


def calculate_per_class_metrics_with_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    class_names: list = None,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute per-class metrics with confidence intervals.
    
    Args:
        y_true: ground-truth labels, shape (n_samples, n_classes)
        y_pred: predicted labels, shape (n_samples, n_classes)
        y_prob: predicted probabilities, shape (n_samples, n_classes)
        class_names: list of class names
        n_bootstraps: number of bootstrap resamples
        confidence_level: confidence level
        random_state: random seed
        
    Returns:
        Dict holding the value and confidence interval of each metric per class
    """
    n_classes = y_true.shape[1]
    
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(n_classes)]
    
    results = {
        'precision_per_class': {},
        'recall_per_class': {},
        'f1_per_class': {},
        'auc_per_class': {} if y_prob is not None else None
    }
    
    # Compute metrics for each class separately
    for i, class_name in enumerate(class_names):
        y_true_class = y_true[:, i]
        y_pred_class = y_pred[:, i]
        y_prob_class = y_prob[:, i] if y_prob is not None else None
        
        # Skip classes without positive samples
        if y_true_class.sum() == 0:
            print(f"Warning: No positive samples for class {class_name}, skipping")
            continue
        
        # Precision
        try:
            point, ci_low, ci_high = bootstrap_ci(
                y_true_class, y_pred_class, None,
                lambda yt, yp: precision_score(yt, yp, zero_division=0),
                'precision',
                n_bootstraps, confidence_level, random_state
            )
            results['precision_per_class'][class_name] = {
                'value': float(point),
                'ci_lower': float(ci_low),
                'ci_upper': float(ci_high),
                'ci': f"[{ci_low:.4f}, {ci_high:.4f}]"
            }
        except:
            pass
        
        # Recall
        try:
            point, ci_low, ci_high = bootstrap_ci(
                y_true_class, y_pred_class, None,
                lambda yt, yp: recall_score(yt, yp, zero_division=0),
                'recall',
                n_bootstraps, confidence_level, random_state
            )
            results['recall_per_class'][class_name] = {
                'value': float(point),
                'ci_lower': float(ci_low),
                'ci_upper': float(ci_high),
                'ci': f"[{ci_low:.4f}, {ci_high:.4f}]"
            }
        except:
            pass
        
        # F1
        try:
            point, ci_low, ci_high = bootstrap_ci(
                y_true_class, y_pred_class, None,
                lambda yt, yp: f1_score(yt, yp, zero_division=0),
                'f1',
                n_bootstraps, confidence_level, random_state
            )
            results['f1_per_class'][class_name] = {
                'value': float(point),
                'ci_lower': float(ci_low),
                'ci_upper': float(ci_high),
                'ci': f"[{ci_low:.4f}, {ci_high:.4f}]"
            }
        except:
            pass
        
        # AUC (when probabilities are available)
        if y_prob_class is not None and results['auc_per_class'] is not None:
            try:
                point, ci_low, ci_high = bootstrap_ci(
                    y_true_class, y_pred_class, y_prob_class,
                    roc_auc_score,
                    'auc',
                    n_bootstraps, confidence_level, random_state
                )
                results['auc_per_class'][class_name] = {
                    'value': float(point),
                    'ci_lower': float(ci_low),
                    'ci_upper': float(ci_high),
                    'ci': f"[{ci_low:.4f}, {ci_high:.4f}]"
                }
            except:
                pass
    
    # Drop the dict if it stayed empty
    if results['auc_per_class'] is not None and not results['auc_per_class']:
        del results['auc_per_class']
    
    return results


if __name__ == "__main__":
    # Smoke test
    np.random.seed(42)
    n_samples = 1000
    n_classes = 5
    
    # Synthetic data
    y_true = np.random.randint(0, 2, (n_samples, n_classes))
    y_prob = np.random.rand(n_samples, n_classes)
    y_pred = (y_prob > 0.5).astype(int)
    
    # Compute metrics
    print("Computing multi-label metrics with 95% CI...")
    results = calculate_multilabel_metrics_with_ci(y_true, y_pred, y_prob)
    
    for metric_name, values in results.items():
        print(f"\n{metric_name}:")
        print(f"  Value: {values['value']:.4f}")
        print(f"  95% CI: {values['ci']}")