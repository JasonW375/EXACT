# confidence_interval.py
"""
计算分类指标的95%置信区间
使用Bootstrap重采样方法
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
    使用Bootstrap方法计算指标的置信区间
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签（0/1）
        y_prob: 预测概率（用于AUC等指标）
        metric_func: 指标计算函数
        metric_name: 指标名称
        n_bootstraps: Bootstrap采样次数
        confidence_level: 置信水平（默认0.95）
        random_state: 随机种子
        
    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)
    
    # 计算点估计
    if y_prob is not None and metric_name in ['auc', 'auprc']:
        point_estimate = metric_func(y_true, y_prob)
    else:
        point_estimate = metric_func(y_true, y_pred)
    
    # Bootstrap采样
    bootstrapped_scores = []
    
    for i in range(n_bootstraps):
        # 有放回采样
        indices = rng.randint(0, n_samples, n_samples)
        
        # 检查采样是否包含至少两个类别
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
    
    # 计算置信区间
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
    计算所有分类指标及其95%置信区间
    
    Args:
        y_true: 真实标签 (n_samples,) 或 (n_samples, n_classes)
        y_pred: 预测标签 (n_samples,) 或 (n_samples, n_classes)
        y_prob: 预测概率 (n_samples,) 或 (n_samples, n_classes)
        n_bootstraps: Bootstrap采样次数
        confidence_level: 置信水平
        random_state: 随机种子
        
    Returns:
        字典，包含每个指标的值和置信区间
    """
    
    # 检查是否是多标签情况
    if len(y_true.shape) > 1:
        # 多标签分类
        return calculate_multilabel_metrics_with_ci(
            y_true, y_pred, y_prob, 
            n_bootstraps, confidence_level, random_state
        )
    
    # 单标签分类
    results = {}
    
    # 定义要计算的指标
    metrics = {
        'accuracy': (accuracy_score, False),
        'precision': (precision_score, False),
        'recall': (recall_score, False),
        'f1': (f1_score, False),
    }
    
    # 如果有概率预测，添加AUC和AUPRC
    if y_prob is not None:
        metrics['auc'] = (roc_auc_score, True)
        metrics['auprc'] = (average_precision_score, True)
    
    # 计算每个指标
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
    计算多标签分类的指标及置信区间
    
    Args:
        y_true: 真实标签 (n_samples, n_classes)
        y_pred: 预测标签 (n_samples, n_classes)
        y_prob: 预测概率 (n_samples, n_classes)
        n_bootstraps: Bootstrap采样次数
        confidence_level: 置信水平
        random_state: 随机种子
        
    Returns:
        字典，包含宏平均和微平均指标及其置信区间
    """
    results = {}
    
    # 定义要计算的指标
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
    
    # 如果有概率，添加AUC指标
    if y_prob is not None:
        metrics_config['macro']['auc_macro'] = (
            lambda yt, yp: roc_auc_score(yt, yp, average='macro'), True
        )
        metrics_config['micro']['auc_micro'] = (
            lambda yt, yp: roc_auc_score(yt, yp, average='micro'), True
        )
    
    # 计算每个指标组
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
    计算每个类别的指标及置信区间
    
    Args:
        y_true: 真实标签 (n_samples, n_classes)
        y_pred: 预测标签 (n_samples, n_classes)
        y_prob: 预测概率 (n_samples, n_classes)
        class_names: 类别名称列表
        n_bootstraps: Bootstrap采样次数
        confidence_level: 置信水平
        random_state: 随机种子
        
    Returns:
        字典，包含每个类别每个指标的值和置信区间
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
    
    # 对每个类别分别计算
    for i, class_name in enumerate(class_names):
        y_true_class = y_true[:, i]
        y_pred_class = y_pred[:, i]
        y_prob_class = y_prob[:, i] if y_prob is not None else None
        
        # 检查是否有正样本
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
        
        # AUC (如果有概率)
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
    
    # 移除空的字典
    if results['auc_per_class'] is not None and not results['auc_per_class']:
        del results['auc_per_class']
    
    return results


if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)
    n_samples = 1000
    n_classes = 5
    
    # 生成模拟数据
    y_true = np.random.randint(0, 2, (n_samples, n_classes))
    y_prob = np.random.rand(n_samples, n_classes)
    y_pred = (y_prob > 0.5).astype(int)
    
    # 计算指标
    print("计算多标签分类指标及95% CI...")
    results = calculate_multilabel_metrics_with_ci(y_true, y_pred, y_prob)
    
    for metric_name, values in results.items():
        print(f"\n{metric_name}:")
        print(f"  Value: {values['value']:.4f}")
        print(f"  95% CI: {values['ci']}")