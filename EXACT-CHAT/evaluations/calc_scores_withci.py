#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute multi-label classification metrics with 95% confidence intervals.
修改版：输出格式为 "value [ci_lower, ci_upper]"
"""

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# 抑制sklearn的UndefinedMetricWarning
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


def bootstrap_ci(y_true, y_pred, metric_func, n_bootstraps=1000, confidence_level=0.95, random_state=42):
    """
    使用Bootstrap方法计算指标的置信区间
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        metric_func: 评估指标函数
        n_bootstraps: Bootstrap采样次数
        confidence_level: 置信水平
        random_state: 随机种子
    
    返回:
        point_estimate: 点估计值
        ci_lower: 置信区间下界
        ci_upper: 置信区间上界
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)
    
    # 计算点估计
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        point_estimate = metric_func(y_true, y_pred)
    
    # Bootstrap采样
    bootstrapped_scores = []
    
    for i in range(n_bootstraps):
        # 有放回抽样
        indices = rng.randint(0, n_samples, n_samples)
        
        # 检查是否至少有两个类别
        if len(np.unique(y_true[indices])) < 2:
            continue
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = metric_func(y_true[indices], y_pred[indices])
            bootstrapped_scores.append(score)
        except:
            continue
    
    # 如果Bootstrap失败，返回点估计
    if len(bootstrapped_scores) == 0:
        return point_estimate, point_estimate, point_estimate
    
    # 计算置信区间
    alpha = (1 - confidence_level) / 2
    ci_lower = np.percentile(bootstrapped_scores, alpha * 100)
    ci_upper = np.percentile(bootstrapped_scores, (1 - alpha) * 100)
    
    return point_estimate, ci_lower, ci_upper


def compute_metric_with_ci(y_true, y_pred, metric_func, metric_name, 
                           n_bootstraps=1000, confidence_level=0.95, random_state=42):
    """
    计算指标及其置信区间，返回格式化字符串
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        metric_func: 指标函数
        metric_name: 指标名称
        n_bootstraps: Bootstrap采样次数
        confidence_level: 置信水平
        random_state: 随机种子
    
    返回:
        格式化的字符串: "value [ci_lower, ci_upper]"
    """
    try:
        # 为不同的指标添加zero_division参数
        if metric_name == 'precision':
            wrapped_func = lambda yt, yp: precision_score(yt, yp, zero_division=0)
        elif metric_name == 'recall':
            wrapped_func = lambda yt, yp: recall_score(yt, yp, zero_division=0)
        elif metric_name == 'f1':
            wrapped_func = lambda yt, yp: f1_score(yt, yp, zero_division=0)
        else:
            wrapped_func = metric_func
        
        point, ci_low, ci_high = bootstrap_ci(
            y_true, y_pred, wrapped_func, 
            n_bootstraps, confidence_level, random_state
        )
        
        # 格式化为字符串: "value [ci_lower, ci_upper]"
        formatted_str = f"{point:.4f} [{ci_low:.4f}, {ci_high:.4f}]"
        
        return formatted_str
        
    except Exception as e:
        # 出错时返回默认值
        if metric_name not in ['precision', 'recall', 'f1']:
            print(f"Warning: Could not compute {metric_name}: {e}")
        return "0.0000 [0.0000, 0.0000]"


def evaluate(pred_csv: Path, gt_csv: Path, out_json: Path, 
            n_bootstraps: int = 1000, confidence_level: float = 0.95, 
            random_state: int = 42):
    """
    评估多标签分类性能，输出带置信区间的指标
    
    参数:
        pred_csv: 预测结果CSV文件路径
        gt_csv: 真实标签CSV文件路径
        out_json: 输出JSON文件路径
        n_bootstraps: Bootstrap采样次数
        confidence_level: 置信水平
        random_state: 随机种子
    """
    
    print(f"Loading data...")
    print(f"  Predictions: {pred_csv}")
    print(f"  Ground truth: {gt_csv}")
    
    # 读取数据
    pred = pd.read_csv(pred_csv)
    gt   = pd.read_csv(gt_csv)

    # 清理AccessionNo（移除文件扩展名）
    pred['AccessionNo'] = pred['AccessionNo'].str.replace('.npz',  '', regex=False)
    gt['AccessionNo']   = gt['AccessionNo'].str.replace('.nii.gz', '', regex=False)

    # 设置索引
    pred.set_index('AccessionNo', inplace=True)
    gt.set_index('AccessionNo',   inplace=True)

    # 对齐索引并转换为整数
    pred = pred.reindex(gt.index).astype(int)

    print(f"\nData shape:")
    print(f"  Samples: {len(gt)}")
    print(f"  Classes: {len(gt.columns)}")
    print(f"\nComputing metrics with {n_bootstraps} bootstrap samples...")
    print(f"  Confidence level: {confidence_level * 100}%")
    print(f"  Output format: value [ci_lower, ci_upper]")
    
    results = {"per_pathology": []}
    
    # 用于收集宏平均的原始数值（用于计算平均值）
    all_prec_values, all_rec_values, all_f1_values, all_acc_values = [], [], [], []
    all_prec_ci_lows, all_rec_ci_lows, all_f1_ci_lows, all_acc_ci_lows = [], [], [], []
    all_prec_ci_highs, all_rec_ci_highs, all_f1_ci_highs, all_acc_ci_highs = [], [], [], []

    # 对每个病理类别分别计算
    for i, col in enumerate(gt.columns, 1):
        print(f"  [{i}/{len(gt.columns)}] Processing: {col}", end='')
        
        y_true = gt[col].values
        y_pred = pred[col].values
        
        # 计算原始数值（用于宏平均）
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prec_val = precision_score(y_true, y_pred, zero_division=0)
            rec_val = recall_score(y_true, y_pred, zero_division=0)
            f1_val = f1_score(y_true, y_pred, zero_division=0)
            acc_val = accuracy_score(y_true, y_pred)
        
        # 计算带CI的格式化字符串
        prec_str = compute_metric_with_ci(
            y_true, y_pred, precision_score, 'precision',
            n_bootstraps, confidence_level, random_state
        )
        rec_str = compute_metric_with_ci(
            y_true, y_pred, recall_score, 'recall',
            n_bootstraps, confidence_level, random_state
        )
        f1_str = compute_metric_with_ci(
            y_true, y_pred, f1_score, 'f1',
            n_bootstraps, confidence_level, random_state
        )
        acc_str = compute_metric_with_ci(
            y_true, y_pred, accuracy_score, 'accuracy',
            n_bootstraps, confidence_level, random_state
        )
        
        # 提取CI边界（用于宏平均）
        def extract_ci(s):
            """从格式化字符串中提取value, ci_low, ci_high"""
            parts = s.split('[')
            value = float(parts[0].strip())
            ci_part = parts[1].rstrip(']').split(',')
            ci_low = float(ci_part[0].strip())
            ci_high = float(ci_part[1].strip())
            return value, ci_low, ci_high
        
        _, prec_ci_low, prec_ci_high = extract_ci(prec_str)
        _, rec_ci_low, rec_ci_high = extract_ci(rec_str)
        _, f1_ci_low, f1_ci_high = extract_ci(f1_str)
        _, acc_ci_low, acc_ci_high = extract_ci(acc_str)
        
        # 显示该类别是否有预测
        if y_pred.sum() == 0:
            print(" [⚠️  No predictions]")
        else:
            print(f" [✓ {y_pred.sum()} predictions]")
        
        # 保存每个类别的结果（新格式）
        results["per_pathology"].append({
            "name": col,
            "precision": prec_str,
            "recall": rec_str,
            "f1": f1_str,
            "accuracy": acc_str
        })
        
        # 收集用于宏平均的原始值和CI
        all_prec_values.append(prec_val)
        all_rec_values.append(rec_val)
        all_f1_values.append(f1_val)
        all_acc_values.append(acc_val)
        
        all_prec_ci_lows.append(prec_ci_low)
        all_rec_ci_lows.append(rec_ci_low)
        all_f1_ci_lows.append(f1_ci_low)
        all_acc_ci_lows.append(acc_ci_low)
        
        all_prec_ci_highs.append(prec_ci_high)
        all_rec_ci_highs.append(rec_ci_high)
        all_f1_ci_highs.append(f1_ci_high)
        all_acc_ci_highs.append(acc_ci_high)

    print(f"\n✅ Per-class metrics computed")
    print(f"Computing macro-averaged metrics...")
    
    # 计算宏平均
    macro_prec = sum(all_prec_values) / len(all_prec_values)
    macro_rec = sum(all_rec_values) / len(all_rec_values)
    macro_f1 = sum(all_f1_values) / len(all_f1_values)
    macro_acc = sum(all_acc_values) / len(all_acc_values)
    
    macro_prec_ci_low = sum(all_prec_ci_lows) / len(all_prec_ci_lows)
    macro_prec_ci_high = sum(all_prec_ci_highs) / len(all_prec_ci_highs)
    
    macro_rec_ci_low = sum(all_rec_ci_lows) / len(all_rec_ci_lows)
    macro_rec_ci_high = sum(all_rec_ci_highs) / len(all_rec_ci_highs)
    
    macro_f1_ci_low = sum(all_f1_ci_lows) / len(all_f1_ci_lows)
    macro_f1_ci_high = sum(all_f1_ci_highs) / len(all_f1_ci_highs)
    
    macro_acc_ci_low = sum(all_acc_ci_lows) / len(all_acc_ci_lows)
    macro_acc_ci_high = sum(all_acc_ci_highs) / len(all_acc_ci_highs)

    # 格式化宏平均结果
    results["macro"] = {
        "precision": f"{macro_prec:.4f} [{macro_prec_ci_low:.4f}, {macro_prec_ci_high:.4f}]",
        "recall": f"{macro_rec:.4f} [{macro_rec_ci_low:.4f}, {macro_rec_ci_high:.4f}]",
        "f1": f"{macro_f1:.4f} [{macro_f1_ci_low:.4f}, {macro_f1_ci_high:.4f}]",
        "accuracy": f"{macro_acc:.4f} [{macro_acc_ci_low:.4f}, {macro_acc_ci_high:.4f}]"
    }

    # 保存结果
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Classification metrics → {out_json}")
    
    # 打印摘要
    print(f"\n📊 Macro-averaged Results:")
    print(f"  Precision: {results['macro']['precision']}")
    print(f"  Recall:    {results['macro']['recall']}")
    print(f"  F1 Score:  {results['macro']['f1']}")
    print(f"  Accuracy:  {results['macro']['accuracy']}")
    
    # 统计有预测的类别数
    classes_with_predictions = sum(1 for val in all_prec_values if val > 0) + \
                              sum(1 for val in all_rec_values if val > 0)
    classes_with_predictions = min(classes_with_predictions, len(gt.columns))
    
    print(f"\n⚠️  Classes with predictions: {classes_with_predictions}/{len(gt.columns)}")
    if classes_with_predictions < len(gt.columns) / 2:
        print(f"   ⚠️  The model appears to predict very few positive samples!")


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description='Compute classification metrics with 95% CI (compact format)'
    )
    ap.add_argument("--pred_csv", required=True, help='Path to predictions CSV')
    ap.add_argument("--gt_csv",   required=True, help='Path to ground truth CSV')
    ap.add_argument("--out_json", required=True, help='Path to output JSON')
    ap.add_argument("--n_bootstraps", type=int, default=1000,
                   help='Number of bootstrap samples (default: 1000)')
    ap.add_argument("--confidence_level", type=float, default=0.95,
                   help='Confidence level (default: 0.95)')
    ap.add_argument("--random_state", type=int, default=42,
                   help='Random seed for reproducibility (default: 42)')
    args = ap.parse_args()

    evaluate(
        Path(args.pred_csv), 
        Path(args.gt_csv), 
        Path(args.out_json),
        args.n_bootstraps,
        args.confidence_level,
        args.random_state
    )