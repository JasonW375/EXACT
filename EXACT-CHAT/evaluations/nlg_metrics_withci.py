#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute NLG scores with 95% confidence intervals using Bootstrap method (with parallel processing).
修复METEOR在并行处理中的pickle问题
"""

import argparse
import json
from pathlib import Path
import warnings
import numpy as np
import time
import multiprocessing as mp

import tqdm
from joblib import Parallel, delayed
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

# 抑制警告
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------ #
# 时间格式化工具
# ------------------------------------------------------------------ #

def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"

# ------------------------------------------------------------------ #
# Bootstrap置信区间计算
# ------------------------------------------------------------------ #

def compute_single_score(gts_subset, res_subset, scorer):
    """
    计算单个子集的分数
    
    Args:
        gts_subset: 真值子集 {idx: [text]}
        res_subset: 预测子集 {idx: [text]}
        scorer: 评分器对象
    
    Returns:
        分数（单个值或列表）
    """
    try:
        score, _ = scorer.compute_score(gts_subset, res_subset, verbose=0)
    except TypeError:
        score, _ = scorer.compute_score(gts_subset, res_subset)
    except FileNotFoundError:
        # METEOR需要Java
        return None
    
    return score


# ← 修改：添加scorer_type参数，在子进程中重新创建scorer
def _single_bootstrap_sample(seed, gts, res, scorer_type, metric_names, indices):
    """
    执行单次Bootstrap采样并计算分数
    
    Args:
        seed: 随机种子
        gts: 完整的真值集
        res: 完整的预测集
        scorer_type: 评分器类型字符串 ('bleu', 'rouge', 'cider', 'meteor')
        metric_names: 指标名称
        indices: 所有样本的索引列表
    
    Returns:
        该次采样的分数（单个值或列表）
    """
    rng = np.random.RandomState(seed)
    n_samples = len(indices)
    
    # 有放回抽样
    sampled_indices = rng.choice(indices, size=n_samples, replace=True)
    
    # 创建子集
    gts_subset = {i: gts[idx] for i, idx in enumerate(sampled_indices)}
    res_subset = {i: res[idx] for i, idx in enumerate(sampled_indices)}
    
    # ← 关键修改：在子进程中重新创建scorer对象
    if scorer_type == 'bleu':
        scorer = Bleu(4)
    elif scorer_type == 'rouge':
        scorer = Rouge()
    elif scorer_type == 'cider':
        scorer = Cider()
    elif scorer_type == 'meteor':
        scorer = Meteor()
    else:
        raise ValueError(f"Unknown scorer type: {scorer_type}")
    
    # 计算分数
    score = compute_single_score(gts_subset, res_subset, scorer)
    
    return score


def bootstrap_ci(gts, res, scorer, metric_names, n_bootstraps=1000, 
                confidence_level=0.95, random_state=42, n_jobs=-1):
    """
    使用Bootstrap方法计算置信区间（并行版本，修复METEOR pickle问题）
    
    Args:
        gts: 完整的真值集 {idx: [text]}
        res: 完整的预测集 {idx: [text]}
        scorer: 评分器对象
        metric_names: 指标名称（字符串或列表）
        n_bootstraps: Bootstrap采样次数
        confidence_level: 置信水平
        random_state: 随机种子
        n_jobs: 并行核心数 (-1表示使用所有核心)
    
    Returns:
        tuple: (results_dict, point_time, bootstrap_time)
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(gts)
    indices = list(gts.keys())
    
    # 计算点估计
    print(f"    Computing point estimate...", end=' ', flush=True)
    point_start = time.time()
    point_scores = compute_single_score(gts, res, scorer)
    point_time = time.time() - point_start
    
    if point_scores is None:
        # METEOR失败（Java不可用）
        print(f"Failed (Java not available)")
        if isinstance(metric_names, list):
            return {name: None for name in metric_names}, 0, 0
        else:
            return {metric_names: None}, 0, 0
    
    print(f"Done ({format_time(point_time)})")
    
    # ← 修改：确定scorer类型
    if isinstance(scorer, Bleu):
        scorer_type = 'bleu'
    elif isinstance(scorer, Rouge):
        scorer_type = 'rouge'
    elif isinstance(scorer, Cider):
        scorer_type = 'cider'
    elif isinstance(scorer, Meteor):
        scorer_type = 'meteor'
    else:
        scorer_type = 'unknown'
    
    # 使用并行Bootstrap采样
    print(f"    Bootstrap sampling ({n_bootstraps} iterations) with parallel processing:", flush=True)
    
    # 确定实际使用的核心数
    if n_jobs == -1:
        actual_jobs = mp.cpu_count()
    else:
        actual_jobs = min(n_jobs, mp.cpu_count())
    
    print(f"      Using {actual_jobs} CPU cores", flush=True)
    
    bootstrap_start = time.time()
    
    # ← 修改：传递scorer_type而不是scorer对象
    bootstrapped_scores = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_single_bootstrap_sample)(
            random_state + i,
            gts, 
            res, 
            scorer_type,  # ← 传递类型字符串
            metric_names,
            indices
        )
        for i in range(n_bootstraps)
    )
    
    # 过滤掉None值（如果有的话）
    bootstrapped_scores = [s for s in bootstrapped_scores if s is not None]
    
    bootstrap_time = time.time() - bootstrap_start
    print(f"      Completed: {len(bootstrapped_scores)}/{n_bootstraps} samples in {format_time(bootstrap_time)}")
    
    # 计算置信区间
    alpha = (1 - confidence_level) / 2
    results = {}
    
    if isinstance(metric_names, list):
        # BLEU返回多个分数
        point_scores_list = point_scores if isinstance(point_scores, (list, tuple)) else [point_scores]
        
        for i, name in enumerate(metric_names):
            if len(bootstrapped_scores) > 0:
                # 提取第i个指标的所有bootstrap值
                metric_bootstrap = [scores[i] for scores in bootstrapped_scores]
                ci_lower = np.percentile(metric_bootstrap, alpha * 100)
                ci_upper = np.percentile(metric_bootstrap, (1 - alpha) * 100)
                point = point_scores_list[i]
            else:
                point = point_scores_list[i]
                ci_lower = point
                ci_upper = point
            
            results[name] = f"{point:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]"
    else:
        # 单个分数（ROUGE, CIDEr, METEOR）
        if len(bootstrapped_scores) > 0:
            ci_lower = np.percentile(bootstrapped_scores, alpha * 100)
            ci_upper = np.percentile(bootstrapped_scores, (1 - alpha) * 100)
        else:
            ci_lower = point_scores
            ci_upper = point_scores
        
        results[metric_names] = f"{point_scores:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]"
    
    return results, point_time, bootstrap_time


# ------------------------------------------------------------------ #
def compute_scores_with_ci(gts, res, n_bootstraps=1000, confidence_level=0.95, 
                          random_state=42, n_jobs=-1):
    """
    计算所有NLG指标及其95%置信区间（并行版本）
    
    Args:
        gts: 真值 {idx: [text]}
        res: 预测 {idx: [text]}
        n_bootstraps: Bootstrap采样次数
        confidence_level: 置信水平
        random_state: 随机种子
        n_jobs: 并行核心数
    
    Returns:
        tuple: (scores_dict, timing_info)
    """
    scorers = [
        (Bleu(4),   ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Rouge(),   "ROUGE_L"),
        (Cider(),   "CIDEr"),
        (Meteor(),  "METEOR")
    ]
    
    out = {}
    timing_info = {}
    
    for scorer, names in scorers:
        metric_name = names if isinstance(names, str) else 'BLEU'
        
        print(f"\n{'='*60}")
        print(f"📊 Computing {metric_name} with 95% CI...")
        print(f"{'='*60}")
        
        metric_start = time.time()
        
        try:
            results, point_time, bootstrap_time = bootstrap_ci(
                gts, res, scorer, names, 
                n_bootstraps, confidence_level, random_state,
                n_jobs=n_jobs
            )
            
            metric_elapsed = time.time() - metric_start
            
            timing_info[metric_name] = {
                'point_estimate': point_time,
                'bootstrap': bootstrap_time,
                'total': metric_elapsed
            }
            
            if isinstance(names, list):
                # BLEU返回多个指标
                for name, value in results.items():
                    if value is not None:
                        out[name] = value
                
                # 计算BLEU_mean的置信区间
                if all(out.get(f"BLEU_{i}") is not None for i in range(1, 5)):
                    # 从字符串中提取数值
                    bleu_values = []
                    for i in range(1, 5):
                        val_str = out[f"BLEU_{i}"]
                        val = float(val_str.split('[')[0].strip())
                        bleu_values.append(val)
                    
                    # 计算BLEU_mean
                    mean_val = sum(bleu_values) / len(bleu_values)
                    
                    # 对BLEU_mean也计算CI
                    ci_lowers = []
                    ci_uppers = []
                    for i in range(1, 5):
                        val_str = out[f"BLEU_{i}"]
                        ci_part = val_str.split('[')[1].rstrip(']').split(',')
                        ci_lowers.append(float(ci_part[0].strip()))
                        ci_uppers.append(float(ci_part[1].strip()))
                    
                    mean_ci_lower = sum(ci_lowers) / len(ci_lowers)
                    mean_ci_upper = sum(ci_uppers) / len(ci_uppers)
                    
                    out["BLEU_mean"] = f"{mean_val:.4f} [{mean_ci_lower:.4f}, {mean_ci_upper:.4f}]"
                
                print(f"\n  ✅ {metric_name} completed in {format_time(metric_elapsed)}")
                print(f"     Point estimate: {format_time(point_time)}")
                print(f"     Bootstrap: {format_time(bootstrap_time)}")
            else:
                # 单个指标
                if results[names] is not None:
                    out[names] = results[names]
                    print(f"\n  ✅ {metric_name}: {results[names]}")
                    print(f"     Time: {format_time(metric_elapsed)} "
                          f"(point: {format_time(point_time)}, bootstrap: {format_time(bootstrap_time)})")
                else:
                    print(f"\n  ⚠️  {metric_name} skipped (Java not found)")
        
        except FileNotFoundError:
            print(f"\n  ⚠️  Java not found → skipping METEOR")
            continue
        except Exception as e:
            print(f"\n  ⚠️  Error: {e}")
            import traceback
            traceback.print_exc()  # ← 添加详细错误信息
            continue
    
    return out, timing_info


# ------------------------------------------------------------------ #
def run(pred_json: Path, gt_json: Path, out_json: Path, 
        n_bootstraps: int = 1000, confidence_level: float = 0.95,
        random_state: int = 42, n_jobs: int = -1):
    """
    运行NLG评估（含95% CI、时间统计和并行处理）
    
    Args:
        pred_json: 预测JSON文件
        gt_json: 真值JSON文件
        out_json: 输出JSON文件
        n_bootstraps: Bootstrap采样次数
        confidence_level: 置信水平
        random_state: 随机种子
        n_jobs: 并行核心数 (-1表示使用所有核心)
    """
    script_start = time.time()
    
    actual_jobs = mp.cpu_count() if n_jobs == -1 else min(n_jobs, mp.cpu_count())
    
    print(f"\n{'='*60}")
    print(f"🚀 NLG Evaluation with 95% CI (Parallel Processing)")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Bootstrap samples: {n_bootstraps}")
    print(f"  Confidence level: {confidence_level * 100}%")
    print(f"  CPU cores: {actual_jobs} (out of {mp.cpu_count()} available)")
    print(f"  Random seed: {random_state}")
    print(f"\nLoading data...")
    print(f"  Predictions: {pred_json}")
    print(f"  Ground truth: {gt_json}")
    
    # 加载数据
    load_start = time.time()
    
    with open(pred_json, "r", encoding="utf-8") as f:
        in_json = json.load(f)
        raw = in_json[0]["outputs"][0]["value"]
        pred_items = raw["generated_reports"]
    
    with open(gt_json, "r", encoding="utf-8") as f:
        gt_items = json.load(f)["generated_reports"]
    
    load_time = time.time() - load_start
    
    # 构建映射
    pred_map = {x["input_image_name"].rsplit(".", 1)[0]: x["report"]
                for x in pred_items}
    gt_map   = {x["input_image_name"].rsplit(".", 1)[0]: x["report"]
                for x in gt_items}
    
    # 匹配预测和真值
    gts, recs = {}, {}
    common_keys = sorted(set(gt_map) & set(pred_map))
    
    print(f"\nData loaded in {format_time(load_time)}:")
    print(f"  Predictions: {len(pred_map)}")
    print(f"  Ground truth: {len(gt_map)}")
    print(f"  Valid samples: {len(common_keys)}")
    
    if len(common_keys) == 0:
        print(f"\n❌ 错误: 没有找到有效的样本对！")
        print(f"   预测文件ID示例: {list(pred_map.keys())[:5]}")
        print(f"   真值文件ID示例: {list(gt_map.keys())[:5]}")
        
        results = {
            '_metadata': {
                'n_samples': 0,
                'error': 'No valid sample pairs found'
            }
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return
    
    # 预处理文本
    preprocess_start = time.time()
    
    for idx, key in enumerate(tqdm.tqdm(common_keys, desc="Preprocessing")):
        gts[idx]  = [gt_map[key].replace("\n", "").replace(" . ", ".").replace(" .", ".")]
        recs[idx] = [pred_map[key].replace("\n", "")
                     .replace("<|eot_id|>", "")
                     .replace("\"", "")
                     .replace("_", "")
                     .replace(" . ", ".")
                     .replace(" .", ".")]
    
    preprocess_time = time.time() - preprocess_start
    
    # 计算带置信区间的分数
    print(f"\nComputing NLG metrics...")
    
    scores, timing_info = compute_scores_with_ci(
        gts, recs, n_bootstraps, confidence_level, random_state, n_jobs=n_jobs
    )
    
    # 添加元数据
    scores['_metadata'] = {
        'n_samples': len(common_keys),
        'n_bootstraps': n_bootstraps,
        'confidence_level': confidence_level,
        'random_state': random_state,
        'n_jobs': actual_jobs,
        'timing': {
            'data_loading': load_time,
            'preprocessing': preprocess_time,
            'per_metric': timing_info,
            'total_script': time.time() - script_start
        }
    }
    
    # 保存结果
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    
    total_time = time.time() - script_start
    
    print(f"\n{'='*60}")
    print(f"✅ NLG metrics with 95% CI → {out_json}")
    print(f"{'='*60}")
    
    # 打印摘要
    print(f"\n📊 NLG Metrics Summary:")
    print("=" * 60)
    for key, value in scores.items():
        if not key.startswith('_'):
            print(f"  {key:12s}: {value}")
    
    # 打印时间统计
    print(f"\n{'='*60}")
    print(f"⏱️  Time Summary (with {actual_jobs} CPU cores)")
    print(f"{'='*60}")
    print(f"  Data loading:    {format_time(load_time)}")
    print(f"  Preprocessing:   {format_time(preprocess_time)}")
    print(f"  ")
    print(f"  Per-metric breakdown:")
    
    total_metric_time = 0
    for metric_name, times in timing_info.items():
        total_metric_time += times['total']
        print(f"    {metric_name:12s}: {format_time(times['total']):>10s} "
              f"(point: {format_time(times['point_estimate']):>8s}, "
              f"bootstrap: {format_time(times['bootstrap']):>10s})")
    
    print(f"  ")
    print(f"  Metrics total:   {format_time(total_metric_time)}")
    print(f"  Script total:    {format_time(total_time)}")
    print(f"{'='*60}")
    
    # 估算提速比
    if n_bootstraps >= 100:
        estimated_serial_time = (
            timing_info.get('BLEU', {}).get('bootstrap', 0) * 10 +
            timing_info.get('ROUGE_L', {}).get('bootstrap', 0) * 10 +
            timing_info.get('CIDEr', {}).get('bootstrap', 0) * 10 +
            timing_info.get('METEOR', {}).get('bootstrap', 0) * 10
        )
        
        if estimated_serial_time > 0:
            speedup = estimated_serial_time / total_metric_time
            print(f"\n🚀 Estimated speedup: {speedup:.1f}x")
            print(f"   (Estimated serial time: {format_time(estimated_serial_time)})")


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description='Compute NLG metrics with 95% CI (Parallel Processing)'
    )
    ap.add_argument("--pred_json", required=True,
                    help="Path to inferred.json (predictions)")
    ap.add_argument("--gt_json", required=True,
                    help="Path to ground_truth.json")
    ap.add_argument("--out_json", required=True,
                    help="Where to write the metrics")
    ap.add_argument("--n_bootstraps", type=int, default=1000,
                    help="Number of bootstrap samples (default: 1000)")
    ap.add_argument("--confidence_level", type=float, default=0.95,
                    help="Confidence level (default: 0.95)")
    ap.add_argument("--random_state", type=int, default=42,
                    help="Random seed for reproducibility (default: 42)")
    ap.add_argument("--n_jobs", type=int, default=-1,
                    help="Number of CPU cores to use (-1 = all cores, default: -1)")
    
    args = ap.parse_args()
    
    run(Path(args.pred_json), Path(args.gt_json), Path(args.out_json),
        args.n_bootstraps, args.confidence_level, args.random_state,
        args.n_jobs)