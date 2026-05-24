#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master evaluation script – now also computes CRG metrics
Modified to work with prepared evaluation files - ENHANCED VERSION with 95% CI
支持文件夹模式：可以指定文件夹路径，自动搜索所有predictions_checkpoint*.json文件
增加95%置信区间功能
"""

import json, subprocess, sys, glob
from pathlib import Path
import re

# === CLI configuration ===
def _parse_cli():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prediction_jsons", nargs="+", required=True,
                    help="One or more EXACT-CHAT inference output JSONs to evaluate.")
    ap.add_argument("--base_dir", required=True, type=Path,
                    help="Directory containing per-prediction subfolders with prepared "
                         "ground_truth.json/csv and result_transformat.json.")
    ap.add_argument("--code_dir", type=Path, default=Path(__file__).resolve().parent,
                    help="Directory of the evaluation scripts (defaults to this file's dir).")
    return ap.parse_args()

_ARGS = _parse_cli()
MANUAL_INPUT_FILES = [Path(p) for p in _ARGS.prediction_jsons]
BASE_DIR = _ARGS.base_dir
CODE_DIR = _ARGS.code_dir
USE_FOLDER_MODE = False
FOLDER_PATH = None
FILE_PATTERN = None

# 评估脚本路径
INFER_SCRIPT = CODE_DIR / "infer.py"
CLS_SCRIPT   = CODE_DIR / "calc_scores_withci.py"  # 修改：使用带CI的版本
CRG_SCRIPT   = CODE_DIR / "crg_score.py"
NLG_SCRIPT   = CODE_DIR / "nlg_metrics_withci.py"  # 修改：使用带CI的版本

def extract_checkpoint_number(filename: str) -> int:
    """从文件名中提取checkpoint编号，用于排序"""
    match = re.search(r'checkpoint(\d+)', filename)
    return int(match.group(1)) if match else 0

def find_prediction_files(folder_path: Path, pattern: str) -> list[Path]:
    """在指定文件夹中查找符合模式的预测文件"""
    print(f"🔍 在文件夹中搜索预测文件: {folder_path}")
    print(f"   搜索模式: {pattern}")
    
    if not folder_path.exists():
        print(f"❌ 文件夹不存在: {folder_path}")
        return []
    
    if not folder_path.is_dir():
        print(f"❌ 路径不是文件夹: {folder_path}")
        return []
    
    # 使用glob模式匹配文件
    files = list(folder_path.glob(pattern))
    
    if not files:
        print(f"⚠️  未找到符合模式的文件")
        print(f"   请检查文件夹中是否存在 {pattern} 格式的文件")
        return []
    
    # 按checkpoint编号排序，确保处理顺序
    files.sort(key=lambda x: extract_checkpoint_number(x.name))
    
    print(f"✅ 找到 {len(files)} 个预测文件:")
    for i, f in enumerate(files, 1):
        checkpoint_num = extract_checkpoint_number(f.name)
        if checkpoint_num > 0:
            print(f"   {i}. {f.name} (checkpoint-{checkpoint_num})")
        else:
            print(f"   {i}. {f.name}")
    
    return files

def get_input_files() -> list[Path]:
    """根据配置获取输入文件列表"""
    if USE_FOLDER_MODE:
        print("📁 使用文件夹模式")
        return find_prediction_files(FOLDER_PATH, FILE_PATTERN)
    else:
        print("📋 使用手动文件列表模式")
        return MANUAL_INPUT_FILES

def get_output_dir_name(json_file_path: Path) -> str:
    """根据json文件名生成输出目录名 - 与准备脚本保持一致"""
    name = json_file_path.stem
    return name.replace(" ", "_")

def run(script: Path, *args, realtime=True):
    """
    执行子脚本
    
    Args:
        script: 脚本路径
        *args: 脚本参数
        realtime: 是否实时显示输出（True=实时，False=缓存后显示）
    """
    cmd = [sys.executable, str(script), *map(str, args)]
    print(">>", " ".join(cmd))
    
    try:
        if realtime:
            # 实时输出模式（推荐，可以看到进度）
            result = subprocess.run(cmd, check=True)
        else:
            # 缓存输出模式
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
    
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 脚本执行失败: {e}")
        if hasattr(e, 'stdout') and e.stdout:
            print("STDOUT:", e.stdout)
        if hasattr(e, 'stderr') and e.stderr:
            print("STDERR:", e.stderr, file=sys.stderr)
        raise

def load(p: Path):
    """加载JSON文件"""
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def check_required_files(prep_dir: Path):
    """检查准备脚本生成的必要文件是否存在"""
    required_files = [
        "result_transformat.json",  # infer.py需要的格式
        "ground_truth.json",        # NLG评估需要的参考数据
        "ground_truth.csv"          # 分类评估需要的参考数据
    ]
    
    missing = []
    for filename in required_files:
        file_path = prep_dir / filename
        if not file_path.exists():
            missing.append(filename)
    
    return missing

def check_csv_content(csv_path: Path):
    """检查CSV文件内容，诊断分类问题"""
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"📊 CSV文件分析 ({csv_path.name}):")
        print(f"   总行数: {len(df)}")
        
        if len(df) > 0:
            # 检查标签列（除了AccessionNo）
            label_cols = [col for col in df.columns if col != 'AccessionNo']
            print(f"   标签列数: {len(label_cols)}")
            
            # 统计每列的正样本数量
            positive_counts = {}
            for col in label_cols:
                if col in df.columns:
                    positive_count = (df[col] == 1).sum()
                    positive_counts[col] = positive_count
            
            # 显示正样本统计
            total_positives = sum(positive_counts.values())
            print(f"   总正样本数: {total_positives}")
            
            if total_positives == 0:
                print("   ⚠️  警告：所有标签都是0，这可能导致分类评估问题")
            
            # 显示前5个有正样本的类别
            positive_classes = [(k, v) for k, v in positive_counts.items() if v > 0]
            if positive_classes:
                print("   有正样本的类别 (前5个):")
                for cls, count in sorted(positive_classes, key=lambda x: x[1], reverse=True)[:5]:
                    print(f"     {cls}: {count}")
            
        else:
            print("   ⚠️  警告：CSV文件为空")
            
    except Exception as e:
        print(f"   ❌ 无法分析CSV文件: {e}")

def parse_metric_string(metric_str):
    """解析指标字符串 "value [ci_lower, ci_upper]" """
    if isinstance(metric_str, (int, float)):
        return {
            'value': float(metric_str),
            'ci_lower': float(metric_str),
            'ci_upper': float(metric_str)
        }
    
    try:
        parts = metric_str.split('[')
        value = float(parts[0].strip())
        ci_part = parts[1].rstrip(']').split(',')
        ci_lower = float(ci_part[0].strip())
        ci_upper = float(ci_part[1].strip())
        
        return {
            'value': value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    except (IndexError, ValueError) as e:
        return {
            'value': 0.0,
            'ci_lower': 0.0,
            'ci_upper': 0.0
        }

def evaluate_single_file(input_file: Path, file_index: int, total_files: int):
    """对单个输入文件进行完整的评估（含95% CI）"""
    checkpoint_num = extract_checkpoint_number(input_file.name)
    
    print(f"\n{'='*60}")
    print(f"处理文件 [{file_index}/{total_files}]: {input_file.name}")
    if checkpoint_num > 0:
        print(f"Checkpoint: {checkpoint_num}")
    print(f"{'='*60}")
    
    # 确定准备文件所在的目录（与准备脚本保持一致）
    output_dir_name = get_output_dir_name(input_file)
    prep_dir = BASE_DIR / output_dir_name
    
    print(f"Prepared files directory: {prep_dir}")
    
    # 检查准备目录是否存在
    if not prep_dir.exists():
        print(f"❌ 准备文件目录不存在: {prep_dir}")
        print("请先运行准备脚本生成必要的文件。")
        return False
    
    # 检查必要文件是否存在
    missing_files = check_required_files(prep_dir)
    if missing_files:
        print(f"❌ 以下必要文件缺失:")
        for f in missing_files:
            print(f"   {f}")
        print("请确保准备脚本已成功运行。")
        return False
    
    # 定义文件路径
    # 输入文件（准备脚本生成的）
    pred_json_transformed = prep_dir / "result_transformat.json"
    gt_json = prep_dir / "ground_truth.json"
    gt_csv = prep_dir / "ground_truth.csv"
    
    # 输出文件（评估结果）
    csv_pred = prep_dir / "inferred.csv"
    cls_json = prep_dir / "classification_scores.json"
    crg_json = prep_dir / "crg_scores.json"
    nlg_json = prep_dir / "nlg_scores.json"
    final_json = prep_dir / "metrics.json"
    
    print(f"使用转换后的预测文件: {pred_json_transformed.name}")
    print(f"Ground truth JSON: {gt_json.name}")
    print(f"Ground truth CSV: {gt_csv.name}")
    
    try:
        # 1. inference → CSV (使用转换后的文件)
        print("\n1. Running inference...")
        run(INFER_SCRIPT,
            "--input_json", pred_json_transformed,
            "--model_path", CODE_DIR / "roberta_local" / "RadBertClassifier.pth",
            "--out_csv", csv_pred,
            realtime=True)
        print(f"✅ 推理完成，结果保存到: {csv_pred.name}")
        
        # 分析推理结果
        if csv_pred.exists():
            check_csv_content(csv_pred)

        # 2. multi-label classification scores (含95% CI)
        print("\n2. Computing classification scores with 95% CI...")
        print(f"   Bootstrap samples: {N_BOOTSTRAPS}")
        print(f"   Confidence level: {CONFIDENCE_LEVEL * 100}%")
        print("   注意：如果看到精确率警告，这通常是因为某些类别预测样本很少")
        run(CLS_SCRIPT,
            "--pred_csv", csv_pred,
            "--gt_csv", gt_csv,
            "--out_json", cls_json,
            "--n_bootstraps", str(N_BOOTSTRAPS),
            "--confidence_level", str(CONFIDENCE_LEVEL),
            "--random_state", str(RANDOM_STATE),
            realtime=True)
        print(f"✅ 分类评估完成（含95% CI），结果保存到: {cls_json.name}")

        # 3. CRG metrics
        print("\n3. Computing CRG metrics...")
        run(CRG_SCRIPT,
            "--pred_csv", csv_pred,
            "--gt_csv", gt_csv,
            "--out_json", crg_json,
            realtime=True)
        print(f"✅ CRG评估完成，结果保存到: {crg_json.name}")

        # 4. NLG metrics (含95% CI + 并行处理)
        print("\n4. Computing NLG metrics with 95% CI (Parallel Processing)...")
        print("   🔧 使用转换后的文件进行NLG评估")
        print(f"   Bootstrap samples: {N_BOOTSTRAPS}")
        print(f"   Confidence level: {CONFIDENCE_LEVEL * 100}%")
        print(f"   🚀 并行核心数: {N_JOBS}")
        print()  # 空行用于清晰显示
        
        run(NLG_SCRIPT,
            "--pred_json", pred_json_transformed,
            "--gt_json", gt_json,
            "--out_json", nlg_json,
            "--n_bootstraps", str(N_BOOTSTRAPS),
            "--confidence_level", str(CONFIDENCE_LEVEL),
            "--random_state", str(RANDOM_STATE),
            "--n_jobs", str(N_JOBS),
            realtime=True)
        
        print()  # 空行
        print(f"✅ NLG评估完成（含95% CI + 并行加速），结果保存到: {nlg_json.name}")

        # 5. Combine all metrics
        print("\n5. Combining all metrics...")
        cls_metrics = load(cls_json)
        crg_metrics = load(crg_json)
        nlg_metrics = load(nlg_json)
        
        # 清理NLG指标中的内部字段
        nlg_metrics_clean = {k: v for k, v in nlg_metrics.items() if not k.startswith('_')}
        
        combined = {
            "generation": nlg_metrics_clean,
            "classification": cls_metrics,
            "crg": crg_metrics,
            "metadata": {
                "checkpoint": checkpoint_num if checkpoint_num > 0 else "N/A",
                "input_file": input_file.name,
                "evaluation_type": "full_evaluation_with_ci_parallel",
                "bootstrap_config": {
                    "n_bootstraps": N_BOOTSTRAPS,
                    "confidence_level": CONFIDENCE_LEVEL,
                    "random_state": RANDOM_STATE,
                    "nlg_n_jobs": N_JOBS
                },
                "note": "分类和NLG指标包含95%置信区间，NLG使用并行加速"
            }
        }
        
        with open(final_json, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 所有评估指标已写入: {final_json}")
        
        # 打印结果摘要
        print(f"\n📊 评估结果详情 (Checkpoint-{checkpoint_num if checkpoint_num > 0 else 'N/A'}):")
        print("=" * 80)
        
        # 分类指标（含CI）
        if "classification" in combined and combined["classification"]:
            cls_data = combined["classification"]
            print("🔍 分类指标 (含95% CI):")
            
            if 'macro' in cls_data:
                print("\n   宏平均指标:")
                macro = cls_data['macro']
                for key in ['precision', 'recall', 'f1', 'accuracy']:
                    if key in macro:
                        value_str = macro[key]
                        print(f"      {key:12s}: {value_str}")
            
            if 'per_pathology' in cls_data and len(cls_data['per_pathology']) > 0:
                print("\n   各类别指标 (前3个):")
                for item in cls_data['per_pathology'][:3]:
                    print(f"\n      {item['name']}:")
                    for key in ['precision', 'recall', 'f1', 'accuracy']:
                        if key in item:
                            value_str = item[key]
                            print(f"         {key:12s}: {value_str}")
                
                if len(cls_data['per_pathology']) > 3:
                    print(f"\n      ... 以及其他 {len(cls_data['per_pathology']) - 3} 个类别")
        
        # 生成指标（含CI）
        if "generation" in combined and combined["generation"]:
            gen_metrics = combined["generation"]
            print(f"\n📝 生成指标 (含95% CI, {N_JOBS}核并行):")
            for key, value in gen_metrics.items():
                if isinstance(value, str):
                    print(f"   {key:12s}: {value}")
                elif isinstance(value, (int, float)):
                    print(f"   {key:12s}: {value:.4f}")
        
        # CRG指标
        if "crg" in combined and combined["crg"]:
            crg_data = combined["crg"]
            print("\n🏥 CRG指标:")
            for key, value in crg_data.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 评估过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def generate_summary_report(successful_files: list[Path], base_dir: Path):
    """生成所有成功评估文件的汇总报告（含95% CI）"""
    if not successful_files:
        return
    
    print(f"\n{'='*60}")
    print("📋 生成汇总报告")
    print(f"{'='*60}")
    
    summary_data = []
    
    for input_file in successful_files:
        output_dir_name = get_output_dir_name(input_file)
        metrics_file = base_dir / output_dir_name / "metrics.json"
        
        if metrics_file.exists():
            try:
                metrics = load(metrics_file)
                checkpoint_num = extract_checkpoint_number(input_file.name)
                
                # 提取关键指标
                summary_item = {
                    "checkpoint": checkpoint_num if checkpoint_num > 0 else "N/A",
                    "filename": input_file.name
                }
                
                # 分类指标（含CI）
                if "classification" in metrics and "macro" in metrics["classification"]:
                    macro = metrics["classification"]["macro"]
                    for key in ['precision', 'recall', 'f1', 'accuracy']:
                        if key in macro:
                            value_str = macro[key]
                            summary_item[f"cls_{key}"] = value_str
                
                # 生成指标（含CI）
                if "generation" in metrics:
                    gen_metrics = metrics["generation"]
                    for key, value in gen_metrics.items():
                        if isinstance(value, str):
                            summary_item[f"gen_{key}"] = value
                        elif isinstance(value, (int, float)):
                            summary_item[f"gen_{key}"] = round(value, 4)
                
                # CRG指标
                if "crg" in metrics:
                    crg_metrics = metrics["crg"]
                    for key, value in crg_metrics.items():
                        if isinstance(value, (int, float)):
                            summary_item[f"crg_{key}"] = round(value, 4)
                
                summary_data.append(summary_item)
                
            except Exception as e:
                print(f"⚠️  无法加载 {metrics_file}: {e}")
    
    # 保存汇总报告
    if summary_data:
        summary_file = base_dir / "evaluation_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"📊 汇总报告已保存: {summary_file}")
        
        # 显示简化表格
        print(f"\n📈 性能对比表格 (含95% CI):")
        print("-" * 150)
        print(f"{'Checkpoint':<12} {'文件名':<30} {'分类F1(95%CI)':<35} {'BLEU-1(95%CI)':<35} {'ROUGE-L(95%CI)':<35}")
        print("-" * 150)
        
        for item in sorted(summary_data, key=lambda x: x['checkpoint'] if isinstance(x['checkpoint'], int) else 0):
            checkpoint = item['checkpoint']
            filename = item['filename'][:27] + "..." if len(item['filename']) > 30 else item['filename']
            
            def format_metric(key):
                if key in item:
                    value = item[key]
                    if isinstance(value, str):
                        return value
                    else:
                        return f"{value:.4f}"
                return "N/A"
            
            cls_f1 = format_metric('cls_f1')
            bleu1 = format_metric('gen_BLEU_1')
            rougeL = format_metric('gen_ROUGE_L')
            
            print(f"{str(checkpoint):<12} {filename:<30} {cls_f1:<35} {bleu1:<35} {rougeL:<35}")
        
        print("-" * 150)
        print("📝 完整数据请查看: evaluation_summary.json")
        
        # 显示可用的评估指标
        if summary_data:
            all_metrics = set()
            for item in summary_data:
                all_metrics.update([k for k in item.keys() if k not in ['checkpoint', 'filename']])
            
            if all_metrics:
                print(f"\n📊 可用的评估指标:")
                cls_metrics = sorted([m for m in all_metrics if m.startswith('cls_')])
                gen_metrics = sorted([m for m in all_metrics if m.startswith('gen_')])
                crg_metrics = sorted([m for m in all_metrics if m.startswith('crg_')])
                
                if cls_metrics:
                    print(f"   分类指标 (含95% CI): {', '.join(cls_metrics)}")
                if gen_metrics:
                    print(f"   生成指标 (含95% CI): {', '.join(gen_metrics)}")
                if crg_metrics:
                    print(f"   CRG指标: {', '.join(crg_metrics)}")

def main():
    """主函数，处理所有输入文件"""
    print("🚀 主评估脚本 - 增强版（含95%置信区间 + 文件夹模式 + 并行加速）")
    print(f"基础目录: {BASE_DIR}")
    print(f"评估脚本目录: {CODE_DIR}")
    print(f"🔬 Bootstrap参数: n={N_BOOTSTRAPS}, CI={CONFIDENCE_LEVEL*100}%")
    print(f"🚀 NLG并行核心数: {N_JOBS}")
    
    # 显示当前配置
    mode_text = "文件夹模式" if USE_FOLDER_MODE else "手动文件列表模式"
    print(f"🔧 当前模式: {mode_text}")
    
    if USE_FOLDER_MODE:
        print(f"📁 搜索文件夹: {FOLDER_PATH}")
        print(f"🔍 文件模式: {FILE_PATTERN}")
    
    # 检查评估脚本目录是否存在
    if not CODE_DIR.exists():
        print(f"❌ 评估脚本目录不存在: {CODE_DIR}")
        return
    
    # 检查必要的评估脚本是否存在
    required_scripts = [INFER_SCRIPT, CLS_SCRIPT, CRG_SCRIPT, NLG_SCRIPT]
    missing_scripts = [script for script in required_scripts if not script.exists()]
    if missing_scripts:
        print("❌ 以下评估脚本缺失:")
        for script in missing_scripts:
            print(f"   {script}")
        print("\n💡 提示：")
        print("   - calc_scores_withci.py: 带置信区间的分类评估脚本")
        print("   - nlg_metrics_withci.py: 带置信区间的NLG评估脚本")
        return
    
    # 获取输入文件列表
    input_files = get_input_files()
    if not input_files:
        print("❌ 没有找到可处理的输入文件")
        if USE_FOLDER_MODE:
            print("提示：请检查文件夹路径和文件模式是否正确")
        return
    
    # 检查输入文件是否存在
    missing_files = [f for f in input_files if not f.exists()]
    if missing_files:
        print("❌ 以下输入文件不存在:")
        for f in missing_files:
            print(f"   {f}")
        return
    
    print(f"\n📁 将处理 {len(input_files)} 个输入文件:")
    for i, f in enumerate(input_files, 1):
        checkpoint_num = extract_checkpoint_number(f.name)
        if checkpoint_num > 0:
            print(f"   {i}. {f.name} (checkpoint-{checkpoint_num})")
        else:
            print(f"   {i}. {f.name}")
    
    # 处理每个文件
    successful = 0
    failed = 0
    successful_files = []
    
    for i, input_file in enumerate(input_files, 1):
        if evaluate_single_file(input_file, i, len(input_files)):
            successful += 1
            successful_files.append(input_file)
        else:
            failed += 1
        
        # 处理过程中显示进度
        if i < len(input_files):
            print(f"\n⏭️  准备处理下一个文件... ({i}/{len(input_files)} 完成)")
    
    # 生成汇总报告
    if successful_files:
        generate_summary_report(successful_files, BASE_DIR)
    
    # 最终总结
    print(f"\n{'='*60}")
    print("🎯 最终结果总结")
    print(f"{'='*60}")
    print(f"总文件数: {len(input_files)}")
    print(f"✅ 成功: {successful}")
    print(f"❌ 失败: {failed}")
    print(f"📊 成功率: {successful/len(input_files)*100:.1f}%" if input_files else "N/A")
    
    if successful > 0:
        print(f"\n🎉 评估结果可在以下目录查看:")
        for input_file in successful_files:
            output_dir_name = get_output_dir_name(input_file)
            result_dir = BASE_DIR / output_dir_name
            print(f"   📊 {result_dir}/metrics.json")
    
    print("\n📋 评估完成！")
    
    # 提供问题诊断建议
    if failed > 0:
        print("\n🔧 问题诊断建议:")
        print("1. 确保准备脚本已正确运行并生成必要文件")
        print("2. 检查ground_truth.csv是否包含有效的标签数据")
        print("3. 确认分类器模型路径正确且模型文件存在")
        print("4. 检查CSV文件格式和列名是否正确")
        print("5. 确认calc_scores_withci.py和nlg_metrics_withci.py已正确放置")
        print("6. 确保安装了所需的NLG库: nltk, rouge_score, joblib")
    
    if successful > 0:
        print("\n💡 结果文件说明:")
        print("   📄 inferred.csv - 模型预测的分类结果")
        print("   📄 classification_scores.json - 详细的分类指标（含95% CI）")
        print("   📄 crg_scores.json - CRG指标")
        print(f"   📄 nlg_scores.json - NLG指标（含95% CI，{N_JOBS}核并行加速）")
        print("   📄 metrics.json - 最终合并的评估结果")
        print("   📄 evaluation_summary.json - 所有文件的汇总对比（含95% CI）")
        print(f"\n🔬 置信区间参数: Bootstrap samples={N_BOOTSTRAPS}, CI level={CONFIDENCE_LEVEL*100}%")
        print(f"🚀 NLG加速: {N_JOBS}核并行处理（实时显示进度）")
        print(f"📊 评估指标: 分类(含CI) + CRG + NLG(含CI)")

if __name__ == "__main__":
    main()