#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master evaluation script – now also computes CRG metrics
Modified to work with prepared evaluation files - ENHANCED VERSION
支持文件夹模式：可以指定文件夹路径，自动搜索所有predictions_checkpoint*.json文件
增加占位类别过滤功能
"""

import json, subprocess, sys, glob
from pathlib import Path
import re
import pandas as pd

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

# === 排除的占位类别 ===
EXCLUDED_CLASSES = [
    "Mosaic attenuation pattern"
]

# 评估脚本路径
INFER_SCRIPT = CODE_DIR / "infer.py"
CLS_SCRIPT   = CODE_DIR / "calc_scores.py"
CRG_SCRIPT   = CODE_DIR / "crg_score_mianyang.py"
NLG_SCRIPT   = CODE_DIR / "nlg_metrics.py"

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
        print(f"   {i}. {f.name} (checkpoint-{checkpoint_num})")
    
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

def run(script: Path, *args):
    """执行子脚本"""
    cmd = [sys.executable, str(script), *map(str, args)]
    print(">>", " ".join(cmd))
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"❌ 脚本执行失败: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
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

def remove_excluded_columns(csv_path: Path, output_path: Path, excluded_classes: list):
    """
    从CSV文件中移除指定的占位类别列
    
    Args:
        csv_path: 原始CSV文件路径
        output_path: 输出CSV文件路径
        excluded_classes: 要排除的类别列表
    """
    df = pd.read_csv(csv_path)
    
    # 记录原始列数
    original_cols = len(df.columns)
    
    # 移除排除的列（如果存在）
    cols_to_drop = [col for col in excluded_classes if col in df.columns]
    
    if cols_to_drop:
        print(f"   移除占位列: {', '.join(cols_to_drop)}")
        df = df.drop(columns=cols_to_drop)
    
    # 保存处理后的CSV
    df.to_csv(output_path, index=False)
    
    print(f"   原始列数: {original_cols}, 处理后列数: {len(df.columns)}")
    
    return len(cols_to_drop)

def check_csv_content(csv_path: Path, excluded_classes: list = None):
    """检查CSV文件内容，诊断分类问题"""
    try:
        df = pd.read_csv(csv_path)
        print(f"📊 CSV文件分析 ({csv_path.name}):")
        print(f"   总行数: {len(df)}")
        
        if len(df) > 0:
            # 检查标签列（除了AccessionNo）
            label_cols = [col for col in df.columns if col != 'AccessionNo']
            
            # 如果指定了排除列表，从统计中排除
            if excluded_classes:
                label_cols = [col for col in label_cols if col not in excluded_classes]
                print(f"   标签列数: {len(label_cols)} (排除占位列后)")
            else:
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

def evaluate_single_file(input_file: Path, file_index: int, total_files: int):
    """对单个输入文件进行完整的评估"""
    checkpoint_num = extract_checkpoint_number(input_file.name)
    
    print(f"\n{'='*60}")
    print(f"处理文件 [{file_index}/{total_files}]: {input_file.name}")
    print(f"Checkpoint: {checkpoint_num}")
    print(f"{'='*60}")
    print(f"⚠️  注意：排除占位类别 {', '.join(EXCLUDED_CLASSES)}")
    
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
    
    # 中间文件
    csv_pred_full = prep_dir / "inferred_full.csv"           # 完整预测结果（包含占位列）
    csv_pred = prep_dir / "inferred.csv"                     # 处理后的预测结果（排除占位列）
    gt_csv_filtered = prep_dir / "ground_truth_filtered.csv" # 处理后的GT（排除占位列）
    
    # 输出文件（评估结果）
    cls_json = prep_dir / "classification_scores.json"
    crg_json = prep_dir / "crg_scores.json"
    nlg_json = prep_dir / "nlg_scores.json"
    final_json = prep_dir / "metrics.json"
    
    print(f"使用转换后的预测文件: {pred_json_transformed.name}")
    print(f"Ground truth JSON: {gt_json.name}")
    print(f"Ground truth CSV: {gt_csv.name}")
    
    try:
        # 1. inference → CSV (先生成完整的CSV)
        print("\n1. Running inference...")
        run(INFER_SCRIPT,
            "--input_json", pred_json_transformed,
            "--model_path", CODE_DIR / "roberta_local" / "RadBertClassifier.pth",
            "--out_csv", csv_pred_full)
        print(f"✅ 推理完成，结果保存到: {csv_pred_full.name}")
        
        # 2. 移除占位列
        print("\n2. Removing placeholder columns...")
        
        # 分析原始文件
        print("\n📋 原始预测结果分析:")
        check_csv_content(csv_pred_full, excluded_classes=None)
        
        print("\n📋 原始Ground Truth分析:")
        check_csv_content(gt_csv, excluded_classes=None)
        
        # 从预测结果中移除占位列
        print(f"\n处理预测结果: {csv_pred_full.name} -> {csv_pred.name}")
        removed_pred = remove_excluded_columns(csv_pred_full, csv_pred, EXCLUDED_CLASSES)
        
        # 从ground truth中移除占位列
        print(f"处理Ground Truth: {gt_csv.name} -> {gt_csv_filtered.name}")
        removed_gt = remove_excluded_columns(gt_csv, gt_csv_filtered, EXCLUDED_CLASSES)
        
        if removed_pred > 0 or removed_gt > 0:
            print(f"✅ 成功移除 {removed_pred} 个占位列")
        else:
            print("⚠️  未找到需要移除的占位列")
        
        # 分析处理后的文件
        print("\n📋 处理后的预测结果分析:")
        check_csv_content(csv_pred, excluded_classes=None)
        
        print("\n📋 处理后的Ground Truth分析:")
        check_csv_content(gt_csv_filtered, excluded_classes=None)

        # 3. multi-label classification scores (使用处理后的CSV)
        print("\n3. Computing classification scores...")
        print("   使用处理后的CSV文件（已排除占位类别）")
        print("   注意：如果看到精确率警告，这通常是因为某些类别预测样本很少")
        run(CLS_SCRIPT,
            "--pred_csv", csv_pred,
            "--gt_csv", gt_csv_filtered,
            "--out_json", cls_json)
        print(f"✅ 分类评估完成，结果保存到: {cls_json.name}")

        # 4. CRG metrics (使用处理后的CSV)
        print("\n4. Computing CRG metrics...")
        print("   使用过滤后的CSV文件（已排除占位类别）")
        print("   使用动态列名版本的CRG脚本")
        run(CRG_SCRIPT,
            "--pred_csv", csv_pred,              # 使用 csv_pred，不是 csv_pred_filtered
            "--gt_csv", gt_csv_filtered,
            "--out_json", crg_json)
        print(f"✅ CRG评估完成，结果保存到: {crg_json.name}")

        # 5. NLG metrics (使用转换后的文件 - 这是关键修复！)
        print("\n5. Computing NLG metrics...")
        print("   🔧 修复：使用转换后的文件进行NLG评估")
        run(NLG_SCRIPT,
            "--pred_json", pred_json_transformed,
            "--gt_json", gt_json,
            "--out_json", nlg_json)
        print(f"✅ NLG评估完成，结果保存到: {nlg_json.name}")

        # 6. Combine all metrics
        print("\n6. Combining all metrics...")
        combined = {
            "generation": load(nlg_json),
            "classification": load(cls_json),
            "crg": load(crg_json),
            "metadata": {
                "checkpoint": checkpoint_num,
                "input_file": input_file.name,
                "evaluation_time": "auto-generated",
                "excluded_classes": EXCLUDED_CLASSES,
                "note": "分类和CRG指标基于排除占位类别后的结果"
            }
        }
        
        with open(final_json, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 所有评估指标已写入: {final_json}")
        
        # 打印详细结果
        print(f"\n📊 评估结果详情 (Checkpoint-{checkpoint_num}, 已排除占位类别):")
        print("=" * 50)
        
        # 分类指标
        if "classification" in combined and combined["classification"]:
            cls_metrics = combined["classification"]
            print("🔍 分类指标:")
            for key, value in cls_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.4f}")
        
        # 生成指标
        if "generation" in combined and combined["generation"]:
            gen_metrics = combined["generation"]
            print("📝 生成指标:")
            for key, value in gen_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.4f}")
        
        # CRG指标
        if "crg" in combined and combined["crg"]:
            crg_metrics = combined["crg"]
            print("🏥 CRG指标:")
            for key, value in crg_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 评估过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def generate_summary_report(successful_files: list[Path], base_dir: Path):
    """生成所有成功评估文件的汇总报告"""
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
                    "checkpoint": checkpoint_num,
                    "filename": input_file.name
                }
                
                # 分类指标
                if "classification" in metrics:
                    cls_metrics = metrics["classification"]
                    for key, value in cls_metrics.items():
                        if isinstance(value, (int, float)):
                            summary_item[f"cls_{key}"] = round(value, 4)
                
                # 生成指标  
                if "generation" in metrics:
                    gen_metrics = metrics["generation"]
                    for key, value in gen_metrics.items():
                        if isinstance(value, (int, float)):
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
        print(f"\n📈 性能对比表格 (前3个关键指标):")
        print("-" * 80)
        print(f"{'Checkpoint':<12} {'文件名':<25} {'样例指标1':<12} {'样例指标2':<12} {'样例指标3':<12}")
        print("-" * 80)
        
        for item in sorted(summary_data, key=lambda x: x['checkpoint']):
            checkpoint = item['checkpoint']
            filename = item['filename'][:22] + "..." if len(item['filename']) > 25 else item['filename']
            
            # 选择前3个数值指标进行展示
            numeric_keys = [k for k, v in item.items() 
                          if k not in ['checkpoint', 'filename'] and isinstance(v, (int, float))][:3]
            
            values = [f"{item[k]:.3f}" if k in item else "N/A" for k in numeric_keys]
            while len(values) < 3:
                values.append("N/A")
            
            print(f"{checkpoint:<12} {filename:<25} {values[0]:<12} {values[1]:<12} {values[2]:<12}")
        
        print("-" * 80)
        print("📝 完整数据请查看: evaluation_summary.json")
        print(f"⚠️  注意：分类和CRG指标均已排除 {', '.join(EXCLUDED_CLASSES)}")

def main():
    """主函数，处理所有输入文件"""
    print("🚀 主评估脚本 - 增强版 (支持文件夹模式 + 占位类别过滤)")
    print(f"基础目录: {BASE_DIR}")
    print(f"评估脚本目录: {CODE_DIR}")
    print(f"⚠️  排除的占位类别: {', '.join(EXCLUDED_CLASSES)}")
    
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
        print(f"   {i}. {f.name} (checkpoint-{checkpoint_num})")
    
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
        generate_summary_report(successful_files, base_dir=BASE_DIR)
    
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
    
    # 使用提示
    if successful > 0:
        print("\n💡 结果文件说明:")
        print("   📄 inferred_full.csv - 模型预测的完整分类结果（包含占位列）")
        print("   📄 inferred.csv - 处理后的预测结果（已排除占位列）")
        print("   📄 ground_truth_filtered.csv - 处理后的GT（已排除占位列）")
        print("   📄 classification_scores.json - 详细的分类指标（基于排除占位列后的数据）")
        print("   📄 crg_scores.json - CRG指标（基于排除占位列后的数据）")
        print("   📄 nlg_scores.json - NLG指标")
        print("   📄 metrics.json - 最终合并的评估结果")
        print("   📄 evaluation_summary.json - 所有文件的汇总对比")

if __name__ == "__main__":
    main()