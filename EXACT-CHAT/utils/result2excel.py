
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估结果表格整理脚本
将多个 metrics.json 文件的结果整理成表格格式
"""

import json
import pandas as pd
from pathlib import Path
import re
from typing import Dict, List, Any

# 配置路径
RESULTS_BASE_DIR = Path("/path/to/CT-CHAT2/llava/train/output")
OUTPUT_CSV = RESULTS_BASE_DIR / "evaluation_summary.csv"
OUTPUT_EXCEL = RESULTS_BASE_DIR / "evaluation_summary.xlsx"

def extract_experiment_name(dir_path: Path) -> str:
    """从目录名提取实验名称，进行简化处理"""
    name = dir_path.name
    
    # 移除常见的前缀
    name = re.sub(r'^predictions_', '', name)
    
    # 提取关键参数
    # 匹配checkpoint数字
    checkpoint_match = re.search(r'checkpoint(\d+)', name)
    checkpoint_num = checkpoint_match.group(1) if checkpoint_match else "unknown"
    
    # 匹配温度参数
    temp_match = re.search(r'temp([\d\.]+)', name)
    temperature = temp_match.group(1) if temp_match else "unknown"
    
    # 匹配token数量
    token_match = re.search(r'tokens(\d+)', name)
    tokens = token_match.group(1) if token_match else "unknown"
    
    # 提取日期时间（如果存在）
    datetime_match = re.search(r'(\d{8}_\d{6})', name)
    datetime_str = datetime_match.group(1) if datetime_match else ""
    
    # 构建简化名称
    simplified_name = f"ckpt{checkpoint_num}_temp{temperature}_tok{tokens}"
    if datetime_str:
        simplified_name += f"_{datetime_str}"
    
    return simplified_name

def load_metrics(metrics_file: Path) -> Dict[str, Any]:
    """加载metrics.json文件"""
    try:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 无法加载 {metrics_file}: {e}")
        return {}

def extract_generation_metrics(data: Dict) -> Dict[str, float]:
    """提取生成指标"""
    gen_data = data.get('generation', {})
    return {
        'BLEU_1': gen_data.get('BLEU_1', 0.0),
        'BLEU_2': gen_data.get('BLEU_2', 0.0),
        'BLEU_3': gen_data.get('BLEU_3', 0.0),
        'BLEU_4': gen_data.get('BLEU_4', 0.0),
        'BLEU_mean': gen_data.get('BLEU_mean', 0.0),
        'ROUGE_L': gen_data.get('ROUGE_L', 0.0),
        'CIDEr': gen_data.get('CIDEr', 0.0),
        'METEOR': gen_data.get('METEOR', 0.0)
    }

def extract_classification_macro_metrics(data: Dict) -> Dict[str, float]:
    """提取分类宏指标"""
    cls_data = data.get('classification', {})
    macro_data = cls_data.get('macro', {})
    return {
        'cls_macro_precision': macro_data.get('precision', 0.0),
        'cls_macro_recall': macro_data.get('recall', 0.0),
        'cls_macro_f1': macro_data.get('f1', 0.0),
        'cls_macro_accuracy': macro_data.get('accuracy', 0.0)
    }

def extract_crg_metrics(data: Dict) -> Dict[str, float]:
    """提取CRG指标"""
    crg_data = data.get('crg', {})
    return {
        'CRG_score': crg_data.get('CRG', 0.0),
        'CRG_TP': crg_data.get('TP', 0),
        'CRG_FN': crg_data.get('FN', 0),
        'CRG_FP': crg_data.get('FP', 0),
        'CRG_X': crg_data.get('X', 0),
        'CRG_A': crg_data.get('A', 0),
        'CRG_r': crg_data.get('r', 0.0),
        'CRG_U': crg_data.get('U', 0.0),
        'CRG_score_s': crg_data.get('score_s', 0.0)
    }

def extract_per_pathology_metrics(data: Dict) -> Dict[str, float]:
    """提取每个疾病的详细指标"""
    cls_data = data.get('classification', {})
    per_path_data = cls_data.get('per_pathology', [])
    
    metrics = {}
    
    for pathology in per_path_data:
        name = pathology.get('name', 'unknown')
        # 清理疾病名称，使其适合作为列名
        clean_name = name.replace(' ', '_').replace('-', '_')
        
        metrics.update({
            f'{clean_name}_precision': pathology.get('precision', 0.0),
            f'{clean_name}_recall': pathology.get('recall', 0.0),
            f'{clean_name}_f1': pathology.get('f1', 0.0),
            f'{clean_name}_accuracy': pathology.get('accuracy', 0.0)
        })
    
    return metrics

def find_all_metrics_files(base_dir: Path) -> List[Path]:
    """找到所有的metrics.json文件"""
    metrics_files = []
    
    for subdir in base_dir.iterdir():
        if subdir.is_dir():
            metrics_file = subdir / "metrics.json"
            if metrics_file.exists():
                metrics_files.append(metrics_file)
    
    return sorted(metrics_files)

def create_summary_table(metrics_files: List[Path]) -> pd.DataFrame:
    """创建汇总表格"""
    all_rows = []
    
    print(f"🔍 找到 {len(metrics_files)} 个metrics.json文件")
    
    for metrics_file in metrics_files:
        print(f"处理: {metrics_file.parent.name}")
        
        # 提取实验名称
        exp_name = extract_experiment_name(metrics_file.parent)
        
        # 加载数据
        data = load_metrics(metrics_file)
        if not data:
            continue
        
        # 构建行数据
        row_data = {'experiment': exp_name}
        
        # 添加生成指标
        row_data.update(extract_generation_metrics(data))
        
        # 添加分类宏指标
        row_data.update(extract_classification_macro_metrics(data))
        
        # 添加CRG指标
        row_data.update(extract_crg_metrics(data))
        
        # 添加每个疾病的详细指标
        row_data.update(extract_per_pathology_metrics(data))
        
        all_rows.append(row_data)
    
    if not all_rows:
        print("❌ 没有找到有效的数据")
        return pd.DataFrame()
    
    # 创建DataFrame
    df = pd.DataFrame(all_rows)
    
    # 重新排列列顺序：实验名称 -> 平均指标 -> 详细指标
    cols_order = ['experiment']
    
    # 生成指标
    gen_cols = ['BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4', 'BLEU_mean', 'ROUGE_L', 'CIDEr', 'METEOR']
    cols_order.extend(gen_cols)
    
    # 分类宏指标
    cls_macro_cols = ['cls_macro_precision', 'cls_macro_recall', 'cls_macro_f1', 'cls_macro_accuracy']
    cols_order.extend(cls_macro_cols)
    
    # CRG指标
    crg_cols = ['CRG_score', 'CRG_TP', 'CRG_FN', 'CRG_FP', 'CRG_X', 'CRG_A', 'CRG_r', 'CRG_U', 'CRG_score_s']
    cols_order.extend(crg_cols)
    
    # 剩余的列（疾病详细指标）
    remaining_cols = [col for col in df.columns if col not in cols_order]
    cols_order.extend(sorted(remaining_cols))
    
    # 按新顺序重排列
    df = df.reindex(columns=cols_order)
    
    return df

def format_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """格式化DataFrame用于显示"""
    if df.empty:
        return df
    
    # 复制DataFrame避免修改原始数据
    display_df = df.copy()
    
    # 格式化数值列（保留4位小数）
    numeric_cols = display_df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if col.startswith(('BLEU', 'ROUGE', 'CIDEr', 'METEOR', 'cls_', 'CRG_score', 'CRG_r')):
            display_df[col] = display_df[col].round(4)
    
    return display_df

def save_results(df: pd.DataFrame):
    """保存结果到文件"""
    if df.empty:
        print("❌ 没有数据可保存")
        return
    
    try:
        # 保存CSV
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
        print(f"✅ CSV结果已保存到: {OUTPUT_CSV}")
        
        # 保存Excel（带格式）
        with pd.ExcelWriter(OUTPUT_EXCEL, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Evaluation_Summary', index=False)
            
            # 获取工作表以设置格式
            worksheet = writer.sheets['Evaluation_Summary']
            
            # 调整列宽
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)  # 最大50字符宽度
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"✅ Excel结果已保存到: {OUTPUT_EXCEL}")
        
    except Exception as e:
        print(f"❌ 保存文件时出错: {e}")

def print_summary_statistics(df: pd.DataFrame):
    """打印汇总统计信息"""
    if df.empty:
        return
    
    print(f"\n📊 数据汇总统计:")
    print(f"   实验数量: {len(df)}")
    print(f"   指标数量: {len(df.columns) - 1}")  # 减去experiment列
    
    # 显示前几个关键指标的统计
    key_metrics = ['BLEU_mean', 'ROUGE_L', 'cls_macro_f1', 'CRG_score']
    existing_metrics = [m for m in key_metrics if m in df.columns]
    
    if existing_metrics:
        print(f"\n🎯 关键指标统计:")
        stats_df = df[existing_metrics].describe()
        print(stats_df.round(4))
    
    # 显示最佳表现的实验
    if 'BLEU_mean' in df.columns:
        best_bleu_idx = df['BLEU_mean'].idxmax()
        best_exp = df.loc[best_bleu_idx, 'experiment']
        best_bleu = df.loc[best_bleu_idx, 'BLEU_mean']
        print(f"\n🏆 最佳BLEU_mean: {best_exp} ({best_bleu:.4f})")
    
    if 'CRG_score' in df.columns:
        best_crg_idx = df['CRG_score'].idxmax()
        best_exp = df.loc[best_crg_idx, 'experiment']
        best_crg = df.loc[best_crg_idx, 'CRG_score']
        print(f"🏆 最佳CRG_score: {best_exp} ({best_crg:.4f})")

def main():
    """主函数"""
    print("🚀 评估结果表格整理脚本")
    print(f"📁 扫描目录: {RESULTS_BASE_DIR}")
    
    # 检查基础目录是否存在
    if not RESULTS_BASE_DIR.exists():
        print(f"❌ 目录不存在: {RESULTS_BASE_DIR}")
        return
    
    # 查找所有metrics.json文件
    metrics_files = find_all_metrics_files(RESULTS_BASE_DIR)
    
    if not metrics_files:
        print("❌ 未找到任何metrics.json文件")
        return
    
    # 创建汇总表格
    df = create_summary_table(metrics_files)
    
    if df.empty:
        print("❌ 无法创建表格")
        return
    
    # 格式化用于显示
    display_df = format_dataframe_for_display(df)
    
    # 显示结果（前几行）
    print(f"\n📋 评估结果预览（前5行）:")
    print("=" * 100)
    
    # 只显示前几个关键列用于预览
    preview_cols = ['experiment'] + [col for col in df.columns if col in [
        'BLEU_mean', 'ROUGE_L', 'CIDEr', 'METEOR', 
        'cls_macro_precision', 'cls_macro_recall', 'cls_macro_f1', 
        'CRG_score'
    ]]
    
    if len(display_df) > 5:
        print(display_df[preview_cols].head().to_string(index=False))
    else:
        print(display_df[preview_cols].to_string(index=False))
    
    # 保存结果
    save_results(df)
    
    # 打印统计信息
    print_summary_statistics(df)
    
    print(f"\n✅ 处理完成！")
    print(f"📊 完整结果请查看:")
    print(f"   CSV: {OUTPUT_CSV}")
    print(f"   Excel: {OUTPUT_EXCEL}")

if __name__ == "__main__":
    main()
