
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备评估所需文件 - 增强版本（支持文件夹模式）：
为每个输入文件分别准备：
1) 将 results.json 转为 infer.py 期望的结构 -> result_transformat.json
2) 从 valid_vqa.json 中抽取参考报告 -> ground_truth.json
3) 从 valid_predicted_labels.csv 中抽取18类标签 -> ground_truth.csv
"""

from pathlib import Path
import json
import pandas as pd
import re

# === 配置选项 ===
# 选择运行模式：True=文件夹模式，False=文件列表模式
USE_FOLDER_MODE = False

# 文件夹模式配置
FOLDER_PATH = Path("/path/to/CT-CHAT2/llava/train/parameter_tuning_results")
FILE_PATTERN = "predictions_checkpoint*.json"  # 文件匹配模式

# 手动文件列表模式配置（保留原有功能）
MANUAL_INPUT_FILES = [
    Path("/path/to/CT-CHAT/output_validation_vicuna_24000_report_generation.json"),
    # Path("/path/to/CT-CHAT2/llava/train/parameter_tuning_results/predictions_checkpoint34000_temp0.0_tokens1024_checkpoint-34000_20250916_193006.json"),
    # Path("/path/to/CT-CHAT2/llava/train/parameter_tuning_results/predictions_checkpoint36000_temp0.0_tokens1024_checkpoint-36000_20250917_010514.json"),
    # Path("/path/to/CT-CHAT2/llava/train/parameter_tuning_results/predictions_checkpoint38000_temp0.0_tokens1024_checkpoint-38000_20250917_051357.json"),
    # Path("/path/to/CT-CHAT2/llava/train/parameter_tuning_results/predictions_checkpoint40000_temp0.0_tokens1024_checkpoint-40000_20250917_091830.json"),
    # Path("/path/to/CT-CHAT2/llava/train/parameter_tuning_results/predictions_checkpoint42000_temp0.0_tokens1024_checkpoint-42000_20250917_161016.json"),
    # Path("/path/to/CT-CHAT2/llava/train/parameter_tuning_results/predictions_checkpoint44000_temp0.0_tokens1024_checkpoint-44000_20250917_223953.json"),
    # Path("/path/to/CT-CHAT2/llava/train/parameter_tuning_results/predictions_checkpoint46000_temp0.0_tokens1024_checkpoint-46000_20250918_045152.json")
]

# 输出基础目录 - 与主评估脚本保持一致
OUTPUT_BASE = Path("/path/to/CT-CHAT2/llava/train/output")

# 参考数据文件路径
VQA_JSON = Path("/path/to/CT-CHAT2/VQA_dataset/by_category/report.json")
CLS_CSV  = Path("/path/to/CT_Report/CT_submission_abnclass/evalution/valid_predicted_labels.csv")

# 18 类标签列（需与评估脚本保持一致）
LABEL_COLS = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification",
    "Hiatal hernia", "Lymphadenopathy", "Emphysema", "Atelectasis",
    "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela",
    "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation",
    "Bronchiectasis", "Interlobular septal thickening"
]

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
    """根据json文件名生成输出目录名 - 与主评估脚本保持一致"""
    name = json_file_path.stem
    return name.replace(" ", "_")

def load_results_json(p: Path):
    """加载结果JSON文件，支持多种格式"""
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    items = None
    
    # 情况1: {"generated_reports": [...]}
    if isinstance(data, dict) and "generated_reports" in data:
        items = data["generated_reports"]
        print(f"[INFO] 检测到格式1: generated_reports 结构")
    
    # 情况2: 直接是列表，包含 input_image_name 和 report
    elif isinstance(data, list) and len(data) > 0:
        first_item = data[0]
        
        # 子情况2a: 对话格式 [{"image": ..., "conversations_out": [{"question": ..., "answer": ...}]}]
        if isinstance(first_item, dict) and "conversations_out" in first_item:
            print(f"[INFO] 检测到格式2a: 对话格式结构")
            items = []
            for item in data:
                if "image" in item and "conversations_out" in item:
                    image_name = item["image"]
                    conversations = item["conversations_out"]
                    
                    # 提取answer作为报告内容
                    for conv in conversations:
                        if isinstance(conv, dict) and "answer" in conv:
                            answer = conv["answer"].strip()
                            # 清理可能的特殊标记
                            if answer.endswith("<|eot_id|>"):
                                answer = answer[:-10].strip()
                            
                            items.append({
                                "input_image_name": image_name,
                                "report": answer
                            })
                            break  # 只取第一个answer
        
        # 子情况2b: 标准格式 [{"input_image_name": ..., "report": ...}]
        elif isinstance(first_item, dict) and ("input_image_name" in first_item or "report" in first_item):
            print(f"[INFO] 检测到格式2b: 标准列表结构")
            items = data
        
        else:
            print(f"[WARN] 列表格式无法识别，尝试通用解析...")
            items = data
    
    # 情况3: 其他结构
    else:
        print(f"[WARN] 未识别的文件结构，尝试自动解析")
        raise ValueError(f"{p} 的结构无法识别。请检查文件格式。")
    
    if not isinstance(items, list):
        raise ValueError("无法找到报告列表。")
    
    # 规范化，确保有 input_image_name 与 report
    norm = []
    for it in items:
        if isinstance(it, dict):
            imgname = (it.get("input_image_name") or 
                      it.get("image") or 
                      it.get("image_name") or 
                      it.get("image_id"))
            
            report = (it.get("report") or 
                     it.get("generated_report") or 
                     it.get("text") or 
                     it.get("answer"))
            
            if imgname and report:
                # 清理图像名称，移除可能的路径和扩展名问题
                imgname = str(imgname).strip()
                report = str(report).strip()
                
                # 清理报告中的特殊标记
                if report.endswith("<|eot_id|>"):
                    report = report[:-10].strip()
                
                norm.append({
                    "input_image_name": imgname, 
                    "report": report
                })
    
    if not norm:
        raise ValueError(f"在 {p.name} 中未找到有效的报告数据。请检查字段名称。")
    
    print(f"[INFO] 从 {p.name} 中加载了 {len(norm)} 条报告")
    
    # 显示前几个样本用于调试
    print("[DEBUG] 前3个样本:")
    for i, item in enumerate(norm[:3]):
        print(f"  {i+1}. 图像: {item['input_image_name']}")
        print(f"     报告: {item['report'][:100]}...")
    
    return norm

def write_transformed_for_infer(items, out_path: Path):
    """
    写出 infer.py 期望的格式
    """
    wrapped = [
        {
            "outputs": [
                {
                    "value": {
                        "generated_reports": items
                    }
                }
            ]
        }
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(wrapped, f, ensure_ascii=False, indent=2)

def load_vqa_index(vqa_path: Path):
    """
    构建 image -> 参考报告 文本的索引
    """
    if not vqa_path.exists():
        print(f"[WARN] VQA文件不存在: {vqa_path}")
        return {}
    
    with open(vqa_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    idx = {}
    
    # 处理不同的VQA文件格式
    if isinstance(data, list):
        for entry in data:
            img = entry.get("image")
            conv = entry.get("conversations", [])
            if not img or not isinstance(conv, list):
                continue

            ref_text = None

            # 优先：找到 human.type == 'report_generation' 的轮次，取其后第一个 gpt 的 value
            for i, turn in enumerate(conv):
                if isinstance(turn, dict) and turn.get("from") == "human":
                    t = turn.get("type") or turn.get("conversation_type") or ""
                    if str(t).strip().lower() == "report_generation":
                        if i + 1 < len(conv) and isinstance(conv[i + 1], dict) and conv[i + 1].get("from") == "gpt":
                            ref_text = conv[i + 1].get("value")
                            break

            # 兜底：第一个 gpt 的 value
            if not ref_text:
                for turn in conv:
                    if isinstance(turn, dict) and turn.get("from") == "gpt":
                        ref_text = turn.get("value")
                        break

            if ref_text:
                idx[img] = ref_text
    
    elif isinstance(data, dict):
        # 如果是字典格式，可能有其他结构
        print("[INFO] VQA文件是字典格式，尝试解析...")
        # 这里可以添加其他格式的解析逻辑
    
    print(f"[INFO] 从VQA数据中加载了 {len(idx)} 条参考报告")
    return idx

def build_ground_truth_json(items, vqa_index: dict, out_path: Path):
    """
    生成 nlg_metrics.py 期望的 GT 结构
    """
    gt_items = []
    miss = []

    for it in items:
        imgname = it["input_image_name"]
        stem = Path(imgname).stem
        
        # 扩展候选匹配名称
        candidates = [
            imgname,                    # 原始名称
            f"{imgname}.nii.gz",       # 加.nii.gz
            f"{stem}.nii.gz",          # stem加.nii.gz
            f"{stem}.npz",             # stem加.npz (根据你的数据)
            stem                        # 仅stem
        ]

        ref = None
        matched_key = None
        for key in candidates:
            if key in vqa_index:
                ref = vqa_index[key]
                matched_key = key
                break

        if ref:
            gt_items.append({"input_image_name": stem, "report": ref})
        else:
            miss.append(imgname)

    payload = {"generated_reports": gt_items}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 生成了 {len(gt_items)} 条参考报告到 ground_truth.json")
    
    if miss:
        print(f"[WARN] {len(miss)} 个样本在VQA中未找到参考报告")
        if len(miss) <= 5:
            for m in miss:
                print(f"  - {m}")
        else:
            for m in miss[:5]:
                print(f"  - {m}")
            print(f"  ... 还有 {len(miss)-5} 个")

def build_ground_truth_csv(items, cls_csv_path: Path, out_path: Path):
    """
    从分类标签CSV中抽取对应行
    """
    if not cls_csv_path.exists():
        print(f"[WARN] 分类标签文件不存在: {cls_csv_path}")
        # 创建空的CSV文件
        df_out = pd.DataFrame(columns=["AccessionNo"] + LABEL_COLS)
        df_out.to_csv(out_path, index=False)
        return

    df_all = pd.read_csv(cls_csv_path)
    if "VolumeName" not in df_all.columns:
        print(f"[WARN] {cls_csv_path} 缺少列 'VolumeName'，创建空CSV")
        df_out = pd.DataFrame(columns=["AccessionNo"] + LABEL_COLS)
        df_out.to_csv(out_path, index=False)
        return

    vol_to_row = {str(row["VolumeName"]): row for _, row in df_all.iterrows()}

    rows_out = []
    miss = []

    for it in items:
        imgname = it["input_image_name"]
        stem = Path(imgname).stem
        
        # 扩展候选匹配名称
        candidates = [
            imgname,                    # 原始名称
            f"{imgname}.nii.gz",       # 加.nii.gz
            f"{stem}.nii.gz",          # stem加.nii.gz
            f"{stem}.npz",             # stem加.npz
            stem                        # 仅stem
        ]

        row = None
        for key in candidates:
            if key in vol_to_row:
                row = vol_to_row[key]
                break
        
        # 尝试按stem匹配所有键
        if row is None:
            keys2 = {Path(k).stem: k for k in vol_to_row.keys()}
            if stem in keys2:
                row = vol_to_row[keys2[stem]]

        if row is None:
            miss.append(imgname)
            # 创建默认行（全0标签）
            out_row = {"AccessionNo": stem}
            for col in LABEL_COLS:
                out_row[col] = 0
            rows_out.append(out_row)
            continue

        out_row = {"AccessionNo": stem}
        for col in LABEL_COLS:
            if col in row:
                out_row[col] = row[col]
            else:
                out_row[col] = 0  # 默认值
        rows_out.append(out_row)

    if rows_out:
        df_out = pd.DataFrame(rows_out, columns=["AccessionNo"] + LABEL_COLS)
        df_out.to_csv(out_path, index=False)
        print(f"[INFO] 生成了 {len(rows_out)} 行分类标签到 ground_truth.csv")
    else:
        df_out = pd.DataFrame(columns=["AccessionNo"] + LABEL_COLS)
        df_out.to_csv(out_path, index=False)
        print("[INFO] 创建了空的 ground_truth.csv")

    if miss:
        print(f"[WARN] {len(miss)} 个样本在分类数据中未找到，使用默认标签")

def process_single_file(input_file: Path, file_index: int, total_files: int):
    """处理单个输入文件"""
    checkpoint_num = extract_checkpoint_number(input_file.name)
    
    print(f"\n{'='*60}")
    print(f"准备文件 [{file_index}/{total_files}]: {input_file.name}")
    print(f"Checkpoint: {checkpoint_num}")
    print(f"{'='*60}")
    
    # 创建对应的输出目录
    output_dir_name = get_output_dir_name(input_file)
    output_dir = OUTPUT_BASE / output_dir_name
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Output directory: {output_dir}")
    
    # 定义输出文件路径
    out_transformed = output_dir / "result_transformat.json"
    out_gt_json     = output_dir / "ground_truth.json"
    out_gt_csv      = output_dir / "ground_truth.csv"
    
    try:
        # 1. 加载结果文件
        items = load_results_json(input_file)
        
        # 2. 生成transformed格式
        write_transformed_for_infer(items, out_transformed)
        print(f"[INFO] ✅ 生成: {out_transformed.name}")
        
        # 3. 加载VQA索引
        vqa_idx = load_vqa_index(VQA_JSON) if VQA_JSON.exists() else {}
        
        # 4. 生成ground truth JSON
        build_ground_truth_json(items, vqa_idx, out_gt_json)
        print(f"[INFO] ✅ 生成: {out_gt_json.name}")
        
        # 5. 生成ground truth CSV
        build_ground_truth_csv(items, CLS_CSV, out_gt_csv)
        print(f"[INFO] ✅ 生成: {out_gt_csv.name}")
        
        # 简要统计信息
        print(f"📊 处理统计 (Checkpoint-{checkpoint_num}):")
        print(f"   报告数量: {len(items)}")
        if out_gt_json.exists():
            with open(out_gt_json, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
                print(f"   参考报告: {len(gt_data.get('generated_reports', []))}")
        if out_gt_csv.exists():
            df_csv = pd.read_csv(out_gt_csv)
            print(f"   分类标签: {len(df_csv)} 行")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] ❌ 处理 {input_file.name} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def generate_preparation_summary(successful_files: list[Path], base_dir: Path):
    """生成准备工作的汇总报告"""
    if not successful_files:
        return
    
    print(f"\n{'='*60}")
    print("📋 生成准备工作汇总报告")
    print(f"{'='*60}")
    
    summary_data = []
    
    for input_file in successful_files:
        output_dir_name = get_output_dir_name(input_file)
        output_dir = base_dir / output_dir_name
        checkpoint_num = extract_checkpoint_number(input_file.name)
        
        summary_item = {
            "checkpoint": checkpoint_num,
            "filename": input_file.name,
            "output_directory": str(output_dir),
            "files_prepared": []
        }
        
        # 检查生成的文件
        expected_files = [
            "result_transformat.json",
            "ground_truth.json", 
            "ground_truth.csv"
        ]
        
        for fname in expected_files:
            fpath = output_dir / fname
            if fpath.exists():
                summary_item["files_prepared"].append(fname)
        
        # 统计数据量
        try:
            transform_file = output_dir / "result_transformat.json"
            if transform_file.exists():
                with open(transform_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    reports = data[0]["outputs"][0]["value"]["generated_reports"]
                    summary_item["report_count"] = len(reports)
            
            gt_json_file = output_dir / "ground_truth.json"
            if gt_json_file.exists():
                with open(gt_json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    summary_item["reference_count"] = len(data.get("generated_reports", []))
                    
            gt_csv_file = output_dir / "ground_truth.csv"
            if gt_csv_file.exists():
                df = pd.read_csv(gt_csv_file)
                summary_item["label_count"] = len(df)
                
        except Exception as e:
            print(f"⚠️  统计 {input_file.name} 数据时出错: {e}")
        
        summary_data.append(summary_item)
    
    # 保存汇总报告
    if summary_data:
        summary_file = base_dir / "preparation_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"📊 准备工作汇总报告已保存: {summary_file}")
        
        # 显示准备情况表格
        print(f"\n📈 准备情况总览:")
        print("-" * 90)
        print(f"{'Checkpoint':<12} {'文件名':<25} {'报告数':<8} {'参考数':<8} {'标签数':<8} {'文件状态':<15}")
        print("-" * 90)
        
        for item in sorted(summary_data, key=lambda x: x['checkpoint']):
            checkpoint = item['checkpoint']
            filename = item['filename'][:22] + "..." if len(item['filename']) > 25 else item['filename']
            report_count = item.get('report_count', 'N/A')
            ref_count = item.get('reference_count', 'N/A')
            label_count = item.get('label_count', 'N/A')
            files_prepared = len(item.get('files_prepared', []))
            
            status = f"{files_prepared}/3 完成" if files_prepared == 3 else f"{files_prepared}/3 ⚠️"
            
            print(f"{checkpoint:<12} {filename:<25} {report_count:<8} {ref_count:<8} {label_count:<8} {status:<15}")
        
        print("-" * 90)
        print("📝 完整数据请查看: preparation_summary.json")

def main():
    """主函数"""
    print("🚀 准备评估文件 - 增强版本（支持文件夹模式）")
    print(f"输出基础目录: {OUTPUT_BASE}")
    print(f"VQA参考文件: {VQA_JSON}")
    print(f"分类标签文件: {CLS_CSV}")
    
    # 显示当前配置
    mode_text = "文件夹模式" if USE_FOLDER_MODE else "手动文件列表模式"
    print(f"🔧 当前模式: {mode_text}")
    
    if USE_FOLDER_MODE:
        print(f"📁 搜索文件夹: {FOLDER_PATH}")
        print(f"🔍 文件模式: {FILE_PATTERN}")
    
    # 创建输出基础目录
    OUTPUT_BASE.mkdir(exist_ok=True, parents=True)
    
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
    
    print(f"\n📁 将准备 {len(input_files)} 个输入文件:")
    for i, f in enumerate(input_files, 1):
        checkpoint_num = extract_checkpoint_number(f.name)
        print(f"   {i}. {f.name} (checkpoint-{checkpoint_num})")
    
    # 检查参考文件状态
    print(f"\n📋 参考文件状态检查:")
    print(f"   VQA文件: {'✅' if VQA_JSON.exists() else '❌'} {VQA_JSON}")
    print(f"   分类文件: {'✅' if CLS_CSV.exists() else '❌'} {CLS_CSV}")
    
    # 处理每个文件
    successful = 0
    failed = 0
    successful_files = []
    
    for i, input_file in enumerate(input_files, 1):
        if process_single_file(input_file, i, len(input_files)):
            successful += 1
            successful_files.append(input_file)
        else:
            failed += 1
        
        # 处理过程中显示进度
        if i < len(input_files):
            print(f"\n⏭️  准备处理下一个文件... ({i}/{len(input_files)} 完成)")
    
    # 生成汇总报告
    if successful_files:
        generate_preparation_summary(successful_files, OUTPUT_BASE)
    
    # 最终总结
    print(f"\n{'='*60}")
    print("🎯 准备工作完成总结")
    print(f"{'='*60}")
    print(f"总文件数: {len(input_files)}")
    print(f"✅ 成功: {successful}")
    print(f"❌ 失败: {failed}")
    print(f"📊 成功率: {successful/len(input_files)*100:.1f}%" if input_files else "N/A")
    
    if successful > 0:
        print(f"\n🎉 准备文件已生成在: {OUTPUT_BASE}")
        print("   现在可以运行主评估脚本了。")
        print(f"\n📋 生成的文件类型:")
        print("   🔄 result_transformat.json - 转换格式的预测结果")
        print("   📝 ground_truth.json - NLG评估参考报告")  
        print("   📊 ground_truth.csv - 分类评估参考标签")
    
    # 提供使用建议
    if successful > 0:
        print(f"\n💡 使用建议:")
        print("1. 检查准备文件的数据量是否合理")
        print("2. 确保主评估脚本使用相同的文件夹配置")
        print("3. 如有数据不匹配警告，检查参考文件路径")
        
    if failed > 0:
        print(f"\n🔧 问题诊断建议:")
        print("1. 检查输入文件格式是否正确")
        print("2. 确保参考数据文件存在且可访问") 
        print("3. 检查文件权限和磁盘空间")

if __name__ == "__main__":
    main()
