# /FM_data/bxg/CT_Report/CT_Report9_test/threshold.py
import os
import argparse
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm

DISEASES_18 = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
    "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening"
]
NAME2IDX = {n: i for i, n in enumerate(DISEASES_18)}

def parse_args():
    p = argparse.ArgumentParser(description="Select voxels by combining relative (top-k%) and absolute threshold")
    p.add_argument("--root",
                   default="/FM_data/bxg/CT_Report/CT_Report9_test/results/segmamba__Monday_13_October_2025_14h_28m_40s/test_results",
                   help="包含 prediction_heatmaps/ 的根目录，如: <work_dir>/test_results 或 <work_dir>/validation_results")
    p.add_argument("--epoch", type=int, default=10,
                   help="指定 epoch 编号；不指定则处理所有 epoch_*")
    p.add_argument("--res", choices=["high-res", "low-res"], default="low-res",
                   help="选择热图分辨率文件后缀")
    p.add_argument("--no-combined", action="store_true",
                   help="使用原始 *_pred.nii.gz，而不是 *_combined_pred.nii.gz")
    p.add_argument("--ratio", type=float, default=0.8,
                   help="相对阈值：取前 ratio 比例的体素，例如0.20表示前20%%")
    p.add_argument("--threshold-file", type=str,
                   default="/FM_data/bxg/CT_Report/CT_Report15_18abn_2decoder/results/segmamba__Thursday_18_September_2025_21h_01m_13s/best_thresholds_epoch_10.npy",
                   help="分类最佳阈值文件路径 (.npy)")
    p.add_argument("--threshold-scale", type=float, default=0.0001,
                   help="绝对阈值缩放因子，默认0.1表示best_threshold的十分之一")
    p.add_argument("--csv-file", type=str,
                   default="/FM_data/bxg/CT_Report/CT_Report9_test/results/segmamba__Monday_13_October_2025_14h_28m_40s/test_results/sample_predictions_epoch_10.csv",
                   help="样本预测结果CSV文件路径")
    p.add_argument("--overwrite", action="store_true", help="已存在则覆盖")
    return p.parse_args()

def load_predictions_csv(csv_path):
    """
    读取CSV文件，返回字典: {sample_name: {disease_name: pred_value}}
    """
    df = pd.read_csv(csv_path)
    predictions = {}
    
    for _, row in df.iterrows():
        sample_name = row['Sample_Name']
        sample_preds = {}
        
        for disease in DISEASES_18:
            pred_col = f"{disease}_Pred"
            if pred_col in row:
                sample_preds[disease] = int(row[pred_col])
            else:
                sample_preds[disease] = 1  # 默认为1，使用正常分割逻辑
        
        predictions[sample_name] = sample_preds
    
    return predictions

def disease_from_filename(fname):
    # 文件名形如: "{Disease Name}_GT0_PD1_high-res_combined_pred.nii.gz"
    base = os.path.basename(fname)
    if "_GT" in base:
        disease = base.split("_GT")[0]
    else:
        disease = base.rsplit("_", 4)[0]
    return disease

def iter_epoch_dirs(root, epoch):
    hm_root = os.path.join(root, "prediction_heatmaps")
    if not os.path.isdir(hm_root):
        raise FileNotFoundError(f"未找到目录: {hm_root}")
    if epoch is not None:
        dirs = [os.path.join(hm_root, f"epoch_{epoch}")]
    else:
        dirs = sorted([d for d in glob.glob(os.path.join(hm_root, "epoch_*")) if os.path.isdir(d)])
    return dirs

def topk_threshold(data: np.ndarray, ratio: float) -> float:
    """返回使得约 ratio 比例体素落入"最高值集合"的阈值（相对阈值）"""
    flat = data.reshape(-1)
    # 仅保留有限值
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return np.inf  # 使得生成全零掩码
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmax == vmin:
        return np.inf  # 常数图 -> 输出全零
    
    # 只在非零值中计算top-k
    nonzero = finite[finite > 0]
    if nonzero.size == 0:
        return np.inf
    
    k = max(1, int(np.ceil(ratio * nonzero.size)))
    # 取前k大的最小值作为阈值
    idx = np.argpartition(nonzero, -k)[-k:]
    thr = float(np.min(nonzero[idx]))
    return thr
import re
# 输入（预测结果：prediction/epoch_X 里的各样本子目录）
BASE_DIR = "/FM_data/bxg/CT_Report/CT_Report9_test/results/segmamba__Monday_13_October_2025_14h_28m_40s/test_results/prediction/epoch_10"
# 输出（保存分割 .nii.gz）
OUT_DIR  = "/FM_data/bxg/CT_Report/CT_Report9_test/segmentation_results"

BINARIZE_THRESHOLD = 0.5  # 读取时再次二值化阈值（安全）

# 18 个疾病名称
disease_names = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
    "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening"
]

# 只保留 (lung + trachea and bronchie + pleura) 相关疾病
target_indices = {1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}
target_diseases = {disease_names[i] for i in target_indices}

# 匹配病变掩码文件
pattern = re.compile(r"^(?P<disease>.+?)_GT\d+_PD\d+_low-res_lesion_mask\.nii\.gz$")

def collect_sample_dirs(base_dir):
    for name in os.listdir(base_dir):
        p = os.path.join(base_dir, name)
        if os.path.isdir(p):
            yield name, p

def process_sample(sample_name, sample_dir, out_root):
    """
    读取一个样本目录下的目标疾病 lesion mask 并生成并集，
    保存为: OUT_DIR/<sample_name>.nii.gz
    """
    union = None
    affine = None
    header = None

    for fname in os.listdir(sample_dir):
        if not fname.endswith("_low-res_lesion_mask.nii.gz"):
            continue
        m = pattern.match(fname)
        if not m:
            continue
        disease = m.group("disease")
        if disease not in target_diseases:
            continue

        path = os.path.join(sample_dir, fname)
        img = nib.load(path)
        arr = img.get_fdata()
        mask = (arr >= BINARIZE_THRESHOLD).astype(np.uint8)

        if affine is None:
            affine = img.affine
            header = img.header.copy()

        if union is None:
            union = mask.copy()
        else:
            if union.shape != mask.shape:
                raise ValueError(f"形状不一致: {disease} {mask.shape} vs {union.shape}")
            union |= mask

    # 保存并集 mask，文件名为样本ID
    if union is not None:
        header_union = header.copy()
        header_union.set_data_dtype(np.uint8)
        output_path = os.path.join(out_root, f"{sample_name}.nii.gz")
        nib.save(nib.Nifti1Image(union.astype(np.uint8), affine, header_union), output_path)
        return True
    return False

def main():

    os.system("rm /FM_data/bxg/CT_Report/CT_Report9_test/segmentation_results/*")
    # os.system("rm -r /FM_data/bxg/CT_Report/CT_Report9_test/results/segmamba__Friday_10_October_2025_17h_29m_31s/test_results/prediction")
    args = parse_args()
    if not (0.0 < args.ratio <= 1.0):
        raise ValueError("--ratio 必须在 (0,1] 内，例如 0.20 表示前20%")
    
    # 加载分类最佳阈值
    if not os.path.exists(args.threshold_file):
        raise FileNotFoundError(f"未找到阈值文件: {args.threshold_file}")
    
    best_thresholds = np.load(args.threshold_file)
    if best_thresholds.shape[0] != 18:
        raise ValueError(f"阈值文件应包含18个值，实际: {best_thresholds.shape[0]}")
    
    # 计算绝对阈值 = best_threshold * scale
    absolute_thresholds = best_thresholds * args.threshold_scale
    print(f"加载分类阈值: {args.threshold_file}")
    print(f"绝对阈值缩放因子: {args.threshold_scale}")
    print(f"绝对阈值向量 (前5个): {absolute_thresholds[:5]}")
    
    # 加载预测结果CSV
    if not os.path.exists(args.csv_file):
        raise FileNotFoundError(f"未找到CSV文件: {args.csv_file}")
    
    predictions = load_predictions_csv(args.csv_file)
    print(f"加载预测结果CSV: {args.csv_file}")
    print(f"共读取 {len(predictions)} 个样本的预测结果")

    target_suffix = f"{args.res}_combined_pred.nii.gz"
    # if args.no_combined:
    #     target_suffix = f"{args.res}_pred.nii.gz"

    zero_mask_count = 0
    normal_mask_count = 0

    for epoch_dir in iter_epoch_dirs(args.root, args.epoch):
        epoch_name = os.path.basename(epoch_dir)  # epoch_XX
        out_epoch_dir = os.path.join(args.root, "prediction", epoch_name)
        os.makedirs(out_epoch_dir, exist_ok=True)

        sample_dirs = [d for d in glob.glob(os.path.join(epoch_dir, "*")) if os.path.isdir(d)]
        for sdir in tqdm(sample_dirs, desc=f"{epoch_name}"):
            sample_name = os.path.basename(sdir)
            out_sample_dir = os.path.join(out_epoch_dir, sample_name)
            os.makedirs(out_sample_dir, exist_ok=True)

            pattern = os.path.join(sdir, f"*_{target_suffix}")
            files = sorted(glob.glob(pattern))
            if not files and not args.no_combined:
                # 兜底到原始 pred
                files = sorted(glob.glob(os.path.join(sdir, f"*_{args.res}_pred.nii.gz")))
            if not files:
                continue

            for f in files:
                disease = disease_from_filename(f)
                if disease not in NAME2IDX:
                    continue
                
                disease_idx = NAME2IDX[disease]
                abs_thr = absolute_thresholds[disease_idx]

                nii = nib.load(f)
                data = np.asarray(nii.get_fdata())

                # 查询该样本该疾病的预测结果
                pred_value = 1  # 默认为1
                if sample_name in predictions:
                    if disease in predictions[sample_name]:
                        pred_value = predictions[sample_name][disease]
                    else:
                        print(f"警告: 样本 {sample_name} 未找到疾病 {disease} 的预测结果，使用默认值1")
                else:
                    print(f"警告: 未找到样本 {sample_name} 的预测结果，使用默认值1")

                # 根据pred_value决定生成mask的方式
                if pred_value == 0:
                    # Pred=0，生成全零mask
                    mask = np.zeros_like(data, dtype=np.uint8)
                    zero_mask_count += 1
                    descrip = f"Zero mask: Pred=0 for {disease}"
                else:
                    # Pred=1，使用相对+绝对阈值方法
                    # 1. 相对阈值：前ratio%的高值（在非零值中）
                    rel_thr = topk_threshold(data, args.ratio)
                    if np.isinf(rel_thr):
                        mask_relative = np.zeros_like(data, dtype=bool)
                    else:
                        mask_relative = (data >= rel_thr)
                    
                    # 2. 绝对阈值：大于 abs_thr 的体素
                    mask_absolute = (data > abs_thr)
                    
                    # 3. 取交集
                    mask_combined = mask_relative & mask_absolute
                    mask = mask_combined.astype(np.uint8)
                    normal_mask_count += 1
                    descrip = (
                        f"Lesion mask: Pred=1, top{args.ratio:.2f} (rel_thr={rel_thr:.6f}) "
                        f"AND abs_thr={abs_thr:.6f} from {os.path.basename(f)}"
                    )

                out_name = os.path.basename(f)
                out_name = out_name.replace("_combined_pred.nii.gz", "_lesion_mask.nii.gz")
                out_name = out_name.replace("_pred.nii.gz", "_lesion_mask.nii.gz")
                out_path = os.path.join(out_sample_dir, out_name)

                if (not args.overwrite) and os.path.exists(out_path):
                    continue

                out_nii = nib.Nifti1Image(mask, nii.affine, nii.header)
                out_nii.header["descrip"] = descrip
                nib.save(out_nii, out_path)

    print(f"\n完成统计:")
    print(f"  - 生成全零mask (Pred=0): {zero_mask_count} 个")
    print(f"  - 使用阈值方法 (Pred=1): {normal_mask_count} 个")
    print(f"  - 总计: {zero_mask_count + normal_mask_count} 个病变mask")
    print(f"\n输出目录: {os.path.join(args.root, 'prediction')}")
    os.makedirs(OUT_DIR, exist_ok=True)
    samples = list(collect_sample_dirs(BASE_DIR))
    processed_count = 0
    
    for sample_name, sample_path in tqdm(samples, desc="Processing samples"):
        if process_sample(sample_name, sample_path, OUT_DIR):
            processed_count += 1
    
    print(f"完成：共处理 {processed_count}/{len(samples)} 个样本")
    print(f"输出目录 -> {OUT_DIR}")

if __name__ == "__main__":
    main()