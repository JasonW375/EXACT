import os
import json
import argparse
import numpy as np
import nibabel as nib
from sklearn.metrics import average_precision_score
from typing import Tuple, List
import math

PIXEL_MIN = 100000

# 默认数据路径（可被命令行参数覆盖）
# gt_dir="/path/to/bxg/storage/ReXGroundingCT/high_res_mask"
gt_dir="/FM_data/bxg/CT_Report/CT_covid_withlabel/visualization_mask"
# gt_dir = "/path/to/bxg/storage/ReXGroundingCT/lesion_mask"
# gt_dir="/FM_data/bxg/CT_Report/CT_covid_withlabel/Infection_Mask"
# gt_dir="/path/to/bxg/storage/MosMedData-Chest-CT-Scans-with-COVID-19-Related-Findings/masks"
pred_dir = "/FM_data/bxg/CT_Report/CT_Report9_test/results/covidfull/overlaid_heatmaps_covidfull"
# pred_dir = "/FM_data/bxg/CT_Report/CT_Report15_18abn_2decoder/results/Rex_No_Resume/test_results/heatmap"
json_path = "/path/to/bxg/storage/ReXGroundingCT/dataset.json"

# 结果路径（文件名模板，实际可能生成 *_per_sample_mean.csv 与 *_global_concat.csv）
out_high = f"{pred_dir}/aupr_scores_gt100k.csv"
out_low  = f"{pred_dir}/aupr_scores_lt100k.csv"
out_all  = f"{pred_dir}/aupr_scores_all.csv"


def _strip_nii_suffix(name: str) -> str:
    base = os.path.basename(name)
    if base.endswith(".nii.gz"):
        return base[:-7].strip()
    if base.endswith(".nii"):
        return base[:-4].strip()
    return os.path.splitext(base)[0].strip()


def _iter_entries(obj):
    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                yield it
    elif isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, list):
                for it in v:
                    if isinstance(it, dict):
                        yield it


def load_pixel_sums(json_path):
    with open(json_path, "r", encoding="utf-8") as jf:
        jd = json.load(jf)
    pixel_sums = {}
    for entry in _iter_entries(jd):
        name = entry.get("name")
        if not name:
            continue
        base = _strip_nii_suffix(name)
        px = entry.get("pixels", {})
        if isinstance(px, dict):
            s = sum(v for v in px.values() if isinstance(v, (int, float)))
        else:
            s = 0
        pixel_sums[base] = s
    return pixel_sums


def collect_pred_map(pred_dir):
    """
    返回 {规范化sample_id(去掉 _overlaid_heatmap): 预测文件完整路径}
    """
    pred_map = {}
    for f in os.listdir(pred_dir):
        if not f.endswith(".nii.gz"):
            continue
        sid_full = _strip_nii_suffix(f)
        sid = sid_full.replace("_overlaid_heatmap", "")
        pred_map[sid] = os.path.join(pred_dir, f)
    return pred_map


def calculate_aupr_global(preds_all: np.ndarray, gts_all: np.ndarray):
    """
    所有样本像素拼接后计算单个全局 AUPR。
    """
    if len(np.unique(gts_all)) < 2:
        return np.nan
    try:
        return average_precision_score(gts_all, preds_all)
    except Exception:
        return np.nan


def calculate_aupr_per_sample(pred: np.ndarray, gt: np.ndarray):
    pred_flat = np.asarray(pred, dtype=np.float32).ravel()
    gt_flat = (gt > 0).astype(np.uint8).ravel()
    valid = np.isfinite(pred_flat)
    if not np.all(valid):
        pred_flat = pred_flat[valid]
        gt_flat = gt_flat[valid]
    if len(np.unique(gt_flat)) < 2:
        return np.nan
    try:
        return average_precision_score(gt_flat, pred_flat)
    except Exception:
        return np.nan


def _bootstrap_mean_ci(values: List[float], n_boot: int = 1000, ci: float = 0.95, rng: np.random.Generator = None) -> Tuple[float, float, float]:
    """Compute mean and (low, high) confidence interval via bootstrap.

    Args:
        values: list of finite float scores
        n_boot: number of bootstrap resamples
        ci: confidence level (e.g., 0.95)
        rng: optional numpy Generator for reproducibility

    Returns:
        mean, ci_low, ci_high
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size == 1:
        # Not enough samples for CI; return NaNs for bounds
        return mean, float("nan"), float("nan")
    if rng is None:
        rng = np.random.default_rng()
    boots = np.empty(n_boot, dtype=np.float64)
    n = arr.size
    for i in range(n_boot):
        sample = arr[rng.integers(0, n, size=n)]
        boots[i] = sample.mean()
    alpha = (1.0 - ci) / 2.0
    low = float(np.quantile(boots, alpha))
    high = float(np.quantile(boots, 1.0 - alpha))
    return mean, low, high


# def eval_group_aupr_per_sample_mean(sample_ids, group_name, outfile, pred_map, gt_root):
#     """
#     对组内每个样本分别计算 AUPR，写 CSV (SampleID,AUPR)，再追加 MEAN。
#     """
#     if not sample_ids:
#         print(f"{group_name}: 无样本，跳过。")
#         return np.nan
#     os.makedirs(os.path.dirname(outfile), exist_ok=True)
#     scores = []
#     used = 0
#     with open(outfile, "w", encoding="utf-8") as rf:
#         rf.write("SampleID,AUPR\n")
#         for sid in sorted(sample_ids):
#             gt_path = os.path.join(gt_root, f"{sid}.nii.gz")
#             pred_path = pred_map.get(sid)
#             sid=sid.replace("_overlaid_heatmap","")
#             if not os.path.exists(gt_path):
#                 print(f"[{group_name}] 缺少GT: {sid}, 跳过。")
#                 rf.write(f"{sid},NaN\n")
#                 continue
#             if (not pred_path) or (not os.path.exists(pred_path)):
#                 print(f"[{group_name}] 缺少Pred: {sid}, 跳过。")
#                 rf.write(f"{sid},NaN\n")
#                 continue
#             try:
#                 gt = nib.load(gt_path).get_fdata()
#                 pred = nib.load(pred_path).get_fdata()
#                 if gt.shape != pred.shape:
#                     print(f"[{group_name}] 形状不匹配 {sid}: GT {gt.shape} vs Pred {pred.shape}，跳过。")
#                     rf.write(f"{sid},NaN\n")
#                     continue
#                 score = calculate_aupr_per_sample(pred, gt)
#                 if np.isnan(score):
#                     rf.write(f"{sid},NaN\n")
#                     print(f"[{group_name}] {sid}: 单一类别，AUPR=NaN。")
#                 else:
#                     rf.write(f"{sid},{score:.6f}\n")
#                     scores.append(score)
#                     used += 1
#             except Exception as e:
#                 print(f"[{group_name}] 处理 {sid} 出错: {e}")
#                 rf.write(f"{sid},NaN\n")
#         if used > 0:
#             mean_score = float(np.mean(scores))
#             rf.write(f"MEAN,{mean_score:.6f}\n")
#             print(f"{group_name}: Mean AUPR over {used} valid samples = {mean_score:.4f}")
#         else:
#             rf.write("MEAN,NaN\n")
#             print(f"{group_name}: 无有效样本计算 Mean AUPR。")
#     return np.mean(scores) if used > 0 else np.nan


# def run_global_concat(sample_ids, group_name, outfile, pred_map, gt_root):
#     """
#     拼接所有样本像素并计算单个全局 AUPR，写入一行 CSV。
#     """
#     preds_list, gts_list = [], []
#     total_pixels = pos_pixels = used = 0
#     for sid in sample_ids:
#         gt_path = os.path.join(gt_root, f"{sid}.nii.gz")
#         pred_path = pred_map.get(sid)
#         if not os.path.exists(gt_path) or (not pred_path) or (not os.path.exists(pred_path)):
#             continue
#         try:
#             gt = nib.load(gt_path).get_fdata()
#             pred = nib.load(pred_path).get_fdata()
#             if gt.shape != pred.shape:
#                 continue
#             gt_bin = (gt > 0).astype(np.uint8)
#             preds_list.append(pred.astype(np.float32).ravel())
#             gts_list.append(gt_bin.ravel())
#             total_pixels += gt_bin.size
#             pos_pixels += int(gt_bin.sum())
#             used += 1
#         except Exception:
#             continue
#     os.makedirs(os.path.dirname(outfile), exist_ok=True)
#     with open(outfile, "w", encoding="utf-8") as wf:
#         wf.write("Group,TotalPixels,Positives,Negatives,AUPR,UsedSamples\n")
#         if not preds_list:
#             wf.write(f"{group_name},0,0,0,NaN,0\n")
#             print("无有效样本计算全局拼接 AUPR。")
#             return np.nan
#         preds_all = np.concatenate(preds_list)
#         gts_all = np.concatenate(gts_list)
#         aupr_global = calculate_aupr_global(preds_all, gts_all)
#         neg_pixels = int(gts_all.size - gts_all.sum())
#         if np.isnan(aupr_global):
#             wf.write(f"{group_name},{total_pixels},{pos_pixels},{neg_pixels},NaN,{used}\n")
#             print(f"{group_name}: 聚合后单一类别，无法计算 AUPR。")
#         else:
#             wf.write(f"{group_name},{total_pixels},{pos_pixels},{neg_pixels},{aupr_global:.6f},{used}\n")
#             print(f"{group_name}: Pixels={total_pixels} Pos={pos_pixels} Neg={neg_pixels} Used={used} AUPR={aupr_global:.4f}")
#     return aupr_global

from concurrent.futures import ProcessPoolExecutor, as_completed

def _compute_one_sample(sid: str, pred_map: dict, gt_root: str):
    """
    单样本读取并计算AUPR，返回 (sid, score or np.nan, err_msg or None)
    """
    gt_path = os.path.join(gt_root, f"{sid}.nii.gz")
    pred_path = pred_map.get(sid)
    if not os.path.exists(gt_path):
        return sid, np.nan, f"缺少GT: {sid}"
    if (not pred_path) or (not os.path.exists(pred_path)):
        return sid, np.nan, f"缺少Pred: {sid}"
    try:
        gt = nib.load(gt_path).get_fdata()
        pred = nib.load(pred_path).get_fdata()
        if gt.shape != pred.shape:
            return sid, np.nan, f"形状不匹配 {sid}: GT {gt.shape} vs Pred {pred.shape}"
        score = calculate_aupr_per_sample(pred, gt)
        if np.isnan(score):
            return sid, np.nan, f"{sid}: 单一类别，AUPR=NaN。"
        return sid, float(score), None
    except Exception as e:
        return sid, np.nan, f"处理 {sid} 出错: {e}"

def _read_for_concat(sid: str, pred_map: dict, gt_root: str):
    """
    单样本读取并返回扁平像素，用于全局拼接。
    """
    gt_path = os.path.join(gt_root, f"{sid}.nii.gz")
    pred_path = pred_map.get(sid)
    if not os.path.exists(gt_path) or (not pred_path) or (not os.path.exists(pred_path)):
        return None
    try:
        gt = nib.load(gt_path).get_fdata()
        pred = nib.load(pred_path).get_fdata()
        if gt.shape != pred.shape:
            return None
        gt_bin = (gt > 0).astype(np.uint8)
        return pred.astype(np.float32).ravel(), gt_bin.ravel()
    except Exception:
        return None

def eval_group_aupr_per_sample_mean(sample_ids, group_name, outfile, pred_map, gt_root, max_workers=os.cpu_count()):
    """
    并行计算每个样本 AUPR，写 CSV，返回均值。
    """
    if not sample_ids:
        print(f"{group_name}: 无样本，跳过。")
        return np.nan
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    scores = []
    used = 0
    with open(outfile, "w", encoding="utf-8") as rf:
        rf.write("SampleID,AUPR\n")
        # 并行
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_compute_one_sample, sid, pred_map, gt_root): sid for sid in sample_ids}
            for fut in as_completed(futures):
                sid, score, err = fut.result()
                if np.isnan(score):
                    rf.write(f"{sid},NaN\n")
                    if err:
                        print(f"[{group_name}] {err}")
                else:
                    rf.write(f"{sid},{score:.6f}\n")
                    scores.append(score)
                    used += 1
        if used > 0:
            mean_score, ci_low, ci_high = _bootstrap_mean_ci(scores, n_boot=1000, ci=0.95)
            rf.write(f"MEAN,{mean_score:.6f}\n")
            rf.write(f"CI95_low,{(ci_low if not np.isnan(ci_low) else float('nan'))}\n")
            rf.write(f"CI95_high,{(ci_high if not np.isnan(ci_high) else float('nan'))}\n")
            print(f"{group_name}: Mean AUPR over {used} valid samples = {mean_score:.4f} (95% CI: {ci_low:.4f} - {ci_high:.4f})")
        else:
            rf.write("MEAN,NaN\n")
            rf.write("CI95_low,NaN\n")
            rf.write("CI95_high,NaN\n")
            print(f"{group_name}: 无有效样本计算 Mean AUPR。")
    return np.mean(scores) if used > 0 else np.nan

# 全局拼接AUPR计算已移除，保留逐样本均值与其95%CI
def main():
    parser = argparse.ArgumentParser(description="计算体素级 AUPR（支持分组模式 / 全部 / both）")
    parser.add_argument("--mode", choices=["split", "all", "both"], default="all",
                        help="split: 基于 json 像素阈值分组并逐样本求平均；all: 不用 json，对全部样本计算；both: 先分组再全部。")
    parser.add_argument("--pixel_min", type=int, default=PIXEL_MIN, help="分组像素阈值（仅 split/both 使用）")
    parser.add_argument("--json", type=str, default=None, help="像素统计 json 路径（split/both 需要）")
    parser.add_argument("--pred_dir", type=str, default=None, help="预测目录，覆盖默认")
    parser.add_argument("--gt_dir", type=str, default=None, help="GT 目录，覆盖默认")
    args = parser.parse_args()

    pixel_min = args.pixel_min
    gt_d = args.gt_dir if args.gt_dir else gt_dir
    pred_d = args.pred_dir if args.pred_dir else pred_dir

    if not os.path.isdir(gt_d):
        raise FileNotFoundError(f"GT目录不存在: {gt_d}")
    if not os.path.isdir(pred_d):
        raise FileNotFoundError(f"预测目录不存在: {pred_d}")

    pred_map = collect_pred_map(pred_d)
    pred_ids = sorted(pred_map.keys())
    if not pred_ids:
        raise RuntimeError("预测目录下未找到任何 .nii.gz 文件。")

    # -------- split / both 需要 json 分组 ----------
    if args.mode in ("split", "both"):
        json_p = args.json if args.json else json_path
        if not json_p or not os.path.exists(json_p):
            raise FileNotFoundError(f"需要像素统计 json 文件，但未找到: {json_p}")
        pixel_sums = load_pixel_sums(json_p)

        missing = [sid for sid in pred_ids if sid not in pixel_sums]
        if missing:
            raise ValueError(f"{len(missing)} 个预测样本不在 json 像素统计中: {missing[:10]} ...")

        valid_ids = pred_ids
        high_ids = [sid for sid in valid_ids if pixel_sums[sid] > pixel_min]
        low_ids  = [sid for sid in valid_ids if pixel_sums[sid] < pixel_min]
        eq_ids   = [sid for sid in valid_ids if pixel_sums[sid] == pixel_min]

        print(f"> {pixel_min}: {len(high_ids)} 个样本")
        print(f"< {pixel_min}: {len(low_ids)} 个样本")
        if eq_ids:
            print(f"== {pixel_min}: {len(eq_ids)} 个样本（不计入高/低分组）")

        def run_split():
            out_high_mean = out_high.replace(".csv", "_per_sample_mean.csv")
            out_low_mean  = out_low.replace(".csv", "_per_sample_mean.csv")
            out_all_mean  = out_all.replace(".csv", "_per_sample_mean.csv")
            print("运行 split: 按像素阈值分组，逐样本 AUPR 平均。")
            eval_group_aupr_per_sample_mean(high_ids, f"AUPR_像素和大于{pixel_min}", out_high_mean, pred_map, gt_d)
            eval_group_aupr_per_sample_mean(low_ids,  f"AUPR_像素和小于{pixel_min}", out_low_mean,  pred_map, gt_d)
            eval_group_aupr_per_sample_mean(valid_ids, "AUPR_全部样本", out_all_mean, pred_map, gt_d)

        def run_all_with_valid():
            out_all_mean   = out_all.replace(".csv", "_per_sample_mean.csv")
            print("运行 all 部分：全部样本逐样本均值。")
            mean_score = eval_group_aupr_per_sample_mean(valid_ids, "AUPR_全部样本", out_all_mean, pred_map, gt_d)
            print(f"样本级 Mean AUPR (全部样本): {mean_score}")

        if args.mode == "split":
            run_split()
            return
        if args.mode == "both":
            run_split()
            run_all_with_valid()
            return

    # -------- mode=all 且不使用 json ----------
    if args.mode == "all":
        print("运行 all：不使用 json，直接全部样本。")
        valid_ids = pred_ids
        out_all_mean   = out_all.replace(".csv", "_per_sample_mean.csv")
        mean_score = eval_group_aupr_per_sample_mean(valid_ids, "AUPR_全部样本", out_all_mean, pred_map, gt_d)
        print(f"样本级 Mean AUPR (全部样本): {mean_score}")


if __name__ == "__main__":
    main()