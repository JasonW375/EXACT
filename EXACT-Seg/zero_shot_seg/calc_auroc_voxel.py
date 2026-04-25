# import os
# import json
# import numpy as np
# import nibabel as nib
# from sklearn.metrics import roc_auc_score

# PIXEL_MIN = 100000

# # 数据路径
# gt_dir = "/FM_data/bxg/CT_Report/CT_Report9_test/ReXGroundingCT/lesion_mask"
# pred_dir = "/FM_data/bxg/CT_Report/CT_Report9_test/overlaid_heatmaps_val"
# json_path = "/FM_data/bxg/CT_Report/CT_Report9_test/ReXGroundingCT/dataset.json"

# out_high = "/FM_data/bxg/CT_Report/CT_Report9_test/auroc_scores_gt100k.csv"
# out_low  = "/FM_data/bxg/CT_Report/CT_Report9_test/auroc_scores_lt100k.csv"

# def calculate_auroc(pred_heatmap, target):
#     """计算体素级 AUROC（预测为连续热图，GT 为二值）"""
#     pred_flat = np.asarray(pred_heatmap, dtype=np.float32).ravel()
#     target_flat = (target > 0).astype(np.uint8).ravel()

#     # 必须同时有正类与负类
#     if len(np.unique(target_flat)) < 2:
#         return np.nan
#     try:
#         return roc_auc_score(target_flat, pred_flat)
#     except Exception:
#         return np.nan

# def _strip_nii_suffix(name: str) -> str:
#     base = os.path.basename(name)
#     if base.endswith(".nii.gz"):
#         return base[:-7].strip()
#     if base.endswith(".nii"):
#         return base[:-4].strip()
#     return os.path.splitext(base)[0].strip()

# def _iter_entries(obj):
#     if isinstance(obj, list):
#         for it in obj:
#             if isinstance(it, dict):
#                 yield it
#     elif isinstance(obj, dict):
#         for v in obj.values():
#             if isinstance(v, list):
#                 for it in v:
#                     if isinstance(it, dict):
#                         yield it

# def load_pixel_sums(json_path):
#     with open(json_path, "r", encoding="utf-8") as jf:
#         jd = json.load(jf)
#     pixel_sums = {}
#     for entry in _iter_entries(jd):
#         name = entry.get("name")
#         if not name:
#             continue
#         base = _strip_nii_suffix(name)
#         px = entry.get("pixels", {})
#         if isinstance(px, dict):
#             s = sum(v for v in px.values() if isinstance(v, (int, float)))
#         else:
#             s = 0
#         pixel_sums[base] = s
#     return pixel_sums

# def collect_pred_map(pred_dir):
#     """
#     返回: {规范化sample_id(去掉 _overlaid_heatmap): 预测文件完整路径}
#     """
#     pred_map = {}
#     for f in os.listdir(pred_dir):
#         if not f.endswith(".nii.gz"):
#             continue
#         sid_full = _strip_nii_suffix(f)
#         sid = sid_full.replace("_overlaid_heatmap", "")
#         pred_map[sid] = os.path.join(pred_dir, f)
#     return pred_map

# def eval_group_auroc(sample_ids, group_name, outfile, pred_map):
#     if not sample_ids:
#         print(f"{group_name}: 无样本，跳过。")
#         return

#     with open(outfile, "w") as rf:
#         rf.write("SampleID,AUROC\n")

#     scores = []
#     for sid in sorted(sample_ids):
#         gt_path = os.path.join(gt_dir, f"{sid}.nii.gz")
#         pred_path = pred_map.get(sid)
#         if not os.path.exists(gt_path):
#             print(f"[{group_name}] 缺少GT: {sid}, 跳过。")
#             continue
#         if not pred_path or (not os.path.exists(pred_path)):
#             print(f"[{group_name}] 缺少Pred: {sid}, 跳过。")
#             continue
#         try:
#             gt = nib.load(gt_path).get_fdata()
#             pred = nib.load(pred_path).get_fdata()
#             if gt.shape != pred.shape:
#                 print(f"[{group_name}] 形状不匹配 {sid}: GT {gt.shape} vs Pred {pred.shape}, 跳过。")
#                 continue
#             auroc = calculate_auroc(pred, gt)
#             if np.isnan(auroc):
#                 print(f"[{group_name}] {sid}: AUROC 无效（单一类别），跳过写入。")
#                 continue
#             scores.append(auroc)
#             with open(outfile, "a") as rf:
#                 rf.write(f"{sid},{auroc:.4f}\n")
#         except Exception as e:
#             print(f"[{group_name}] 处理 {sid} 出错: {e}")

#     if scores:
#         arr = np.array(scores, dtype=float)
#         print(f"{group_name}: 样本数={len(scores)} | "
#               f"Mean={arr.mean():.4f} | Median={np.median(arr):.4f} | "
#               f"Std={arr.std(ddof=0):.4f} | Min={arr.min():.4f} | Max={arr.max():.4f}")
#         print(f"{group_name} 结果写入: {outfile}")
#     else:
#         print(f"{group_name}: 无有效分数写入。")

# def main():
#     if not os.path.isdir(gt_dir):
#         raise FileNotFoundError(f"GT目录不存在: {gt_dir}")
#     if not os.path.isdir(pred_dir):
#         raise FileNotFoundError(f"Pred目录不存在: {pred_dir}")
#     if not os.path.exists(json_path):
#         raise FileNotFoundError(f"dataset.json 不存在: {json_path}")

#     pixel_sums = load_pixel_sums(json_path)
#     pred_map = collect_pred_map(pred_dir)

#     # 仅统计既有像素条目、又有预测文件的样本
#     pred_ids = set(pred_map.keys())
#     valid_ids = [sid for sid in pred_ids if sid in pixel_sums]
#     missing_json = sorted(pred_ids - set(pixel_sums.keys()))
#     if missing_json[:5]:
#         print(f"警告：{len(missing_json)} 个预测样本在 dataset.json 中无像素条目，例如前5个: {missing_json[:5]}")

#     high_ids = [sid for sid in valid_ids if pixel_sums[sid] > PIXEL_MIN]
#     low_ids  = [sid for sid in valid_ids if pixel_sums[sid] < PIXEL_MIN]
#     eq_ids   = [sid for sid in valid_ids if pixel_sums[sid] == PIXEL_MIN]

#     print(f"> {PIXEL_MIN}: {len(high_ids)} 个样本")
#     print(f"< {PIXEL_MIN}: {len(low_ids)} 个样本")
#     if eq_ids:
#         print(f"== {PIXEL_MIN}: {len(eq_ids)} 个样本（不计入两组）")

#     eval_group_auroc(high_ids, f"AUROC(像素和>{PIXEL_MIN})", out_high, pred_map)
#     eval_group_auroc(low_ids,  f"AUROC(像素和<{PIXEL_MIN})", out_low,  pred_map)

# if __name__ == "__main__":
#     main()

import os
import argparse
import numpy as np
import nibabel as nib
from sklearn.metrics import roc_auc_score

def calculate_auroc(pred_heatmap, target):
    """计算体素级 AUROC（预测为连续热图，GT 为二值）"""
    pred_flat = np.asarray(pred_heatmap, dtype=np.float32).ravel()
    target_flat = (target > 0).astype(np.uint8).ravel()
    # 必须同时有正类与负类
    if len(np.unique(target_flat)) < 2:
        return np.nan
    try:
        return roc_auc_score(target_flat, pred_flat)
    except Exception:
        return np.nan

def strip_nii_suffix(name: str) -> str:
    base = os.path.basename(name)
    if base.endswith(".nii.gz"):
        return base[:-7]
    if base.endswith(".nii"):
        return base[:-4]
    return os.path.splitext(base)[0]

def is_nii_file(name: str) -> bool:
    return name.endswith(".nii.gz") or name.endswith(".nii")

def find_gt(gt_dir: str, sid: str) -> str | None:
    p1 = os.path.join(gt_dir, f"{sid}.nii.gz")
    p2 = os.path.join(gt_dir, f"{sid}.nii")
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    return None

def main():
    ap = argparse.ArgumentParser(description="计算所有样本的体素级AUROC（不分组）。")
    ap.add_argument("--gt_dir", default="/path/to/bxg/storage/ReXGroundingCT/high_res_mask", help="GT目录（NIfTI）")
    ap.add_argument("--pred_dir", default="/FM_data/bxg/fvlm/grad_cam_valid", help="预测热图目录（NIfTI）")
    ap.add_argument("--out", default="/FM_data/bxg/CT_Report/CT_Report9_test/auroc_scores_all.csv", help="输出CSV文件")
    ap.add_argument("--remove_suffix", default="_overlaid_heatmap", help="从预测文件名中去除的后缀以对齐GT（可为空）")
    args = ap.parse_args()

    if not os.path.isdir(args.gt_dir):
        raise FileNotFoundError(f"GT目录不存在: {args.gt_dir}")
    if not os.path.isdir(args.pred_dir):
        raise FileNotFoundError(f"预测目录不存在: {args.pred_dir}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("SampleID,AUROC\n")

    scores = []
    n_total = 0
    n_ok = 0
    n_skip = 0

    for fname in sorted(os.listdir(args.pred_dir)):
        if not is_nii_file(fname):
            continue
        n_total += 1
        sid_full = strip_nii_suffix(fname)
        sid = sid_full.replace(args.remove_suffix, "") if args.remove_suffix else sid_full
        # sid=f"{sid}_mask"
        # sid=sid.replace("_overlaid_heatmap_mask","")
        pred_path = os.path.join(args.pred_dir, fname)
        gt_path = find_gt(args.gt_dir, sid)
        if gt_path is None:
            print(f"[缺少GT] {sid}，跳过该样本。")
            n_skip += 1
            continue

        try:
            gt = nib.load(gt_path).get_fdata()
            pred = nib.load(pred_path).get_fdata()
            if gt.shape != pred.shape:
                print(f"[形状不匹配] {sid}: GT{gt.shape} vs Pred{pred.shape}，跳过。")
                n_skip += 1
                continue

            auroc = calculate_auroc(pred, gt)
            if np.isnan(auroc):
                print(f"[单一类别] {sid}: 无法计算AUROC，跳过写入。")
                n_skip += 1
                continue

            with open(args.out, "a") as f:
                f.write(f"{sid},{auroc:.4f}\n")
            scores.append(auroc)
            n_ok += 1
        except Exception as e:
            print(f"[错误] 处理 {sid} 失败: {e}")
            n_skip += 1

    if scores:
        arr = np.asarray(scores, dtype=float)
        print(f"完成: 写入 {n_ok}/{n_total} 条 -> {args.out}")
        print(f"AUROC 统计: mean={arr.mean():.4f}, median={np.median(arr):.4f}, "
              f"std={arr.std(ddof=0):.4f}, min={arr.min():.4f}, max={arr.max():.4f}")
    else:
        print(f"无有效AUROC写入（共尝试 {n_total}，跳过 {n_skip}）。")

if __name__ == "__main__":
    main()