# import os
# import re
# import glob
# import argparse
# import numpy as np
# import nibabel as nib
# from tqdm import tqdm

# # 参与叠加/平均的关注疾病（与 overlay_heatmap.py 一致）
# OVERLAY_DISEASES = [
#     "Arterial wall calcification", "Emphysema", "Atelectasis", "Lung nodule",
#     "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",
#     "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation",
#     "Bronchiectasis", "Interlobular septal thickening",
# ]
# OVERLAY_DISEASES_SET = set(OVERLAY_DISEASES)


# def parse_args():
#     p = argparse.ArgumentParser(
#         description="从已叠加的平均热图目录读取并阈值化，保存分割结果（不再执行叠加）"
#     )
#     p.add_argument(
#         "--root",
#         default="/FM_data/bxg/CT_Report/CT_Report9_test/results/segmamba__Monday_13_October_2025_14h_28m_40s/test_results",
#         help="历史参数，保留以兼容（本脚本当前不使用）",
#     )
#     p.add_argument(
#         "--epoch",
#         type=int,
#         default=10,
#         help="历史参数，保留以兼容（本脚本当前不使用）",
#     )
#     p.add_argument(
#         "--res",
#         choices=["high-res", "low-res"],
#         default="low-res",
#         help="历史参数，保留以兼容（本脚本当前不使用）",
#     )
#     p.add_argument(
#         "--no-combined",
#         action="store_true",
#         help="历史参数，保留以兼容（本脚本当前不使用）",
#     )
#     # 阈值模式：abs=绝对阈值；rel=相对阈值；both=两者取交集
#     p.add_argument(
#         "--thresh-mode",
#         choices=["abs", "rel", "both"],
#         default="abs",
#         help="二值化方式：abs=绝对；rel=相对；both=二者交集",
#     )
#     p.add_argument(
#         "--binary-threshold",
#         type=float,
#         default=0.02,
#         help="绝对阈值，范围[0,1]（当 --thresh-mode 包含 abs 时生效）",
#     )
#     p.add_argument(
#         "--ratio",
#         type=float,
#         default=0.02,
#         help="相对阈值比例(0,1]，如0.2表示取前20%最高值（当 --thresh-mode 包含 rel 时生效）",
#     )
#     p.add_argument(
#         "--in-overlay",
#         default="/FM_data/bxg/CT_Report/CT_Report9_test/results/mosmed_zeroshot/overlaid_heatmaps_train",
#         help="已叠加的平均热图目录（读取 *_overlaid_heatmap.nii.gz）",
#     )
#     p.add_argument(
#         "--out-overlay",
#         default="/FM_data/bxg/CT_Report/CT_Report9_test/results/mosmed_zeroshot/segmentations",
#         help="保留以兼容（当前不写平均热图，仅读取）",
#     )
#     p.add_argument(
#         "--out-seg",
#         default="/FM_data/bxg/CT_Report/CT_Report9_test/results/mosmed_zeroshot/segmentations",
#         help="分割结果输出目录",
#     )
#     p.add_argument(
#         "--overwrite",
#         action="store_true",
#         help="若目标文件已存在则覆盖",
#     )
#     return p.parse_args()


# def iter_epoch_dirs(root: str, epoch: int | None):
#     hm_root = os.path.join(root, "prediction_heatmaps")
#     if not os.path.isdir(hm_root):
#         raise FileNotFoundError(f"未找到目录: {hm_root}")
#     if epoch is not None:
#         return [os.path.join(hm_root, f"epoch_{epoch}")]
#     return sorted(
#         [d for d in glob.glob(os.path.join(hm_root, "epoch_*")) if os.path.isdir(d)]
#     )


# def build_overlay_for_sample(
#     sample_dir: str,
#     res: str,
#     prefer_combined: bool,
#     diseases_set: set[str],
# ):
#     """
#     返回 (avg_vol, affine, header, used_type)
#     若该样本无法构建平均热图则返回 (None, None, None, None)
#     """
#     pred_types = ["combined_pred", "pred"] if prefer_combined else ["pred"]

#     affine = None
#     header = None

#     for pred_type in pred_types:
#         suffix = f"{res}_{pred_type}.nii.gz"
#         rx = re.compile(rf"^(?P<disease>.+?)_GT\d+_PD\d+_{re.escape(suffix)}$")
#         sum_vol = None
#         count_any = 0

#         for fname in os.listdir(sample_dir):
#             if not fname.endswith(suffix):
#                 continue
#             m = rx.match(fname)
#             if not m:
#                 continue
#             disease = m.group("disease")
#             if disease not in diseases_set:
#                 continue

#             path = os.path.join(sample_dir, fname)
#             img = nib.load(path)
#             arr = img.get_fdata().astype(np.float32)

#             if sum_vol is None:
#                 sum_vol = np.zeros_like(arr, dtype=np.float32)
#                 affine = img.affine
#                 header = img.header.copy()

#             if sum_vol.shape != arr.shape:
#                 raise ValueError(
#                     f"体素形状不一致: {fname} {arr.shape} vs {sum_vol.shape}"
#                 )

#             sum_vol += arr
#             count_any += 1

#         if sum_vol is not None and count_any > 0:
#             # 按“关注疾病总数”求平均（与 overlay_heatmap.py 一致）
#             avg_vol = sum_vol / float(len(diseases_set))
#             return avg_vol, affine, header, pred_type

#     return None, None, None, None


# def topk_threshold(vol: np.ndarray, ratio: float) -> float:
#     """返回使得约 ratio 比例体素落入最高值集合的阈值（仅在>0的体素上计算）"""
#     flat = vol.reshape(-1)
#     finite = flat[np.isfinite(flat)]
#     if finite.size == 0:
#         return np.inf
#     nonzero = finite[finite > 0]
#     if nonzero.size == 0:
#         return np.inf
#     k = max(1, int(np.ceil(ratio * nonzero.size)))
#     idx = nonzero.size - k  # 选择前k大的最小值
#     thr = float(np.partition(nonzero, idx)[idx])
#     return thr


# def main():
#     args = parse_args()

#     if "abs" in args.thresh_mode:
#         if not (0.0 <= args.binary_threshold <= 1.0):
#             raise ValueError("--binary-threshold 必须在 [0,1]")
#     if "rel" in args.thresh_mode:
#         if not (0.0 < args.ratio <= 1.0):
#             raise ValueError("--ratio 必须在 (0,1]，如 0.2 表示前20%")

#     # 仅创建分割输出目录；不再写平均热图
#     os.makedirs(args.out_seg, exist_ok=True)

#     in_dir = args.in_overlay
#     if not os.path.isdir(in_dir):
#         raise FileNotFoundError(f"未找到已叠加热图目录: {in_dir}")

#     # 优先匹配 *_overlaid_heatmap.nii.gz；若没有则回退所有 nii.gz
#     overlay_files = sorted(glob.glob(os.path.join(in_dir, "*_overlaid_heatmap.nii.gz")))
#     if not overlay_files:
#         overlay_files = sorted(glob.glob(os.path.join(in_dir, "*.nii.gz")))
#     if not overlay_files:
#         raise FileNotFoundError(f"目录中未找到任何 .nii.gz 文件: {in_dir}")

#     total_samples = len(overlay_files)
#     success_samples = 0

#     for ofile in tqdm(overlay_files, desc="thresholding"):
#         base = os.path.basename(ofile)
#         # 去除后缀得到样本名
#         sample_name = base.replace("_overlaid_heatmap.nii.gz", "")
#         if sample_name.endswith(".nii.gz"):
#             sample_name = sample_name[:-7]

#         try:
#             img = nib.load(ofile)
#             avg_vol = img.get_fdata().astype(np.float32)
#             affine = img.affine
#             header = img.header.copy()

#             # 绝对阈值掩码
#             mask_abs = None
#             if args.thresh_mode in ("abs", "both"):
#                 mask_abs = (avg_vol >= args.binary_threshold).astype(np.uint8)

#             # 相对阈值掩码
#             mask_rel = None
#             if args.thresh_mode in ("rel", "both"):
#                 thr_rel = topk_threshold(avg_vol, args.ratio)
#                 if np.isfinite(thr_rel):
#                     mask_rel = (avg_vol >= thr_rel).astype(np.uint8)
#                 else:
#                     mask_rel = np.zeros_like(avg_vol, dtype=np.uint8)

#             # 合成最终掩码
#             if args.thresh_mode == "abs":
#                 seg_mask = mask_abs
#             elif args.thresh_mode == "rel":
#                 seg_mask = mask_rel
#             else:  # both -> 取交集
#                 if mask_abs is None or mask_rel is None:
#                     seg_mask = np.zeros_like(avg_vol, dtype=np.uint8)
#                 else:
#                     seg_mask = (mask_abs & mask_rel).astype(np.uint8)

#             seg_header = header.copy()
#             seg_header.set_data_dtype(np.uint8)
#             seg_path = os.path.join(args.out_seg, f"{sample_name}.nii.gz")
#             if args.overwrite or (not os.path.exists(seg_path)):
#                 nib.save(nib.Nifti1Image(seg_mask, affine, seg_header), seg_path)
#                 success_samples += 1
#         except Exception as e:
#             print(f"处理 {base} 出错: {e}")

#     print(f"完成：阈值分割 {success_samples}/{total_samples} 个样本")
#     print(f"输入平均热图目录 -> {in_dir}")
#     print(f"分割输出目录 -> {args.out_seg}")
#     print(f"默认优先使用 *_combined_pred.nii.gz；缺失则回退 *_pred.nii.gz")


# if __name__ == "__main__":
#     main()

import os
import re
import glob
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
import json  # 新增

# 参与叠加/平均的关注疾病（与 overlay_heatmap.py 一致）
OVERLAY_DISEASES = [
    "Arterial wall calcification", "Emphysema", "Atelectasis", "Lung nodule",
    "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",
    "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation",
    "Bronchiectasis", "Interlobular septal thickening",
]
OVERLAY_DISEASES_SET = set(OVERLAY_DISEASES)

PIXEL_MIN = 100000  # 阈值

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

def load_pixel_sums(json_path: str) -> dict[str, int]:
    """
    读取 dataset.json，返回 {sample_id: 像素总数}；sample_id 与 NIfTI 文件名的基本名对齐。
    """
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

def parse_args():
    p = argparse.ArgumentParser(
        description="从已叠加的平均热图目录读取并阈值化，保存分割结果（不再执行叠加）"
    )
    p.add_argument(
        "--root",
        default="/FM_data/bxg/CT_Report/CT_Report9_test/results/segmamba__Monday_13_October_2025_14h_28m_40s/test_results",
        help="历史参数，保留以兼容（本脚本当前不使用）",
    )
    p.add_argument(
        "--epoch",
        type=int,
        default=10,
        help="历史参数，保留以兼容（本脚本当前不使用）",
    )
    p.add_argument(
        "--res",
        choices=["high-res", "low-res"],
        default="low-res",
        help="历史参数，保留以兼容（本脚本当前不使用）",
    )
    p.add_argument(
        "--no-combined",
        action="store_true",
        help="历史参数，保留以兼容（本脚本当前不使用）",
    )
    # 阈值模式：abs=绝对阈值；rel=相对阈值；both=两者取交集
    p.add_argument(
        "--thresh-mode",
        choices=["abs", "rel", "both"],
        default="abs",
        help="二值化方式：abs=绝对；rel=相对；both=二者交集",
    )
    p.add_argument(
        "--binary-threshold",
        type=float,
        default=0.004,
        help="绝对阈值，范围[0,1]（当 --thresh-mode 包含 abs 时生效）",
    )
    p.add_argument(
        "--ratio",
        type=float,
        default=0.1,
        help="相对阈值比例(0,1]，如0.2表示取前20%最高值（当 --thresh-mode 包含 rel 时生效）",
    )
    p.add_argument(
        "--in-overlay",
        default="/FM_data/bxg/CT_Report/CT_Report9_test/results/covidfull/overlaid_heatmaps_covidfull",
        help="已叠加的平均热图目录（读取 *_overlaid_heatmap.nii.gz）",
    )
    p.add_argument(
        "--out-seg",
        default="/FM_data/bxg/CT_Report/CT_Report9_test/results/covidfull/segmentation_results_covidfull",
        help="分割结果输出目录",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="若目标文件已存在则覆盖",
    )
    # 新增：子集选择与 json 路径
    p.add_argument(
        "--subset",
        choices=["gt100k", "lt100k", "none"],
        default="none",
        help="选择样本子集：gt100k(病变像素和>100k)、lt100k(<100k)、none(不读取json，全部样本)"
    )
    p.add_argument(
        "--json",
        type=str,
        default="/path/to/bxg/storage/ReXGroundingCT/dataset.json",
        help="像素统计 json 路径（subset 为 gt100k/lt100k 时需要）"
    )
    return p.parse_args()

def topk_threshold(vol: np.ndarray, ratio: float) -> float:
    """返回使得约 ratio 比例体素落入最高值集合的阈值（仅在>0的体素上计算）"""
    flat = vol.reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return np.inf
    nonzero = finite[finite > 0]
    if nonzero.size == 0:
        return np.inf
    k = max(1, int(np.ceil(ratio * nonzero.size)))
    idx = nonzero.size - k  # 选择前k大的最小值
    thr = float(np.partition(nonzero, idx)[idx])
    return thr

def main():
    args = parse_args()

    if "abs" in args.thresh_mode:
        if not (0.0 <= args.binary_threshold <= 1.0):
            raise ValueError("--binary-threshold 必须在 [0,1]")
    if "rel" in args.thresh_mode:
        if not (0.0 < args.ratio <= 1.0):
            raise ValueError("--ratio 必须在 (0,1]，如 0.2 表示前20%")

    # 仅创建分割输出目录；不再写平均热图
    os.makedirs(args.out_seg, exist_ok=True)

    in_dir = args.in_overlay
    if not os.path.isdir(in_dir):
        raise FileNotFoundError(f"未找到已叠加热图目录: {in_dir}")

    # 优先匹配 *_overlaid_heatmap.nii.gz；若没有则回退所有 nii.gz
    overlay_files = sorted(glob.glob(os.path.join(in_dir, "*_overlaid_heatmap.nii.gz")))
    if not overlay_files:
        overlay_files = sorted(glob.glob(os.path.join(in_dir, "*.nii.gz")))
    if not overlay_files:
        raise FileNotFoundError(f"目录中未找到任何 .nii.gz 文件: {in_dir}")

    # 根据 subset 过滤样本列表
    # 构造 {sample_id: fullpath}
    file_map = {}
    for ofile in overlay_files:
        base = os.path.basename(ofile)
        sid = base.replace("_overlaid_heatmap.nii.gz", "")
        if sid.endswith(".nii.gz"):
            sid = sid[:-7]
        file_map[sid] = ofile

    selected_ids = sorted(file_map.keys())
    if args.subset != "none":
        json_path = args.json
        if not json_path or not os.path.exists(json_path):
            raise FileNotFoundError(f"subset={args.subset} 需要 json 文件，但未找到: {json_path}")
        pixel_sums = load_pixel_sums(json_path)

        # 报告在 json 中缺失的样本（不处理）
        missing = [sid for sid in selected_ids if sid not in pixel_sums]
        if missing:
            print(f"警告：{len(missing)} 个预测样本在 dataset.json 中无像素条目，例如前5个: {missing[:5]}")

        # 仅保留同时存在于 json 的样本
        selected_ids = [sid for sid in selected_ids if sid in pixel_sums]
        if args.subset == "gt100k":
            selected_ids = [sid for sid in selected_ids if pixel_sums[sid] > PIXEL_MIN]
        elif args.subset == "lt100k":
            selected_ids = [sid for sid in selected_ids if pixel_sums[sid] < PIXEL_MIN]

    if not selected_ids:
        print("没有可处理的样本（过滤后为空）。")
        return

    total_samples = len(selected_ids)
    success_samples = 0

    for sid in tqdm(selected_ids, desc="thresholding"):
        ofile = file_map[sid]
        try:
            img = nib.load(ofile)
            avg_vol = img.get_fdata().astype(np.float32)
            affine = img.affine
            header = img.header.copy()

            # 绝对阈值掩码
            mask_abs = None
            if args.thresh_mode in ("abs", "both"):
                mask_abs = (avg_vol >= args.binary_threshold).astype(np.uint8)

            # 相对阈值掩码
            mask_rel = None
            if args.thresh_mode in ("rel", "both"):
                thr_rel = topk_threshold(avg_vol, args.ratio)
                if np.isfinite(thr_rel):
                    mask_rel = (avg_vol >= thr_rel).astype(np.uint8)
                else:
                    mask_rel = np.zeros_like(avg_vol, dtype=np.uint8)

            # 合成最终掩码
            if args.thresh_mode == "abs":
                seg_mask = mask_abs
            elif args.thresh_mode == "rel":
                seg_mask = mask_rel
            else:  # both -> 取交集
                if mask_abs is None or mask_rel is None:
                    seg_mask = np.zeros_like(avg_vol, dtype=np.uint8)
                else:
                    seg_mask = (mask_abs & mask_rel).astype(np.uint8)

            seg_header = header.copy()
            seg_header.set_data_dtype(np.uint8)
            seg_path = os.path.join(args.out_seg, f"{sid}.nii.gz")
            if args.overwrite or (not os.path.exists(seg_path)):
                nib.save(nib.Nifti1Image(seg_mask, affine, seg_header), seg_path)
                success_samples += 1
        except Exception as e:
            print(f"处理 {sid} 出错: {e}")

    print(f"完成：阈值分割 {success_samples}/{total_samples} 个样本")
    print(f"输入平均热图目录 -> {in_dir}")
    print(f"分割输出目录 -> {args.out_seg}")
    if args.subset == "none":
        print("子集选择: 全部样本（不读取 json）")
    elif args.subset == "gt100k":
        print(f"子集选择: 病变体素和 > {PIXEL_MIN}")
    else:
        print(f"子集选择: 病变体素和 < {PIXEL_MIN}")

if __name__ == "__main__":
    main()