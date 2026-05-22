import argparse
import os
import re
import nibabel as nib
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate disease heatmaps into one overlaid heatmap per sample.")
    parser.add_argument(
        "--input-root",
        type=str,
        required=True,
        help="Directory containing sample folders (or a single flat sample folder) of heatmaps.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for aggregated per-sample heatmaps.",
    )
    parser.add_argument(
        "--res",
        choices=["high-res", "low-res"],
        default="high-res",
        help="Heatmap resolution suffix to match.",
    )
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Use *_pred.nii.gz instead of *_combined_pred.nii.gz.",
    )
    parser.add_argument(
        "--diseases",
        type=str,
        default="",
        help="Comma-separated disease names. Empty means using all diseases found in each sample.",
    )
    parser.add_argument(
        "--aggregate",
        choices=["mean", "sum"],
        default="mean",
        help="Aggregation operation across selected diseases.",
    )
    return parser.parse_args()


def parse_disease_list(diseases_arg: str):
    if not diseases_arg:
        return None
    names = [d.strip() for d in diseases_arg.split(",") if d.strip()]
    return set(names) if names else None


def build_pattern(res: str, use_combined: bool):
    suffix = f"{res}_combined_pred" if use_combined else f"{res}_pred"
    return re.compile(rf"^(?P<disease>.+?)_GT\d+_PD\d+_{re.escape(suffix)}\.nii\.gz$")


def process_sample_dir(sample_dir: str, out_dir: str, pattern: re.Pattern, allowed_diseases, aggregate: str) -> bool:
    files = sorted(os.listdir(sample_dir))
    sum_vol = None
    count = 0
    ref_img = None

    for fname in files:
        # if not fname.endswith("_low-res_combined_pred.nii.gz"):
        #     continue
        m = pattern.match(fname)
        if not m:
            continue
        disease = m.group("disease")
        # print(disease)
        if allowed_diseases is not None and disease not in allowed_diseases:
            continue

        fpath = os.path.join(sample_dir, fname)
        img = nib.load(fpath)
        arr = img.get_fdata().astype(np.float32)

        if sum_vol is None:
            sum_vol = np.zeros_like(arr, dtype=np.float32)
            ref_img = img
        elif sum_vol.shape != arr.shape:
            print(f"[WARN] 形状不一致，跳过: {fpath} (got {arr.shape}, expect {sum_vol.shape})")
            continue

        sum_vol += arr
        count += 1

    if count == 0:
        print(f"[INFO] 样本无可用热图，跳过: {sample_dir}")
        return False

    if aggregate == "mean":
        out_vol = sum_vol / count
    else:
        out_vol = sum_vol

    sample_name = os.path.basename(sample_dir.rstrip("/"))
    out_path = os.path.join(out_dir, f"{sample_name}_overlaid_heatmap.nii.gz")
    out_img = nib.Nifti1Image(out_vol, ref_img.affine, ref_img.header)
    nib.save(out_img, out_path)
    print(f"[OK] {sample_name}: 聚合 {count} 个疾病热图 -> {out_path}")
    return True


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pattern = build_pattern(args.res, use_combined=not args.no_combined)
    allowed_diseases = parse_disease_list(args.diseases)

    entries = sorted(os.listdir(args.input_root))
    has_subdirs = any(os.path.isdir(os.path.join(args.input_root, e)) for e in entries)

    processed = 0
    if has_subdirs:
        for e in entries:
            sample_dir = os.path.join(args.input_root, e)
            if os.path.isdir(sample_dir):
                processed |= process_sample_dir(sample_dir, args.output_dir, pattern, allowed_diseases, args.aggregate)
    else:
        processed |= process_sample_dir(args.input_root, args.output_dir, pattern, allowed_diseases, args.aggregate)

    if not processed:
        print("[DONE] 未生成任何叠加热图，请检查文件命名与匹配模式。")
    else:
        print("[DONE] 叠加热图已全部生成。")

if __name__ == "__main__":
    main()