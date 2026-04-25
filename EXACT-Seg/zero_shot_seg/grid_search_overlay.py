import os
import sys
import argparse
import subprocess
import numpy as np
import pandas as pd

# 路径配置
OVERLAY_SCRIPT = "/FM_data/bxg/CT_Report/CT_Report9_test/threshold_overlay.py"
CALC_DICE_SCRIPT = "/FM_data/bxg/CT_Report/CT_Report9_test/calc_dice.py"

def arange_inclusive(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("step 必须为正数")
    count = int(np.floor((stop - start) / step + 1e-9)) + 1
    arr = start + np.arange(max(count, 0)) * step
    arr = arr[(arr >= start - 1e-12) & (arr <= stop + 1e-12)]
    return np.round(arr, 10)

def run_overlay(in_overlay: str, out_seg: str, mode: str, abs_thr: float | None, ratio: float | None, subset: str, json_path: str | None, overwrite: bool) -> bool:
    cmd = [
        sys.executable, OVERLAY_SCRIPT,
        "--in-overlay", in_overlay,
        "--out-seg", out_seg,
        "--thresh-mode", mode,
        "--overwrite" if overwrite else "",
        "--subset", subset,
    ]
    # 清理空字符串参数
    cmd = [c for c in cmd if c != ""]
    if mode in ("abs", "both") and abs_thr is not None:
        cmd += ["--binary-threshold", str(float(abs_thr))]
    if mode in ("rel", "both") and ratio is not None:
        cmd += ["--ratio", str(float(ratio))]
    if subset in ("gt100k", "lt100k") and json_path:
        cmd += ["--json", json_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERR] overlay 生成失败(mode={mode}, thr={abs_thr}, ratio={ratio}, subset={subset}): {res.stderr}")
        return False
    if res.stdout:
        print(res.stdout.strip())
    return True

def run_calc_dice(mode: str, pred_dir: str, gt_dir: str, json_path: str | None, threshold: float | None) -> bool:
    """
    调用新的 calc_dice.py 接口：支持 --mode split/all/both 与 --threshold
    结果 CSV 将写在 pred_dir 下。
    """
    cmd = [
        sys.executable, CALC_DICE_SCRIPT,
        "--mode", mode,
        "--pred_dir", pred_dir,
        "--gt_dir", gt_dir,
    ]
    if threshold is not None:
        cmd += ["--threshold", str(float(threshold))]
    if mode in ("split", "both") and json_path:
        cmd += ["--json", json_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERR] 计算Dice失败: {res.stderr}")
        return False
    if res.stdout:
        print(res.stdout.strip())
    return True

def main():
    ap = argparse.ArgumentParser(description="对已叠加平均热图进行阈值网格搜索，随后调用 calc_dice.py 统计。")
    ap.add_argument("--in_overlay", default="/FM_data/bxg/CT_Report/CT_Report9_test/results/covidfull/overlaid_heatmaps_covidfull", help="输入平均热图目录")
    ap.add_argument("--out_seg", default="/FM_data/bxg/CT_Report/CT_Report9_test/results/covidfull/segmentation_results_covidfull", help="阈值分割输出目录")
    ap.add_argument("--subset", choices=["gt100k", "lt100k", "none"], default="none", help="选择子集：gt100k/lt100k/none")
    ap.add_argument("--json", default="/path/to/bxg/storage/ReXGroundingCT/dataset.json", help="像素统计 json 路径（subset 需要）")
    ap.add_argument("--overlay_mode", choices=["abs", "rel", "both"], default="abs", help="叠加热图的阈值模式")
    # 网格参数
    ap.add_argument("--abs-start", type=float, default=0.0)
    ap.add_argument("--abs-stop", type=float, default=0.5)
    ap.add_argument("--abs-step", type=float, default=0.05)
    ap.add_argument("--ratio-start", type=float, default=0.001)
    ap.add_argument("--ratio-stop", type=float, default=0.05)
    ap.add_argument("--ratio-step", type=float, default=0.001)
    ap.add_argument("--overwrite", action="store_true")
    # calc_dice 统计参数
    ap.add_argument("--dice_mode", choices=["split", "all", "both"], default="all", help="calc_dice 统计模式")
    ap.add_argument("--gt_dir", default="/FM_data/bxg/CT_Report/CT_covid_withlabel/visualization_mask", help="GT mask 目录")
    ap.add_argument("--dice_threshold", type=float, default=None, help="calc_dice 中用于二值化预测的阈值(可选)")
    args = ap.parse_args()

    os.makedirs(args.out_seg, exist_ok=True)

    summary = []
    if args.overlay_mode == "abs":
        grid = arange_inclusive(args.abs_start, args.abs_stop, args.abs_step)
        print(f"开始绝对阈值网格搜索，共 {len(grid)} 个阈值；subset={args.subset}")
        for i, thr in enumerate(grid, 1):
            print(f"\n[{i}/{len(grid)}] abs_thr={thr:.6f}")
            ok = run_overlay(args.in_overlay, args.out_seg, "abs", thr, None, args.subset, args.json, args.overwrite)
            if not ok:
                print("  跳过该阈值。")
                continue
            # 生成后，调用 calc_dice 统计（在 out_seg 下输出 CSV）
            ok2 = run_calc_dice(args.dice_mode, args.out_seg, args.gt_dir, args.json if args.dice_mode in ("split", "both") else None, args.dice_threshold)
            summary.append({"overlay_mode": "abs", "thr": thr, "ok": ok and ok2})
    elif args.overlay_mode == "rel":
        grid = arange_inclusive(args.ratio_start, args.ratio_stop, args.ratio_step)
        grid = grid[(grid > 0) & (grid <= 1.0)]
        print(f"开始相对阈值网格搜索，共 {len(grid)} 个比例；subset={args.subset}")
        for i, ratio in enumerate(grid, 1):
            print(f"\n[{i}/{len(grid)}] ratio={ratio:.6f}")
            ok = run_overlay(args.in_overlay, args.out_seg, "rel", None, ratio, args.subset, args.json, args.overwrite)
            if not ok:
                print("  跳过该比例。")
                continue
            ok2 = run_calc_dice(args.dice_mode, args.out_seg, args.gt_dir, args.json if args.dice_mode in ("split", "both") else None, args.dice_threshold)
            summary.append({"overlay_mode": "rel", "ratio": ratio, "ok": ok and ok2})
    else:  # both -> 二维网格：先 abs 后 rel 取交集
        abs_grid = arange_inclusive(args.abs_start, args.abs_stop, args.abs_step)
        rel_grid = arange_inclusive(args.ratio_start, args.ratio_stop, args.ratio_step)
        rel_grid = rel_grid[(rel_grid > 0) & (rel_grid <= 1.0)]
        total = len(abs_grid) * len(rel_grid)
        print(f"开始二维网格搜索，共 {total} 组；subset={args.subset}")
        k = 0
        for thr in abs_grid:
            for ratio in rel_grid:
                k += 1
                print(f"\n[{k}/{total}] abs_thr={thr:.6f}, ratio={ratio:.6f}")
                ok = run_overlay(args.in_overlay, args.out_seg, "both", thr, ratio, args.subset, args.json, args.overwrite)
                if not ok:
                    print("  跳过该组合。")
                    continue
                ok2 = run_calc_dice(args.dice_mode, args.out_seg, args.gt_dir, args.json if args.dice_mode in ("split", "both") else None, args.dice_threshold)
                summary.append({"overlay_mode": "both", "thr": thr, "ratio": ratio, "ok": ok and ok2})

    # 汇总到屏幕（如需CSV可后续扩展）
    if summary:
        df = pd.DataFrame(summary)
        print("\n网格搜索完成，概览：")
        print(df.head())
    else:
        print("未得到任何有效结果。")

if __name__ == "__main__":
    main()
    main()