# runs_rtts_eval12n.py
from pathlib import Path
import multiprocessing as mp
import random, os
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ---------------- config ----------------
MODEL = r"E:\Ivs_FrankGuo\models\yolo12n.pt"
PROJ  = r"E:\Ivs_FrankGuo\Yolo12_Dehazed\runs_rtts_eval12n"
SEED  = 405            # determinism
RECT  = False        # make fixed 640x640 canvases (consistent preview sizes)

DATASETS = {
    "test_original": r"E:\Ivs_FrankGuo\Yolo12_Dehazed\dataset\RTTS-YOLO\RTTS_coco.yaml",
    "test_AOD":      r"E:\Ivs_FrankGuo\Yolo12_Dehazed\dataset\RTTS-YOLO\RTTS_coco_AOD.yaml",
    "test_DCP":      r"E:\Ivs_FrankGuo\Yolo12_Dehazed\dataset\RTTS-YOLO\RTTS_coco_DCP.yaml",
    "test_enhanced": r"E:\Ivs_FrankGuo\Yolo12_Dehazed\dataset\RTTS-YOLO\RTTS_coco_enhanced.yaml",
    "test_FFA":      r"E:\Ivs_FrankGuo\Yolo12_Dehazed\dataset\RTTS-YOLO\RTTS_coco_FFA.yaml",
}
# ----------------------------------------


def set_determinism(seed: int = 0):
    """Best-effort determinism for val pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Torch determinism mainly affects training; harmless to set here.
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def write_latex_table(df: pd.DataFrame, out_path: Path):
    """Emit a compact LaTeX table with main metrics."""
    # order rows as in df (already sorted by mAP50-95 in main())
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{RTTS test results (COCO-pretrained YOLOv12n, imgsz=640, conf=0.001, IoU=0.7).}",
        r"\label{tab:rtts-yolo12n-main}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & mAP@0.5:0.95 & mAP@0.5 & Precision & Recall \\",
        r"\midrule",
    ]
    name_map = {
        "test_original": "Original",
        "test_AOD": "AOD",
        "test_DCP": "DCP",
        "test_enhanced": r"\textbf{Enhanced (ours)}",
        "test_FFA": "FFA",
    }
    # bold best in each column
    best_map5095 = df["mAP50-95"].max()
    best_map50   = df["mAP50"].max()
    best_p       = df["precision"].max()
    best_r       = df["recall"].max()

    def maybe_bold(x, best):
        return r"\textbf{%.3f}" % x if abs(x - best) < 1e-12 or x == best else f"{x:.3f}"

    for _, row in df.iterrows():
        method = name_map.get(row["split"], row["split"])
        lines.append(
            f"{method} & "
            f"{maybe_bold(row['mAP50-95'], best_map5095)} & "
            f"{maybe_bold(row['mAP50'], best_map50)} & "
            f"{maybe_bold(row['precision'], best_p)} & "
            f"{maybe_bold(row['recall'], best_r)} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    set_determinism(SEED)
    model = YOLO(MODEL)
    proj_path = Path(PROJ)
    proj_path.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, yaml in DATASETS.items():
        print(f"\n=== Evaluating: {name} ===")
        metrics = model.val(
            data=yaml,
            split="test",
            imgsz=640,
            batch=16,
            conf=0.001,
            iou=0.7,
            rect=RECT,          # <-- consistent preview size across splits
            seed=SEED,          # <-- deterministic ordering/plots
            workers=0,          # <-- Windows-safe
            device=0,
            save_json=True,
            plots=True,         # saves PR curves, confusion, and val_batch0_pred.jpg
            project=PROJ,
            name=name,
        )
        rows.append({
            "split": name,
            "mAP50-95": float(metrics.box.map),
            "mAP50":    float(metrics.box.map50),
            "mAP75":    float(metrics.box.map75),
            "precision": float(metrics.box.mp),
            "recall":    float(metrics.box.mr),
        })

    # Summaries
    df = pd.DataFrame(rows)
    df = df.sort_values("mAP50-95", ascending=False)
    print("\n=== Summary (sorted by mAP50-95) ===")
    print(df)

    # Save CSVs
    df.to_csv(proj_path / "summary.csv", index=False)

    # Deltas vs Original (if present)
    if "test_original" in df["split"].values:
        base = float(df.loc[df["split"] == "test_original", "mAP50-95"].iloc[0])
        dfx = df.copy()
        dfx["delta_mAP50-95"] = dfx["mAP50-95"] - base
        dfx["delta_%"] = 100.0 * dfx["delta_mAP50-95"] / base
        dfx.to_csv(proj_path / "summary_deltas.csv", index=False)

    # LaTeX table (using the current sort)
    write_latex_table(df, proj_path / "summary_table.tex")
    print(f"\nSaved: {proj_path / 'summary.csv'}")
    print(f"Saved: {proj_path / 'summary_deltas.csv'}")
    print(f"Saved: {proj_path / 'summary_table.tex'}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
