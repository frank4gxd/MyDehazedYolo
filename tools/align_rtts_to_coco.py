"""
Align RTTS 5-class YOLO labels to COCO80 indices.
Default: write to labels_coco (non-destructive). Use --inplace to overwrite with backup.

RTTS names (given): ['bicycle','bus','car','motorbike','person']
COCO indices: person=0, bicycle=1, car=2, motorcycle=3, bus=5
Mapping (RTTS->COCO): {0:1, 1:5, 2:2, 3:3, 4:0}
"""

import argparse
import shutil
from pathlib import Path

MAPPING = {0: 1, 1: 5, 2: 2, 3: 3, 4: 0}  # RTTS idx -> COCO80 idx


def remap_file(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        # 写空文件以保持文件结构一致
        dst.write_text("", encoding="utf-8")
        return
    lines = src.read_text(encoding="utf-8").splitlines()
    out_lines = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        try:
            cls = int(float(parts[0]))  # 兼容有些工具写成 "0.0"
        except Exception:
            # 非法行直接跳过（或可抛异常）
            continue
        if cls not in MAPPING:
            raise ValueError(f"Class id {cls} not in mapping for file: {src}")
        parts[0] = str(MAPPING[cls])
        out_lines.append(" ".join(parts))
    dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=r"E:\Ivs_FrankGuo\Yolo12_Dehazed\dataset\RTTS-YOLO",
        help="Dataset root that contains images/ and labels/",
    )
    ap.add_argument(
        "--inplace", action="store_true", help="Overwrite labels/ in-place (backup to labels_5cls_backup/ first)"
    )
    args = ap.parse_args()

    root = Path(args.root)
    labels_dir = root / "labels"
    if not labels_dir.exists():
        raise SystemExit(f"labels dir not found: {labels_dir}")

    if args.inplace:
        backup = root / "labels_5cls_backup"
        if backup.exists():
            raise SystemExit(f"Backup folder already exists: {backup} (to be safe, stop now)")
        print(f"[INFO] Backing up original labels -> {backup}")
        shutil.copytree(labels_dir, backup)
        out_root = labels_dir  # overwrite in place
    else:
        out_root = root / "labels_coco"
        print(f"[INFO] Non-destructive mode. Writing COCO-indexed labels to: {out_root}")

    splits = ["train", "val", "test"]
    count_files = 0
    for sp in splits:
        src_split = labels_dir / sp
        out_root / sp
        if not src_split.exists():
            print(f"[WARN] Missing split folder: {src_split} (skip)")
            continue
        for txt in src_split.rglob("*.txt"):
            rel = txt.relative_to(labels_dir)
            dst = out_root / rel
            remap_file(txt, dst)
            count_files += 1

    print(f"[DONE] Remapped {count_files} label files using mapping {MAPPING}.")
    if not args.inplace:
        print("       To evaluate with Ultralytics, temporarily rename:")
        print(f"       {out_root}  ->  {root / 'labels'}  (or pass --inplace to overwrite with backup)")


if __name__ == "__main__":
    main()
