#!/usr/bin/env python3
# count_yolo_classes.py
"""
Count YOLO-format class instances in a dataset.

- Scans a labels/ directory of .txt files (YOLO format: "cls cx cy w h" per line).
- Reports:
  * total images (optional if --images provided)
  * total label files, empty label files
  * total instances
  * per-class instance counts
  * per-class image counts (how many images contain each class)

You can provide class names from:
  1) a dataset YAML via --data (expects 'names' field), or
  2) a comma-separated list via --names,
  3) otherwise defaults to 5-class RTTS order: bicycle,bus,car,motorbike,person.

Usage examples:
  # Minimal: just labels
  python count_yolo_classes.py --labels "E:/Ivs_FrankGuo/Yolo12_Dehazed/dataset/RTTS-YOLO/labels/train"

  # With images dir (to confirm 3025 items and compute missing-label files)
  python count_yolo_classes.py --labels "E:/.../labels/train" --images "E:/.../images/train"

  # With dataset YAML to get class names
  python count_yolo_classes.py --labels ".../labels/train" --data ".../RTTS_coco.yaml"

  # Override names explicitly (comma-separated)
  python count_yolo_classes.py --labels ".../labels/train" --names "bicycle,bus,car,motorbike,person"

  # Save CSV + JSON summaries
  python count_yolo_classes.py --labels ".../labels/train" --save_csv stats.csv --save_json stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set

# Fallback for names if neither --data nor --names is given (your 5-class study)
DEFAULT_NAMES_5 = ["bicycle", "bus", "car", "motorbike", "person"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_names_from_yaml(yaml_path: Path) -> List[str]:
    try:
        import yaml  # Requires PyYAML
    except Exception as e:
        print(f"[WARN] PyYAML not installed: {e}. Falling back to defaults if needed.", file=sys.stderr)
        return []

    if not yaml_path.exists():
        print(f"[WARN] YAML not found at {yaml_path}.", file=sys.stderr)
        return []
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    # 'names' can be dict or list depending on dataset yaml format
    if isinstance(names, dict):
        # Convert index->name dict to list ordered by key
        names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    if not isinstance(names, list):
        print("[WARN] 'names' in YAML is not a list/dict. Ignoring.", file=sys.stderr)
        return []
    return [str(x) for x in names]


def parse_names_arg(names_arg: str | None) -> List[str]:
    if not names_arg:
        return []
    return [s.strip() for s in names_arg.split(",") if s.strip()]


def iter_label_files(labels_dir: Path) -> List[Path]:
    return sorted(p for p in labels_dir.rglob("*.txt") if p.is_file())


def read_label_file(label_path: Path) -> List[int]:
    """Return list of class ids in this label file. Ignores blank/comment lines."""
    cls_ids = []
    try:
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                try:
                    cls = int(parts[0])
                except Exception:
                    # If malformed, skip this line
                    continue
                cls_ids.append(cls)
    except Exception as e:
        print(f"[WARN] Failed to read {label_path}: {e}", file=sys.stderr)
    return cls_ids


def collect_image_stems(images_dir: Path) -> Set[str]:
    stems = set()
    for p in images_dir.rglob("*"):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            stems.add(p.stem)
    return stems


def main():
    ap = argparse.ArgumentParser(description="Count YOLO-format class instances.")
    ap.add_argument("--labels", type=Path, required=True, help="Path to labels/ dir (e.g., .../labels/train)")
    ap.add_argument("--images", type=Path, default=None, help="Optional images/ dir to confirm image count")
    ap.add_argument("--data", type=Path, default=None, help="Optional dataset YAML to read 'names'")
    ap.add_argument("--names", type=str, default=None, help="Optional comma-separated class names")
    ap.add_argument("--save_csv", type=Path, default=None, help="Optional CSV output path")
    ap.add_argument("--save_json", type=Path, default=None, help="Optional JSON output path")
    ap.add_argument("--strict", action="store_true", help="Error if class id out of range")
    args = ap.parse_args()

    labels_dir: Path = args.labels
    if not labels_dir.exists():
        print(f"[ERROR] Labels dir not found: {labels_dir}", file=sys.stderr)
        sys.exit(1)

    # Resolve class names priority: --names > --data > default
    names: List[str] = []
    names = parse_names_arg(args.names) or names
    if not names and args.data:
        names = load_names_from_yaml(args.data)
    if not names:
        names = DEFAULT_NAMES_5
        print(f"[INFO] Using default class names: {names}", file=sys.stderr)
    num_classes = len(names)

    label_files = iter_label_files(labels_dir)
    if not label_files:
        print(f"[WARN] No .txt labels found in {labels_dir}", file=sys.stderr)

    cls_counts: Counter = Counter()
    cls_image_counts: Counter = Counter()
    total_instances = 0
    empty_files = 0
    stems_with_labels: Set[str] = set()

    # For image-wise class presence
    per_image_classes: Dict[str, Set[int]] = defaultdict(set)

    for lf in label_files:
        cls_ids = read_label_file(lf)
        stem = lf.stem
        if len(cls_ids) == 0:
            empty_files += 1
        else:
            stems_with_labels.add(stem)
        # Instance counts
        for cid in cls_ids:
            if cid < 0 or cid >= num_classes:
                msg = f"[{'ERROR' if args.strict else 'WARN'}] Class id {cid} out of range [0,{num_classes-1}] in {lf}"
                print(msg, file=sys.stderr)
                if args.strict:
                    sys.exit(2)
                # Skip counting if out-of-range, but continue processing file
                continue
            cls_counts[cid] += 1
            total_instances += 1
            per_image_classes[stem].add(cid)

    # Image counts per class
    for stem, class_set in per_image_classes.items():
        for cid in class_set:
            cls_image_counts[cid] += 1

    # If images dir provided, confirm total items and detect missing labels
    n_images = None
    missing_label_stems = set()
    if args.images and args.images.exists():
        image_stems = collect_image_stems(args.images)
        n_images = len(image_stems)
        label_stems = {p.stem for p in label_files}
        missing_label_stems = image_stems - label_stems

    # ---- Print Summary ----
    print("\n=== YOLO Class Statistics ===")
    if n_images is not None:
        print(f"Images (in --images): {n_images}")
    print(f"Label files found: {len(label_files)}")
    if n_images is not None:
        print(f"Images missing label files: {len(missing_label_stems)}")
    print(f"Empty label files: {empty_files}")
    print(f"Total instances: {total_instances}\n")

    # Per-class table
    header = f"{'ID':>3}  {'Class':<20}  {'Instances':>10}  {'Images w/ class':>15}"
    print(header)
    print("-" * len(header))
    for cid in range(num_classes):
        cname = names[cid] if cid < len(names) else f"class_{cid}"
        inst = cls_counts.get(cid, 0)
        imgc = cls_image_counts.get(cid, 0)
        print(f"{cid:>3}  {cname:<20}  {inst:>10}  {imgc:>15}")

    # ---- Save CSV/JSON if requested ----
    if args.save_csv:
        try:
            import csv
            with args.save_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["class_id", "class_name", "instances", "images_with_class"])
                for cid in range(num_classes):
                    cname = names[cid] if cid < len(names) else f"class_{cid}"
                    w.writerow([cid, cname, cls_counts.get(cid, 0), cls_image_counts.get(cid, 0)])
            print(f"\n[OK] CSV saved to {args.save_csv}")
        except Exception as e:
            print(f"[WARN] Failed to save CSV: {e}", file=sys.stderr)

    if args.save_json:
        out = {
            "num_classes": num_classes,
            "class_names": names,
            "summary": {
                "images": n_images,
                "label_files": len(label_files),
                "images_missing_labels": len(missing_label_stems) if n_images is not None else None,
                "empty_label_files": empty_files,
                "total_instances": total_instances,
            },
            "per_class": [
                {
                    "class_id": cid,
                    "class_name": names[cid] if cid < len(names) else f"class_{cid}",
                    "instances": cls_counts.get(cid, 0),
                    "images_with_class": cls_image_counts.get(cid, 0),
                }
                for cid in range(num_classes)
            ],
        }
        try:
            with args.save_json.open("w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print(f"[OK] JSON saved to {args.save_json}")
        except Exception as e:
            print(f"[WARN] Failed to save JSON: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
