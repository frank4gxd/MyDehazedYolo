#!/usr/bin/env python3
"""
Convert RTTS (VOC-style) to YOLO format with random split (e.g., 70/20/10)

- Creates images/{train,val,test} and labels/{train,val,test}
- Generates data.yaml with class names inferred from XML (or user-specified).
"""

from __future__ import annotations

import argparse
import random
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from shutil import copy2

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG"]


def parse_args():
    ap = argparse.ArgumentParser(description="Convert RTTS VOC to YOLO with random split")
    ap.add_argument("rtts_root", type=str, help="Path to RTTS root (contains JPEGImages/, Annotations/)")
    ap.add_argument("out_root", type=str, help="Output root for YOLO dataset")
    ap.add_argument(
        "--splits",
        type=float,
        nargs=3,
        default=[0.7, 0.2, 0.1],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split ratios for train/val/test, must sum≈1.0 (default 0.7 0.2 0.1)",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for split")
    ap.add_argument("--names", type=str, nargs="*", default=None, help="Explicit class names order (optional)")
    ap.add_argument(
        "--ignore-imagesets",
        action="store_true",
        help="Ignore ImageSets/Main even if present (default: True for random split)",
    )
    return ap.parse_args()


def find_image_for_stem(img_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def parse_voc_xml(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    W = int(size.find("width").text)
    H = int(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        cls = obj.find("name").text.strip()
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        xmin = max(0.0, min(xmin, W - 1))
        xmax = max(0.0, min(xmax, W - 1))
        ymin = max(0.0, min(ymin, H - 1))
        ymax = max(0.0, min(ymax, H - 1))
        if xmax <= xmin or ymax <= ymin:
            continue
        cx = ((xmin + xmax) / 2.0) / W
        cy = ((ymin + ymax) / 2.0) / H
        ww = (xmax - xmin) / W
        hh = (ymax - ymin) / H
        boxes.append((cls, clamp01(cx), clamp01(cy), clamp01(ww), clamp01(hh)))
    return W, H, boxes


def robust_split(items: list[Path], ratios: tuple[float, float, float]) -> dict:
    n = len(items)
    r_train, r_val, r_test = ratios
    if r_train + r_val + r_test <= 0:
        raise ValueError("Invalid --splits; sum must be > 0")
    s = r_train + r_val + r_test
    r_train, r_val, r_test = r_train / s, r_val / s, r_test / s
    n_train = int(n * r_train)
    n_val = int(n * r_val)
    n_test = n - n_train - n_val
    # 确保每份至少 1 张（若可能）
    if n >= 3:
        if n_train == 0:
            n_train, n_test = 1, n_test - 1
        if n_val == 0:
            n_val, n_test = 1, n_test - 1
        if n_test == 0:
            n_test, n_val = 1, n_val - 1
    return {"train": items[:n_train], "val": items[n_train : n_train + n_val], "test": items[n_train + n_val :]}


def main():
    args = parse_args()
    random.seed(args.seed)

    rtts = Path(args.rtts_root)
    out = Path(args.out_root)
    ann_dir = rtts / "Annotations"
    img_dir = rtts / "JPEGImages"
    if not ann_dir.exists() or not img_dir.exists():
        print(f"[ERR] Not found: {ann_dir} or {img_dir}", file=sys.stderr)
        sys.exit(1)

    xmls = sorted(ann_dir.glob("*.xml"))
    if not xmls:
        print(f"[ERR] No XML files under {ann_dir}", file=sys.stderr)
        sys.exit(1)

    # 类别统计/顺序
    cls_counter = Counter()
    for xp in xmls:
        _, _, boxes = parse_voc_xml(xp)
        for c, *_ in boxes:
            cls_counter[c] += 1
    names = args.names if args.names else sorted(cls_counter.keys())
    name2id = {n: i for i, n in enumerate(names)}

    # 输出目录
    for sp in ["train", "val", "test"]:
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)

    # 随机划分
    random.shuffle(xmls)
    split_xmls = robust_split(xmls, tuple(args.splits))
    print(
        f"[INFO] Random split -> train={len(split_xmls['train'])}, val={len(split_xmls['val'])}, test={len(split_xmls['test'])}"
    )

    # 转换
    total_objs, skipped_no_image = 0, 0
    counts = defaultdict(int)
    for split, xlist in split_xmls.items():
        for xp in xlist:
            stem = xp.stem
            imp = find_image_for_stem(img_dir, stem)
            if imp is None:
                skipped_no_image += 1
                continue
            dst_img = out / "images" / split / imp.name
            if not dst_img.exists():
                copy2(imp, dst_img)
            _, _, boxes = parse_voc_xml(xp)
            dst_lab = out / "labels" / split / f"{stem}.txt"
            with open(dst_lab, "w", encoding="utf-8") as f:
                for cls, x, y, w, h in boxes:
                    if cls not in name2id:
                        continue
                    f.write(f"{name2id[cls]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                    total_objs += 1
            counts[split] += 1
            if not boxes:
                dst_lab.touch(exist_ok=True)

    # 写 data.yaml
    data_yaml = out / "RTTS.yaml"
    names_list = "[" + ", ".join(f"'{n}'" for n in names) + "]"
    data_yaml.write_text(
        f"""# Auto-generated RTTS → YOLO (random split)
path: {out.as_posix()}
train: images/train
val: images/val
test: images/test
names: {names_list}
""",
        encoding="utf-8",
    )

    print("\n==== Summary ====")
    print("Classes:", names)
    print("Images converted:", dict(counts))
    print("Total objects:", total_objs)
    print("Missing images:", skipped_no_image)
    print("data.yaml ->", data_yaml.as_posix())


if __name__ == "__main__":
    main()
