import sys, re
from pathlib import Path

# Remove trailing "_AOD-Net" or "_AOD-net" before the extension
SUFFIX_RE = re.compile(r"_AOD-[Nn]et$")            # end of stem
VALID_EXTS = {".jpg", ".jpeg", ".png"}             # case-insensitive

def main(img_dir: Path) -> int:
    if not img_dir.is_dir():
        print(f"[ERR] Not a directory: {img_dir}")
        return 1

    changed = 0
    total = 0

    for p in img_dir.iterdir():
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext not in VALID_EXTS:
            continue
        total += 1

        stem = p.stem
        new_stem = SUFFIX_RE.sub("", stem)
        if new_stem != stem:
            new_path = p.with_name(new_stem + p.suffix)
            if new_path.exists():
                print(f"[SKIP] Target exists: {new_path.name}")
                continue
            print(f"{p.name}  ->  {new_path.name}")
            p.rename(new_path)
            changed += 1

    # Clear YOLO cache so labels are rescanned: images/<split> -> labels/<split>.cache
    labels_dir = img_dir.parent.parent / "labels" / img_dir.name
    cache = labels_dir.with_suffix(".cache")
    try:
        cache.unlink()
        print(f"[OK] Removed cache: {cache}")
    except FileNotFoundError:
        pass

    if changed == 0:
        # Help debug if nothing matched
        examples = [p.name for p in img_dir.iterdir()
                    if p.is_file() and any(s in p.name for s in ["AOD", "aod"])][:10]
        if examples:
            print("[INFO] No renames performed. Examples in folder:")
            for name in examples:
                print("  -", name)

    print(f"[DONE] Scanned {total} image(s), renamed {changed}.")
    return 0

if __name__ == "__main__":
    default = r"E:\Ivs_FrankGuo\Yolo12_Dehazed\dataset\RTTS-YOLO\images\test_AOD"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(default)
    raise SystemExit(main(path))
