# convert_to_paligemma.py
import os, json, yaml, shutil, random
from PIL import Image
from pathlib import Path
from class_map import CLASS_NORMALIZATION, CANONICAL_CLASSES, CANONICAL_TO_IDX

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def normalize_class(raw_name: str):
    """Return canonical class name or None if should be skipped."""
    raw_name = raw_name.strip()
    if raw_name in CLASS_NORMALIZATION:
        return CLASS_NORMALIZATION[raw_name]
    # Fallback: lowercase match
    lower = raw_name.lower()
    for k, v in CLASS_NORMALIZATION.items():
        if k.lower() == lower:
            return v
    print(f"  ⚠️  UNMAPPED class: '{raw_name}' — skipping")
    return None


def yolo_to_loc_tokens(cx: float, cy: float, w: float, h: float):
    """
    Convert YOLO normalized (cx, cy, w, h) → PaliGemma <loc> tokens.
    PaliGemma order:  <locYmin><locXmin><locYmax><locXmax>  in 0–1023 range.
    """
    xmin = cx - w / 2
    ymin = cy - h / 2
    xmax = cx + w / 2
    ymax = cy + h / 2
    # clamp to [0, 1]
    xmin, ymin = max(0.0, xmin), max(0.0, ymin)
    xmax, ymax = min(1.0, xmax), min(1.0, ymax)
    # convert to 1024-bin integers  (PaliGemma: Ymin Xmin Ymax Xmax)
    y0 = int(ymin * 1024)
    x0 = int(xmin * 1024)
    y1 = int(ymax * 1024)
    x1 = int(xmax * 1024)
    # final clamp so we never exceed 1023
    y0, x0 = min(y0, 1023), min(x0, 1023)
    y1, x1 = min(y1, 1023), min(x1, 1023)
    return f"<loc{y0:04d}><loc{x0:04d}><loc{y1:04d}><loc{x1:04d}>"


def load_class_names(dataset_dir: str):
    """Read class names list from data.yaml inside a Roboflow download."""
    for yaml_file in Path(dataset_dir).rglob("data.yaml"):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        if "names" in data:
            names = data["names"]
            if isinstance(names, dict):
                return [names[i] for i in sorted(names.keys())]
            return list(names)
    return []


# ─────────────────────────────────────────────
# Core converter
# ─────────────────────────────────────────────

def convert_dataset(dataset_dir: str, dataset_name: str,
                    output_img_dir: str, class_names: list):
    """
    Walk all splits (train / valid / test) of one Roboflow YOLOv8 dataset.
    Returns a list of PaliGemma-ready dicts.
    """
    examples = []
    skipped_no_label  = 0
    skipped_all_none  = 0

    for split in ("train", "valid", "test"):
        img_dir = Path(dataset_dir) / split / "images"
        lbl_dir = Path(dataset_dir) / split / "labels"

        if not img_dir.exists():
            continue

        img_paths = (list(img_dir.glob("*.jpg"))
                   + list(img_dir.glob("*.jpeg"))
                   + list(img_dir.glob("*.png"))
                   + list(img_dir.glob("*.JPG"))
                   + list(img_dir.glob("*.PNG")))

        print(f"  [{dataset_name}] {split}: {len(img_paths)} images")

        for img_path in img_paths:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                skipped_no_label += 1
                continue

            # ── Load image ──
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"    Bad image {img_path.name}: {e}")
                continue

            # ── Parse YOLO annotations ──
            suffix_parts = []
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_idx = int(parts[0])
                    if cls_idx >= len(class_names):
                        continue
                    raw_name   = class_names[cls_idx]
                    canonical  = normalize_class(raw_name)
                    if canonical is None:
                        continue   # skip non-GHS / background classes

                    cx, cy, w, h = (float(parts[1]), float(parts[2]),
                                    float(parts[3]), float(parts[4]))
                    loc = yolo_to_loc_tokens(cx, cy, w, h)
                    suffix_parts.append(f"{loc} {canonical}")

            if not suffix_parts:
                skipped_all_none += 1
                continue

            # ── Save image to unified output folder ──
            out_name  = f"{dataset_name}_{split}_{img_path.name}"
            out_path  = os.path.join(output_img_dir, out_name)
            # Resize to 224×224 here so training uses consistent resolution
            img.resize((224, 224), Image.BILINEAR).save(out_path, "JPEG", quality=95)

            examples.append({
                "image_path": out_path,
                "prefix":     "detect ghs symbols",
                "suffix":     " ; ".join(suffix_parts),
                "source":     dataset_name,
            })

    print(f"  [{dataset_name}] ✅ {len(examples)} examples "
          f"| skipped: {skipped_no_label} no-label, "
          f"{skipped_all_none} all-background")
    return examples


# ─────────────────────────────────────────────
# Main — merge all 3 datasets
# ─────────────────────────────────────────────

def build_merged_dataset(out_dir="./merged_dataset"):
    os.makedirs(f"{out_dir}/images", exist_ok=True)

    DATASETS = [
        ("./datasets/ds1_gefahren",  "gefahren"),
        ("./datasets/ds2_pictogram", "pictogram"),
        ("./datasets/ds3_ghs",       "ghs"),
    ]

    all_examples = []
    class_dist   = {c: 0 for c in CANONICAL_CLASSES}

    for ds_path, ds_name in DATASETS:
        print(f"\n📂  {ds_name}  ({ds_path})")
        class_names = load_class_names(ds_path)
        if not class_names:
            print("  ❌ data.yaml not found — skipping")
            continue
        print(f"  Classes ({len(class_names)}): {class_names[:5]}{'...' if len(class_names)>5 else ''}")

        examples = convert_dataset(ds_path, ds_name,
                                   f"{out_dir}/images", class_names)
        all_examples.extend(examples)

        # tally distribution
        for ex in examples:
            for part in ex["suffix"].split(";"):
                # label is last token after stripping <loc...> tags
                import re
                label = re.sub(r"<loc\d{4}>", "", part).strip()
                if label in class_dist:
                    class_dist[label] += 1

    # ── Print distribution ──
    print(f"\n{'═'*55}")
    print(f"  TOTAL examples: {len(all_examples)}")
    print(f"\n  Class distribution:")
    print(f"  {'Class':<22} {'Count':>6}  {'Bar'}")
    print(f"  {'-'*50}")
    for cls in CANONICAL_CLASSES:
        count = class_dist[cls]
        bar   = "█" * min(count // 3, 40)
        warn  = " ⚠️  LOW" if count < 30 else ""
        print(f"  {cls:<22} {count:>6}  {bar}{warn}")

    print(f"\n  Classes present:  "
          f"{sum(1 for v in class_dist.values() if v > 0)}/9")

    # ── Shuffle & split 85/15 ──
    random.seed(42)
    random.shuffle(all_examples)
    split_at      = int(len(all_examples) * 0.85)
    train_data    = all_examples[:split_at]
    val_data      = all_examples[split_at:]

    with open(f"{out_dir}/train.json", "w") as f:
        json.dump(train_data, f, indent=2)
    with open(f"{out_dir}/val.json", "w") as f:
        json.dump(val_data, f, indent=2)

    print(f"\n  Train: {len(train_data)}  |  Val: {len(val_data)}")
    print(f"  Saved → {out_dir}/train.json  +  val.json")
    print(f"{'═'*55}\n")
    return train_data, val_data


if __name__ == "__main__":
    train, val = build_merged_dataset()