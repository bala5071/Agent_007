# augment_compressed_gas.py
import json, os, random
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pathlib import Path

TRAIN_JSON  = "./merged_dataset/train.json"
OUTPUT_DIR  = "./merged_dataset/images"
TARGET      = 120   # boost compressed_gas to this count

with open(TRAIN_JSON) as f:
    train_data = json.load(f)

# Find all compressed_gas examples
cg_examples = [
    ex for ex in train_data
    if "compressed_gas" in ex["suffix"]
]
print(f"Found {len(cg_examples)} compressed_gas examples → targeting {TARGET}")

def augment_image(img: Image.Image, seed: int) -> Image.Image:
    """Apply a random but deterministic augmentation."""
    random.seed(seed)
    ops = random.sample(range(6), k=random.randint(2, 4))

    for op in ops:
        if op == 0:   # brightness
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.5))
        elif op == 1: # contrast
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.4))
        elif op == 2: # slight blur
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))
        elif op == 3: # horizontal flip  (bbox fix: mirror x coords)
            img = ImageOps.mirror(img)
        elif op == 4: # small rotation
            img = img.rotate(random.uniform(-12, 12), fillcolor=(220, 220, 220))
        elif op == 5: # colour jitter
            img = ImageEnhance.Color(img).enhance(random.uniform(0.6, 1.4))
    return img

def mirror_suffix(suffix: str) -> str:
    """
    When we horizontally flip an image, x-coordinates must be mirrored.
    PaliGemma loc tokens: <locYmin><locXmin><locYmax><locXmax>
    Mirror: new_Xmin = 1023 - old_Xmax,  new_Xmax = 1023 - old_Xmin
    """
    import re
    new_parts = []
    for part in suffix.split(";"):
        locs = re.findall(r"<loc(\d{4})>", part)
        label = re.sub(r"<loc\d{4}>", "", part).strip()
        if len(locs) == 4:
            ymin, xmin, ymax, xmax = [int(x) for x in locs]
            new_xmin = 1023 - xmax
            new_xmax = 1023 - xmin
            new_parts.append(
                f"<loc{ymin:04d}><loc{new_xmin:04d}>"
                f"<loc{ymax:04d}><loc{new_xmax:04d}> {label}"
            )
        else:
            new_parts.append(part.strip())
    return " ; ".join(new_parts)

# Generate augmented examples
new_examples = []
needed       = TARGET - len(cg_examples)
seed         = 1000

while len(new_examples) < needed:
    base = random.choice(cg_examples)
    img  = Image.open(base["image_path"]).convert("RGB")

    do_flip = (seed % 3 == 0)   # flip every 3rd augmentation
    aug     = augment_image(img, seed)
    suffix  = mirror_suffix(base["suffix"]) if do_flip else base["suffix"]

    out_name = f"aug_cg_{seed:05d}.jpg"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    aug.save(out_path, "JPEG", quality=92)

    new_examples.append({
        "image_path": out_path,
        "prefix":     "detect ghs symbols",
        "suffix":     suffix,
        "source":     "augmented_compressed_gas",
    })
    seed += 1

print(f"Generated {len(new_examples)} new compressed_gas examples")

# Add to train.json and save
train_data.extend(new_examples)
random.seed(42)
random.shuffle(train_data)

with open(TRAIN_JSON, "w") as f:
    json.dump(train_data, f, indent=2)

# Verify new count
cg_count = sum(1 for ex in train_data if "compressed_gas" in ex["suffix"])
print(f"compressed_gas in train: {cg_count} ✅")
print(f"Total train examples: {len(train_data)}")