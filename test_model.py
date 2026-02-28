# test_model.py
import torch, re
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import PeftModel

# ── Config ──
BASE_MODEL   = "google/paligemma2-3b-pt-224"
ADAPTER_PATH = "./labellens_model"

# ── Load fine-tuned model ──
print("Loading fine-tuned model...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
base = PaliGemmaForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
model     = PeftModel.from_pretrained(base, ADAPTER_PATH)
model.eval()
processor = PaliGemmaProcessor.from_pretrained(BASE_MODEL)
print("✅ Model loaded\n")

# ── Detection function ──
def detect_ghs(image_path: str) -> list[dict]:
    """Run GHS detection on any image. Returns list of detections."""
    img = Image.open(image_path).convert("RGB")
    img_224 = img.resize((224, 224))

    inputs = processor(
        images=img_224,
        text="<image> detect ghs symbols",
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,        # greedy = stable for detection
        )

    # Decode only the new tokens (strip the input prompt)
    input_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][input_len:]
    raw = processor.decode(new_tokens, skip_special_tokens=True).strip()

    print(f"  Raw model output: '{raw}'")

    # Parse detections
    detections = []
    for part in raw.split(";"):
        part = part.strip()
        locs = re.findall(r"<loc(\d{4})>", part)
        label = re.sub(r"<loc\d{4}>", "", part).strip()
        if len(locs) == 4 and label:
            ymin, xmin, ymax, xmax = [int(x) for x in locs]
            detections.append({
                "label": label,
                "loc":   (ymin, xmin, ymax, xmax),  # 0-1023 range
            })
    return detections, raw


def draw_detections(image_path: str, detections: list, save_path: str):
    """Draw bounding boxes on the original image and save it."""
    img  = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    COLORS = {
        "explosive":      "#FF0000",
        "flammable":      "#FF6600",
        "oxidizer":       "#0066FF",
        "compressed_gas": "#00CCFF",
        "corrosive":      "#FF9900",
        "toxic":          "#9900CC",
        "harmful":        "#FFCC00",
        "health_hazard":  "#FF00CC",
        "environmental":  "#00AA44",
    }

    for det in detections:
        ymin, xmin, ymax, xmax = det["loc"]
        # Convert from 1024-bin back to pixel coords
        x0 = int(xmin / 1024 * w)
        y0 = int(ymin / 1024 * h)
        x1 = int(xmax / 1024 * w)
        y1 = int(ymax / 1024 * h)

        color = COLORS.get(det["label"], "#FFFFFF")
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.rectangle([x0, y0 - 20, x0 + len(det["label"]) * 8, y0],
                       fill=color)
        draw.text((x0 + 2, y0 - 18), det["label"], fill="black")

    img.save(save_path)
    print(f"  Saved → {save_path}")


# ── Test on your validation images ──
import json, os, random

print("=" * 55)
print("TEST 1: Run on 5 random validation images")
print("=" * 55)

with open("./merged_dataset/val.json") as f:
    val_data = json.load(f)

# Pick 5 random samples
samples = random.sample(val_data, min(5, len(val_data)))

for i, item in enumerate(samples):
    print(f"\nImage {i+1}: {os.path.basename(item['image_path'])}")
    print(f"  Expected: {item['suffix'][:80]}...")

    detections, raw = detect_ghs(item["image_path"])

    # Compare expected vs detected labels
    expected_labels = set(re.sub(r"<loc\d{4}>", "", p).strip()
                          for p in item["suffix"].split(";"))
    detected_labels = set(d["label"] for d in detections)

    match = "✅" if detected_labels == expected_labels else "⚠️ "
    print(f"  Expected labels: {expected_labels}")
    print(f"  Detected labels: {detected_labels}  {match}")

    # Save annotated image
    save_path = f"./test_output_{i+1}.jpg"
    draw_detections(item["image_path"], detections, save_path)

# ── Test on a real chemical label (if you have one) ──
print("\n" + "=" * 55)
print("TEST 2: Accuracy on full validation set (first 50)")
print("=" * 55)

correct = 0
total   = 0
class_correct = {}
class_total   = {}

for item in val_data[:50]:
    total += 1
    detections, _ = detect_ghs(item["image_path"])

    expected = set(re.sub(r"<loc\d{4}>", "", p).strip()
                   for p in item["suffix"].split(";"))
    detected = set(d["label"] for d in detections)

    # Per-class tracking
    for cls in expected:
        class_total[cls]   = class_total.get(cls, 0) + 1
        if cls in detected:
            class_correct[cls] = class_correct.get(cls, 0) + 1

    if detected == expected:
        correct += 1

print(f"\nExact match accuracy: {correct}/{total}  ({100*correct/total:.1f}%)")
print(f"\nPer-class recall:")
for cls in sorted(class_total.keys()):
    c = class_correct.get(cls, 0)
    t = class_total[cls]
    bar = "█" * int(c/t * 20) if t > 0 else ""
    print(f"  {cls:<22} {c:3}/{t:3}  {bar}")