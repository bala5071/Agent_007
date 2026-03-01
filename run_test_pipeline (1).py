import json
import os
import re
import textwrap
import time
import threading
from pathlib import Path

import torch
from PIL import Image

# ── Voice engine (pyttsx3 — offline, no API key needed) ────────────────────
try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty("rate", 160)   # slightly slower for clarity
    _tts_engine.setProperty("volume", 1.0)
    HAS_TTS = True
    print("Voice engine: pyttsx3 ready")
except ImportError:
    HAS_TTS = False
    print("WARNING: pyttsx3 not installed. Run: pip install pyttsx3")
    print("         Voice output will be skipped.\n")


def speak_async(text: str):
    """Speak text in a background thread so it doesn't block pipeline."""
    if not HAS_TTS or not text:
        return
    def _speak():
        try:
            _tts_engine.say(text)
            _tts_engine.runAndWait()
        except Exception as e:
            print(f"  [TTS error: {e}]")
    t = threading.Thread(target=_speak, daemon=True)
    t.start()
    return t  # caller can .join() if they want to wait


# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
PALIGEMMA_BASE  = "google/paligemma2-3b-pt-224"
PG_ADAPTER      = BASE_DIR / "labellens_model"
GEMMA_BASE      = "google/gemma-3-4b-it"
GEMMA_ADAPTER   = BASE_DIR / "lora_adapter"
TEST_DIR        = BASE_DIR / "merged_dataset" / "test"
ANNOT_FILE      = TEST_DIR / "_annotations.test.jsonl"
OUTPUT_DIR      = BASE_DIR / "test_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Label bridge ───────────────────────────────────────────────────────────
CANONICAL_TO_DISPLAY = {
    "explosive":      "GHS01-Explosive (Exploding Bomb)",
    "flammable":      "GHS02-Flammable",
    "oxidizer":       "GHS03-Oxidizer (Flame Over Circle)",
    "compressed_gas": "GHS04-Compressed Gas (Gas Cylinder)",
    "corrosive":      "GHS05-Corrosive",
    "toxic":          "GHS06-Toxic (Skull & Crossbones)",
    "harmful":        "GHS07-Harmful / Irritant (Exclamation Mark)",
    "health_hazard":  "GHS08-Serious Health Hazard",
    "environmental":  "GHS09-Environmental Hazard",
}

GT_TO_CANONICAL = {
    "GHS_Symbol_EXPLODING_BOMB":       "explosive",
    "GHS_Symbol_FLAME":                "flammable",
    "GHS_Symbol_FLAME_OVER_CIRCLE":    "oxidizer",
    "GHS_Symbol_GAS_CYLINDER":         "compressed_gas",
    "GHS_Symbol_CORROSION":            "corrosive",
    "GHS_Symbol_SKULL_AND_CROSSBONES": "toxic",
    "GHS_Symbol_EXCLAMATION_MARK":     "harmful",
    "GHS_Symbol_HEALTH_HAZARD":        "health_hazard",
    "GHS_Symbol_ENVIRONMENT":          "environmental",
}

SYSTEM_PROMPT = (
    "You are an on-device safety assistant for industrial workers. "
    "Given detected GHS hazard symbols (and optionally OCR text from a product label), "
    "provide a structured safety brief with: severity level, hazard summary, required PPE, "
    "step-by-step handling SOP, storage requirements, emergency/first-aid procedures, "
    "autonomous safety actions, and a short spoken voice script (under 40 words). "
    "Be concise, factual, and prioritize worker safety."
)

# ── Load models ─────────────────────────────────────────────────────────────
def load_paligemma():
    from transformers import BitsAndBytesConfig, PaliGemmaForConditionalGeneration, PaliGemmaProcessor
    from peft import PeftModel

    print("Loading PaliGemma (Agent_007/labellens_model)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = PaliGemmaForConditionalGeneration.from_pretrained(
        PALIGEMMA_BASE,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, str(PG_ADAPTER))
    model.eval()
    model.config.use_cache = True
    processor = PaliGemmaProcessor.from_pretrained(PALIGEMMA_BASE)
    print("  PaliGemma ready.\n")
    return model, processor


def load_gemma():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print("Loading Gemma (lora_adapter / gemma-3-4b-it)...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        GEMMA_BASE,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base, str(GEMMA_ADAPTER))
    model.eval()
    model.config.use_cache = True
    tokenizer = AutoTokenizer.from_pretrained(str(GEMMA_ADAPTER))
    print("  Gemma ready.\n")
    return model, tokenizer


# ── PaliGemma inference ─────────────────────────────────────────────────────
def detect_symbols(pg_model, pg_proc, image: Image.Image) -> tuple[list[dict], str]:
    img_224 = image.resize((224, 224))
    inputs = pg_proc(
        images=img_224,
        text="<image> detect ghs symbols",
        return_tensors="pt",
    ).to(pg_model.device, dtype=torch.bfloat16)

    with torch.no_grad():
        output_ids = pg_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    raw = pg_proc.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

    detections = []
    for part in raw.split(";"):
        part = part.strip()
        locs = re.findall(r"<loc(\d{4})>", part)
        label = re.sub(r"<loc\d{4}>", "", part).strip()
        if len(locs) == 4 and label:
            detections.append({
                "label": label,
                "loc": tuple(int(x) for x in locs),
            })
    return detections, raw


# ── Gemma inference ─────────────────────────────────────────────────────────
def generate_brief(gm_model, gm_tok, symbols: list[str]) -> str:
    if not symbols:
        return "No GHS hazard symbols detected. Area appears safe."

    display_symbols = [CANONICAL_TO_DISPLAY.get(s, s) for s in symbols]
    det_str = ", ".join(f"{s} (detected)" for s in display_symbols)

    user_content = (
        f"Symbols: {det_str}\n"
        "Provide: hazards, PPE, handling SOP, storage, emergency/first-aid, "
        "autonomous safety actions, and a short voice script."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    prompt = gm_tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = gm_tok(prompt, return_tensors="pt").to(gm_model.device)

    with torch.no_grad():
        output_ids = gm_model.generate(
            **inputs,
            max_new_tokens=600,
            do_sample=False,
            repetition_penalty=1.1,
        )

    response = gm_tok.decode(
        output_ids[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def extract_voice_script(brief: str) -> str:
    """
    Extract only the quoted voice script from the Gemma safety brief.
    Handles: VOICE SCRIPT\n"text here"  OR  8) "text"  OR  fallback
    """
    # Try: 8) or VOICE SCRIPT header followed by quoted or unquoted text
    match = re.search(
        r'(?:8\s*[\.\):]|VOICE\s*SCRIPT)[:\s]*["\']?\s*(.+?)\s*["\']?\s*$',
        brief, re.DOTALL | re.IGNORECASE,
    )
    if match:
        text = match.group(1).strip().strip("\"'")
        # Take only first 1-2 sentences (keep under 40 words)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = ""
        for s in sentences:
            if len((result + " " + s).split()) > 45:
                break
            result = (result + " " + s).strip()
        return result if result else text[:200]

    # Fallback: find any long quoted string
    match = re.search(r'"([^"]{20,})"', brief)
    if match:
        return match.group(1).strip()

    # Last fallback: last 2 non-empty lines
    lines = [l.strip() for l in brief.split('\n') if l.strip()]
    return " ".join(lines[-2:]).strip('"') if lines else ""


def parse_gt_labels(suffix: str) -> set[str]:
    labels = set()
    for part in suffix.split(";"):
        label = re.sub(r"<loc\d{4}>", "", part).strip()
        if label in GT_TO_CANONICAL:
            labels.add(GT_TO_CANONICAL[label])
        elif label:
            labels.add(label.lower().replace(" ", "_"))
    return labels


# ── Main test loop ───────────────────────────────────────────────────────────
def main():
    with open(ANNOT_FILE, encoding="utf-8") as f:
        annotations = [json.loads(l) for l in f if l.strip()]

    labeled = [a for a in annotations if a.get("suffix", "").strip()]
    test_samples = labeled[:10]

    print(f"Selected {len(test_samples)} test images from {TEST_DIR}\n")
    print("=" * 70)

    pg_model, pg_proc = load_paligemma()
    gm_model, gm_tok  = load_gemma()

    results = []
    correct = 0

    for idx, sample in enumerate(test_samples, 1):
        img_path = TEST_DIR / sample["image"]
        gt_labels = parse_gt_labels(sample["suffix"])

        print(f"\n{'='*70}")
        print(f"Image {idx}/10: {sample['image']}")
        print(f"  Path : {img_path}")
        print(f"  GT   : {sorted(gt_labels)}")

        image = Image.open(img_path).convert("RGB")
        t0 = time.time()

        # ── Stage 1: PaliGemma detection ──
        detections, raw_pg = detect_symbols(pg_model, pg_proc, image)
        detected_labels = list({d["label"] for d in detections})

        print(f"  PaliGemma raw  : '{raw_pg}'")
        print(f"  Detected labels: {sorted(detected_labels)}")

        match_flag = set(detected_labels) == gt_labels
        recall_ok  = gt_labels.issubset(set(detected_labels))
        if match_flag:
            correct += 1
        # USE ASCII-SAFE status markers instead of emoji
        status = "[EXACT]" if match_flag else ("[PARTIAL]" if recall_ok else "[MISS]")
        print(f"  Detection      : {status}")

        # ── Stage 2: Gemma safety brief ──
        brief = generate_brief(gm_model, gm_tok, detected_labels)
        voice = extract_voice_script(brief)
        elapsed = time.time() - t0

        print(f"\n  -- Safety Brief --")
        for line in textwrap.wrap(brief, width=68):
            print(f"  {line}")

        print(f"\n  -- Voice Script --")
        print(f'  "{voice}"')
        print(f"\n  Latency: {elapsed:.1f}s")

        # ── Stage 3: Voice output ──
        print(f"  Speaking voice script...")
        tts_thread = speak_async(voice)
        # Wait for speech to finish before moving to next image
        if tts_thread:
            tts_thread.join(timeout=15)

        # Save result
        result = {
            "image": sample["image"],
            "image_path": str(img_path),
            "gt_labels": sorted(gt_labels),
            "detected_labels": sorted(detected_labels),
            "paligemma_raw": raw_pg,
            "detection_status": status,
            "safety_brief": brief,
            "voice_script": voice,
            "latency_s": round(elapsed, 2),
        }
        results.append(result)

        # ── FIX: open with encoding="utf-8" to avoid CP1252 errors ──
        result_file = OUTPUT_DIR / f"result_{idx:02d}_{Path(sample['image']).stem[:30]}.txt"
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(f"Image: {sample['image']}\n")
            f.write(f"Ground Truth: {sorted(gt_labels)}\n")
            f.write(f"Detected:     {sorted(detected_labels)}\n")
            f.write(f"Status:       {status}\n")
            f.write(f"Latency:      {elapsed:.1f}s\n")
            f.write("\n-- Safety Brief --\n")
            f.write(brief + "\n")
            f.write("\n-- Voice Script --\n")
            f.write(f'"{voice}"\n')

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Image':<45} {'GT Labels':<25} {'Detected':<25} Status")
    print("-" * 120)
    for r in results:
        gt_str  = ", ".join(r["gt_labels"])[:22]
        det_str = ", ".join(r["detected_labels"])[:22]
        name    = r["image"][:42]
        print(f"{name:<45} {gt_str:<25} {det_str:<25} {r['detection_status']}")

    print(f"\nExact-match detection accuracy: {correct}/10 ({correct*10}%)")
    print(f"\nAll results saved to: {OUTPUT_DIR}/")

    # Save full JSON results — also UTF-8
    with open(OUTPUT_DIR / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Full JSON: {OUTPUT_DIR / 'all_results.json'}")


if __name__ == "__main__":
    main()