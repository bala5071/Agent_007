"""
live_demo.py
────────────
LabelLens Live Demo — Webcam + PaliGemma + Gemma + Voice

Show a product label (from phone/tablet/printed) to your laptop webcam.
Press SPACE to scan. Hear the safety warning. See results on screen + console.

Controls:
    SPACE  = Scan current frame
    M      = Toggle mode (SCAN / DELIVERY)
    R      = Print full compliance report to console
    Q      = Quit

Run:
    python live_demo.py
"""

import json
import os
import re
import sys
import textwrap
import time
import threading
import datetime
from pathlib import Path

import cv2
import torch
from PIL import Image

# ── Voice engine ───────────────────────────────────────────────────────────
try:
    import pyttsx3

    _tts_lock = threading.Lock()

    def _create_engine():
        engine = pyttsx3.init()
        engine.setProperty("rate", 155)
        engine.setProperty("volume", 1.0)
        # Try to pick a clearer voice if available
        voices = engine.getProperty("voices")
        for v in voices:
            if "zira" in v.name.lower() or "david" in v.name.lower():
                engine.setProperty("voice", v.id)
                break
        return engine

    HAS_TTS = True
    print("[OK] pyttsx3 voice engine ready")
except ImportError:
    HAS_TTS = False
    print("[WARN] pyttsx3 not installed — pip install pyttsx3")
    print("       Voice output disabled.\n")


def speak_async(text: str):
    """Speak in background thread. pyttsx3 needs a fresh engine per thread."""
    if not HAS_TTS or not text:
        return None
    def _speak():
        try:
            with _tts_lock:
                eng = _create_engine()
                eng.say(text)
                eng.runAndWait()
                eng.stop()
        except Exception as e:
            print(f"  [TTS error: {e}]")
    t = threading.Thread(target=_speak, daemon=True)
    t.start()
    return t


# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
PALIGEMMA_BASE = "google/paligemma2-3b-pt-224"
PG_ADAPTER     = BASE_DIR / "labellens_model"
GEMMA_BASE     = "google/gemma-3-4b-it"
GEMMA_ADAPTER  = BASE_DIR / "lora_adapter"
LOG_FILE       = BASE_DIR / "compliance_log.json"

# ── Label maps ─────────────────────────────────────────────────────────────
CANONICAL_TO_DISPLAY = {
    "explosive":      "GHS01 - Explosive",
    "flammable":      "GHS02 - Flammable",
    "oxidizer":       "GHS03 - Oxidizer",
    "compressed_gas": "GHS04 - Compressed Gas",
    "corrosive":      "GHS05 - Corrosive",
    "toxic":          "GHS06 - Toxic",
    "harmful":        "GHS07 - Harmful / Irritant",
    "health_hazard":  "GHS08 - Health Hazard",
    "environmental":  "GHS09 - Environmental Hazard",
}

RISK_LEVELS = {
    "explosive": 4, "toxic": 4,
    "corrosive": 3, "oxidizer": 3, "health_hazard": 3,
    "flammable": 2, "compressed_gas": 2,
    "harmful": 2, "environmental": 1,
}

SYSTEM_PROMPT = (
    "You are an on-device safety assistant for industrial workers. "
    "Given detected GHS hazard symbols (and optionally OCR text from a product label), "
    "provide a structured safety brief with: severity level, hazard summary, required PPE, "
    "step-by-step handling SOP, storage requirements, emergency/first-aid procedures, "
    "autonomous safety actions, and a short spoken voice script (under 40 words). "
    "Be concise, factual, and prioritize worker safety."
)


# ── Compliance Logger ──────────────────────────────────────────────────────
class ComplianceLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.entries = []
        if self.path.exists():
            try:
                self.entries = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.entries = []

    def log(self, symbols, brief, voice, mode, latency):
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "mode": mode,
            "detected_symbols": symbols,
            "risk_level": max((RISK_LEVELS.get(s, 1) for s in symbols), default=0),
            "safety_brief": brief,
            "voice_script": voice,
            "latency_s": round(latency, 2),
        }
        self.entries.append(entry)
        self.path.write_text(
            json.dumps(self.entries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return entry

    def print_report(self):
        print(f"\n{'='*70}")
        print(f"  COMPLIANCE LOG  —  {len(self.entries)} scan(s)")
        print(f"{'='*70}")
        for i, e in enumerate(self.entries, 1):
            print(f"\n  [{i}] {e['timestamp']}")
            print(f"      Mode    : {e['mode']}")
            print(f"      Symbols : {', '.join(e['detected_symbols']) or 'none'}")
            print(f"      Risk    : {e['risk_level']}/4")
            print(f"      Voice   : \"{e['voice_script']}\"")
            print(f"      Latency : {e['latency_s']}s")
        print(f"{'='*70}\n")


# ── Load models ────────────────────────────────────────────────────────────
def load_paligemma():
    from transformers import (
        BitsAndBytesConfig,
        PaliGemmaForConditionalGeneration,
        PaliGemmaProcessor,
    )
    from peft import PeftModel

    print("\n[1/2] Loading PaliGemma (detection model)...")
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
    print("  [OK] PaliGemma ready.")
    return model, processor


def load_gemma():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print("[2/2] Loading Gemma (safety brief model)...")
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
    print("  [OK] Gemma ready.\n")
    return model, tokenizer


# ── PaliGemma detection ───────────────────────────────────────────────────
def detect_symbols(image: Image.Image, pg_model, pg_proc) -> tuple[list[str], str]:
    """Returns (list_of_canonical_labels, raw_model_output)."""
    img_224 = image.resize((224, 224))
    inputs = pg_proc(
        images=img_224,
        text="<image> detect ghs symbols",
        return_tensors="pt",
    ).to(pg_model.device, dtype=torch.bfloat16)

    with torch.no_grad():
        ids = pg_model.generate(**inputs, max_new_tokens=256, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    raw = pg_proc.decode(ids[0][input_len:], skip_special_tokens=True).strip()

    labels = []
    for part in raw.split(";"):
        part = part.strip()
        locs = re.findall(r"<loc(\d{4})>", part)
        label = re.sub(r"<loc\d{4}>", "", part).strip()
        if len(locs) == 4 and label:
            labels.append(label)

    return list(set(labels)), raw


# ── Gemma safety brief ────────────────────────────────────────────────────
def generate_brief(symbols: list[str], gm_model, gm_tok) -> str:
    if not symbols:
        return "No GHS hazard symbols detected. Area appears safe."

    display = [CANONICAL_TO_DISPLAY.get(s, s) for s in symbols]
    det_str = ", ".join(f"{s} (detected)" for s in display)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Symbols: {det_str}\n"
                "Provide: hazards, PPE, handling SOP, storage, "
                "emergency/first-aid, autonomous safety actions, "
                "and a short voice script."
            ),
        },
    ]

    prompt = gm_tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = gm_tok(prompt, return_tensors="pt").to(gm_model.device)

    with torch.no_grad():
        ids = gm_model.generate(
            **inputs,
            max_new_tokens=600,
            do_sample=False,
            repetition_penalty=1.1,
        )

    resp = gm_tok.decode(ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return resp.strip()


def extract_voice_script(brief: str) -> str:
    """Extract the spoken voice script from the brief."""
    # Try structured headers: "8)" or "VOICE SCRIPT"
    match = re.search(
        r'(?:8\s*[\.\):]|VOICE\s*SCRIPT)[:\s]*["\']?\s*(.+?)["\']?\s*$',
        brief, re.DOTALL | re.IGNORECASE,
    )
    if match:
        text = match.group(1).strip().strip("\"'")
        sentences = re.split(r"(?<=[.!?])\s+", text)
        result = ""
        for s in sentences:
            if len((result + " " + s).split()) > 45:
                break
            result = (result + " " + s).strip()
        return result if result else text[:200]

    # Fallback: quoted string
    match = re.search(r'"([^"]{20,})"', brief)
    if match:
        return match.group(1).strip()

    # Last resort: last lines
    lines = [l.strip() for l in brief.split("\n") if l.strip()]
    return " ".join(lines[-2:]).strip('"') if lines else ""


# ── Drawing helpers ────────────────────────────────────────────────────────
RISK_COLORS = {
    1: (0, 200, 100),   # green
    2: (0, 200, 255),   # yellow
    3: (0, 120, 255),   # orange
    4: (0, 0, 255),     # red
}

def draw_overlay(frame, state):
    """Draw mode, status, and results onto the display frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent header bar
    cv2.rectangle(overlay, (0, 0), (w, 50), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Mode indicator
    mode = state["mode"].upper()
    mode_color = (0, 255, 150) if state["mode"] == "delivery" else (0, 200, 255)
    cv2.putText(frame, f"MODE: {mode}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

    # Controls hint (right side)
    cv2.putText(frame, "SPACE=Scan  M=Mode  R=Report  Q=Quit",
                (w - 430, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Scanning indicator
    if state["scanning"]:
        # Pulsing green border
        t = int(time.time() * 4) % 2
        border_color = (0, 255, 100) if t else (0, 180, 70)
        cv2.rectangle(frame, (2, 2), (w - 3, h - 3), border_color, 4)
        cv2.putText(frame, "SCANNING...", (w // 2 - 100, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 3)
        return frame

    # Results display
    r = state.get("last_result")
    if r:
        symbols = r["symbols"]
        risk = r["risk"]
        voice = r["voice"]
        color = RISK_COLORS.get(risk, (200, 200, 200))

        y = 75

        # Risk level bar
        risk_label = ["", "LOW", "MEDIUM", "HIGH", "CRITICAL"][min(risk, 4)]
        cv2.rectangle(frame, (5, y - 5), (w - 5, y + 25), color, -1)
        cv2.putText(frame, f"  RISK: {risk_label} ({risk}/4)", (10, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 45

        # Detected symbols
        sym_text = ", ".join(CANONICAL_TO_DISPLAY.get(s, s) for s in symbols)
        cv2.putText(frame, f"Detected: {sym_text[:70]}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        y += 35

        # Voice script (what was spoken)
        cv2.putText(frame, "Voice:", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        y += 25
        # Wrap voice text for display
        for line in textwrap.wrap(voice, width=65)[:3]:
            cv2.putText(frame, line, (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
            y += 22

        # Latency
        y += 10
        cv2.putText(frame, f"Latency: {r['latency']:.1f}s", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    else:
        # No scan yet — show instruction
        cv2.putText(frame, "Hold a product label to the camera and press SPACE",
                    (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    return frame


# ── Scan pipeline (runs in background thread) ─────────────────────────────
def run_scan(frame, state, models, logger):
    """Full pipeline: detect -> brief -> voice -> log. Runs in a thread."""
    pg_model, pg_proc, gm_model, gm_tok = models

    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    t0 = time.time()

    # Stage 1: PaliGemma detection
    symbols, raw_pg = detect_symbols(pil, pg_model, pg_proc)
    t_detect = time.time() - t0

    # Console log — detection
    print(f"\n{'='*70}")
    print(f"  SCAN  |  {datetime.datetime.now().strftime('%H:%M:%S')}  |  Mode: {state['mode'].upper()}")
    print(f"{'='*70}")
    print(f"  PaliGemma raw : {raw_pg}")
    print(f"  Detected      : {symbols if symbols else '(none)'}")
    print(f"  Detection time: {t_detect:.2f}s")

    # Stage 2: Gemma safety brief
    brief = generate_brief(symbols, gm_model, gm_tok)
    voice = extract_voice_script(brief)
    elapsed = time.time() - t0

    risk = max((RISK_LEVELS.get(s, 1) for s in symbols), default=0)

    # Console log — full brief
    print(f"\n  -- Safety Brief --")
    for line in textwrap.wrap(brief, width=66):
        print(f"  {line}")
    print(f"\n  -- Voice Script --")
    print(f'  "{voice}"')
    print(f"\n  Risk level : {risk}/4")
    print(f"  Total time : {elapsed:.1f}s")

    # Stage 3: Voice output
    if voice:
        print(f"  Speaking...")
        speak_async(voice)

    # Stage 4: Compliance log
    entry = logger.log(symbols, brief, voice, state["mode"], elapsed)
    print(f"  Logged to  : {logger.path}")
    print(f"{'='*70}")

    # Update shared state for UI
    state["last_result"] = {
        "symbols": symbols,
        "risk": risk,
        "brief": brief,
        "voice": voice,
        "latency": elapsed,
        "raw": raw_pg,
    }
    state["scanning"] = False


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 70)
    print("  LabelLens Live Demo")
    print("=" * 70)

    # Load models
    pg_model, pg_proc = load_paligemma()
    gm_model, gm_tok  = load_gemma()
    models = (pg_model, pg_proc, gm_model, gm_tok)

    logger = ComplianceLogger(str(LOG_FILE))

    # State shared between main loop and scan thread
    state = {
        "mode": "scan",
        "scanning": False,
        "last_result": None,
        "scan_count": 0,
    }

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check camera connection.")
        sys.exit(1)

    # Try to set decent resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n[OK] Webcam opened: {actual_w}x{actual_h}")
    print(f"\n  Controls:")
    print(f"    SPACE = Scan label")
    print(f"    M     = Toggle mode (Scan/Delivery)")
    print(f"    R     = Print compliance report")
    print(f"    Q     = Quit")
    print(f"\n  Ready! Hold a label to the camera and press SPACE.\n")

    # Startup voice
    speak_async("LabelLens ready. Hold a product label to the camera and press space to scan.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from webcam.")
            break

        # Draw UI overlay
        display = draw_overlay(frame.copy(), state)
        cv2.imshow("LabelLens Live Demo", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("\n[EXIT] Shutting down...")
            break

        elif key == ord(" ") and not state["scanning"]:
            state["scanning"] = True
            state["scan_count"] += 1
            print(f"\n>>> Scan #{state['scan_count']} triggered...")
            t = threading.Thread(
                target=run_scan,
                args=(frame.copy(), state, models, logger),
                daemon=True,
            )
            t.start()

        elif key == ord("m"):
            state["mode"] = "delivery" if state["mode"] == "scan" else "scan"
            print(f"\n>>> Mode switched to: {state['mode'].upper()}")
            speak_async(f"Mode changed to {state['mode']}")

        elif key == ord("r"):
            logger.print_report()

    cap.release()
    cv2.destroyAllWindows()

    # Final report
    if logger.entries:
        print("\n--- Final Compliance Report ---")
        logger.print_report()

    print("Done.")


if __name__ == "__main__":
    main()