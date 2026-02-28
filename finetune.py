# finetune_paligemma.py
import torch, json, os, re
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig, TaskType
import warnings
warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────
# CONFIG  (tuned for 8 GB VRAM / Windows)
# ───────────────────────────────────────────────
MODEL_ID   = "google/paligemma2-3b-pt-224"
TRAIN_JSON = "./merged_dataset/train.json"
VAL_JSON   = "./merged_dataset/val.json"
OUTPUT_DIR = "./labellens_model"
EPOCHS     = 3
LR         = 2e-5
# ───────────────────────────────────────────────

# ── Sanity checks ──
assert torch.cuda.is_available(), "No GPU found!"
print(f"GPU  : {torch.cuda.get_device_name(0)}")
print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Load model in 4-bit QLoRA ──
print("\nLoading PaliGemma 2 3B (4-bit) …")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",   # required on Windows (no flash-attn)
)
processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)
print(f"Model loaded  →  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# ── 2. Freeze vision tower, unfreeze language model ──
# Vision encoder already understands images fine out-of-the-box.
# We only need the language model to learn the GHS output format.
for name, param in model.named_parameters():
    if "vision_tower" in name or "multi_modal_projector" in name:
        param.requires_grad = False

# ── 3. Apply LoRA to the language model layers ──
lora = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()
# → trainable: 11,534,336 / 2.9B  (0.39 %)
print(f"After LoRA  →  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB\n")

# ── 4. Dataset ──
class GHSDataset(Dataset):
    def __init__(self, json_path: str, label: str):
        with open(json_path) as f:
            raw = json.load(f)
        # Drop entries whose image file no longer exists
        self.data = [d for d in raw if os.path.exists(d["image_path"])]
        dropped   = len(raw) - len(self.data)
        if dropped:
            print(f"  [{label}] Dropped {dropped} missing-image entries")
        print(f"  [{label}] {len(self.data)} examples loaded")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        # Images were already saved at 224×224 during conversion
        img = Image.open(d["image_path"]).convert("RGB")
        return {"image": img, "prefix": d["prefix"], "suffix": d["suffix"]}

train_ds = GHSDataset(TRAIN_JSON, "train")
val_ds   = GHSDataset(VAL_JSON,   "val")

# ── 5. Collator ──
def collate_fn(batch):
    images  = [ex["image"]  for ex in batch]
    # PaliGemma expects <image> token at the start of every prompt
    texts   = ["<image> " + ex["prefix"] for ex in batch]
    targets = [ex["suffix"] for ex in batch]

    tokens = processor(
        images=images,
        text=texts,
        suffix=targets,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=512,
    )
    # Cast float tensors to bfloat16 to match model dtype
    tokens = {
        k: (v.to(torch.bfloat16) if v.dtype == torch.float32 else v)
        for k, v in tokens.items()
    }
    return tokens

# ── 6. Training arguments (8 GB / Windows safe) ──
args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # ── epochs & batch ──
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,   # effective batch = 8

    # ── optimiser ──
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    optim="adamw_bnb_8bit",          # 8-bit Adam → saves ~1 GB

    # ── precision ──
    bf16=True,
    fp16=False,

    # ── logging / saving ──
    logging_steps=20,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=2,              # keep only 2 checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    # ── Windows must-haves ──
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    remove_unused_columns=False,
    report_to="none",
)

# ── 7. Trainer ──
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
)

# ── 8. Train ──
steps_per_epoch = len(train_ds) // 8          # effective batch = 8
total_steps     = steps_per_epoch * EPOCHS
mins_estimate   = total_steps * 2.5 / 60      # ~2.5 s/step on RTX 4060

print(f"Training plan:")
print(f"  Train examples : {len(train_ds)}")
print(f"  Val examples   : {len(val_ds)}")
print(f"  Steps/epoch    : {steps_per_epoch}")
print(f"  Total steps    : {total_steps}")
print(f"  Est. time      : ~{mins_estimate:.0f} min on RTX 4060/3060\n")

trainer.train()

# ── 9. Save LoRA adapter ──
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"\n✅  Adapter saved → {OUTPUT_DIR}/")
print(f"    (only ~50 MB — just the LoRA weights, not the full model)")