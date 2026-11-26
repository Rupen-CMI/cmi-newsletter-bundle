import glob, os, shutil, torch, transformers
from peft import PeftModel
BASE, ADAPTER, OUT = "/home/qwen35base", "/home/v4adapter", "/home/merged"
cls = getattr(transformers, "Qwen3_5ForConditionalGeneration")
base = cls.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cpu")
model = PeftModel.from_pretrained(base, ADAPTER)
n = sum(1 for n_, _ in model.named_modules() if "lora_A" in n_)
print("lora modules attached:", n, flush=True)
model = model.merge_and_unload()
model.save_pretrained(OUT, safe_serialization=True)
for f in glob.glob(BASE + "/*"):
    b = os.path.basename(f)
    if not b.endswith(".safetensors") and b not in ("model.safetensors.index.json", "config.json") and os.path.isfile(f):
        shutil.copy(f, OUT)
for b in ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json"):
    if os.path.exists(os.path.join(ADAPTER, b)):
        shutil.copy(os.path.join(ADAPTER, b), OUT)
print("merged ->", OUT, flush=True)
