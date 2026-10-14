import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE    = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = "/home/v1adapter"
OUT     = "/home/merged"

base = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.float16, device_map="cpu")
model = PeftModel.from_pretrained(base, ADAPTER)
model = model.merge_and_unload()
model.save_pretrained(OUT, safe_serialization=True)
AutoTokenizer.from_pretrained(ADAPTER).save_pretrained(OUT)
print("merged ->", OUT)
