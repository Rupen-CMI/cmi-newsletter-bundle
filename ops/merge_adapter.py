"""Merge the LoRA adapter into the base for vLLM. Run on the GPU box.

vLLM cannot apply a PEFT adapter as cheaply as transformers can, and continuous
batching -- the whole reason for using it -- needs the merged weights resident.
fp16 on CPU: the merge is arithmetic, not inference, so CPU RAM is cheaper than
holding 16GB of VRAM for it.
"""
import torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE, ADAPTER, OUT = "Qwen/Qwen2.5-7B-Instruct", "/home/v1adapter", "/home/merged"
print("loading base fp16 on cpu ...", flush=True)
try:
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16, device_map="cpu")
except TypeError:
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map="cpu")
print("applying + merging adapter ...", flush=True)
m = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload()
print("saving ...", flush=True)
m.save_pretrained(OUT, safe_serialization=True)
AutoTokenizer.from_pretrained(ADAPTER).save_pretrained(OUT)
print("MERGE DONE ->", OUT, flush=True)
