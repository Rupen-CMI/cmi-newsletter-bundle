---
library_name: peft
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
tags:
- generated_from_trainer
model-index:
- name: out
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/axolotl-ai-cloud/axolotl)
<details><summary>See axolotl config</summary>

axolotl version: `0.5.0`
```yaml
# QLoRA fine-tune of Qwen2.5-7B-Instruct on the CMI house-voice dataset.
#
# Sized for a single on-demand L4 (24GB) container on Jarvislabs (axolotl 0.5.0).
# Phase 3 of PLAN.md, stepped down from the 14B in PLAN.md 2.1 so the first run
# fits the current balance (L4 ~Rs.41/hr vs A100 ~Rs.84/hr).
#
#   jl upload <id> <this dir> /home/train
#   jl exec <id> -- bash -lc 'cd /home/train && bash run_train.sh'
#
# Dataset is PRE-TEMPLATED by prepare_dataset.py into Qwen ChatML, so the type is
# `completion` with field `text` — axolotl must NOT re-apply a chat template on
# top of one that's already there.
base_model: Qwen/Qwen2.5-7B-Instruct
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer

load_in_4bit: true
adapter: qlora

datasets:
  - path: train.jsonl
    type: completion
    field: text

test_datasets:
  - path: val.jsonl
    type: completion
    field: text
    split: train

dataset_prepared_path: ./prepared
output_dir: ./out

# 2051 tokens is the longest example in train.jsonl; 2048 would silently truncate
# the tail (the assistant's closing sentences) on the longest briefs. 2560 clears
# it with headroom.
sequence_len: 2560

# Packing concatenates multiple short examples into one sequence. Off here: it
# blurs the boundary between independent (system,user,output) triples, and with
# 559 examples the throughput win is not worth muddying per-example learning.
sample_packing: false
pad_to_sequence_len: true

# PLAN.md 3.3: rank 16-32, alpha 32-64, attention + MLP projections.
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

gradient_accumulation_steps: 8
micro_batch_size: 1
num_epochs: 3
optimizer: paged_adamw_8bit
lr_scheduler: cosine
learning_rate: 0.0002

bf16: auto
tf32: false
gradient_checkpointing: true
flash_attention: true

warmup_ratio: 0.03
weight_decay: 0.0
max_grad_norm: 1.0

# Eval + checkpoint every 20 steps. Cheap insurance on a per-minute-billed box:
# if the run dies for any reason, run_train.sh resumes from the last checkpoint
# instead of paying to recompute from zero.
evaluation_strategy: steps
save_strategy: steps
eval_steps: 20
save_steps: 20
save_total_limit: 3
logging_steps: 1

# 559 examples is small enough to overfit within 3 epochs. Keep the best-by-eval
# checkpoint rather than trusting the final one.
load_best_model_at_end: true
metric_for_best_model: eval_loss
greater_is_better: false

special_tokens:
  pad_token: "<|endoftext|>"

```

</details><br>

# out

This model is a fine-tuned version of [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3592

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 8
- optimizer: Use OptimizerNames.PAGED_ADAMW_8BIT with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 6
- num_epochs: 3

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 2.8663        | 0.0143 | 1    | 3.0989          |
| 0.9288        | 0.2862 | 20   | 0.8362          |
| 0.5357        | 0.5725 | 40   | 0.5340          |
| 0.3969        | 0.8587 | 60   | 0.4640          |
| 0.2584        | 1.1449 | 80   | 0.4268          |
| 0.3423        | 1.4311 | 100  | 0.3917          |
| 0.4112        | 1.7174 | 120  | 0.3786          |
| 0.396         | 2.0036 | 140  | 0.3618          |
| 0.2074        | 2.2898 | 160  | 0.3616          |
| 0.1537        | 2.5760 | 180  | 0.3613          |
| 0.2066        | 2.8623 | 200  | 0.3592          |


### Framework versions

- PEFT 0.13.2
- Transformers 4.46.1
- Pytorch 2.3.1+cu121
- Datasets 3.0.1
- Tokenizers 0.20.3