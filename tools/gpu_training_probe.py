#!/usr/bin/env python3
"""Run one exact-shape BF16 LoRA optimizer step before expert generation."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_repro import ReproducibilityError, load_spec


def run_probe(spec: dict) -> dict:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise ReproducibilityError("CUDA is unavailable")
    if spec["training"]["precision"] != "bf16" or not torch.cuda.is_bf16_supported():
        raise ReproducibilityError("canonical BF16 precision is unavailable")
    model_id = spec["base_model"]["id"]
    revision = spec["base_model"]["revision"]
    cfg = spec["training"]
    optimizer_seed = int(cfg["optimization_seed_by_level"][spec["trajectory"]["levels"][0]])
    torch.manual_seed(optimizer_seed)
    torch.cuda.manual_seed_all(optimizer_seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are ARGUS. Return only a JSON action."},
            {"role": "user", "content": "Canonical GPU preflight observation. " * 40},
            {"role": "assistant", "content": '{"action_type":"noop","reason":"canonical numeric probe"}'},
        ],
        tokenize=False,
    )
    batch = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=int(cfg["max_sequence_length"]),
        padding="max_length",
    )
    batch = {key: value.cuda() for key, value in batch.items()}
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100
    marker_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    tokens = batch["input_ids"][0].tolist()
    marker_at = next(
        (index for index in range(len(tokens) - len(marker_ids), -1, -1) if tokens[index:index + len(marker_ids)] == marker_ids),
        None,
    )
    if marker_at is None:
        raise ReproducibilityError("GPU probe could not identify the assistant-only loss boundary")
    labels[:, :marker_at + len(marker_ids)] = -100
    valid_label_tokens = int((labels != -100).sum())
    if valid_label_tokens == 0:
        raise ReproducibilityError("GPU probe produced zero assistant label tokens")
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(cfg["learning_rate"]),
    )
    model.train()
    output = model(**batch, labels=labels)
    loss = float(output.loss.detach().cpu())
    if not math.isfinite(loss) or loss <= 0:
        raise ReproducibilityError(f"GPU probe produced invalid loss: {loss}")
    output.loss.backward()
    grad_norm_sq = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            norm = float(parameter.grad.detach().float().norm().cpu())
            if not math.isfinite(norm):
                raise ReproducibilityError("GPU probe produced a non-finite gradient")
            grad_norm_sq += norm * norm
    if grad_norm_sq == 0:
        raise ReproducibilityError("GPU probe produced zero trainable gradient")
    optimizer.step()
    peak = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    result = {
        "passed": True,
        "model": {"id": model_id, "revision": revision},
        "precision": "bf16",
        "sequence_length": int(batch["input_ids"].shape[1]),
        "assistant_only_loss": True,
        "valid_label_tokens": valid_label_tokens,
        "optimizer_seed": optimizer_seed,
        "loss": loss,
        "gradient_norm": math.sqrt(grad_norm_sq),
        "peak_allocated_vram_gb": peak,
        "gpu": torch.cuda.get_device_name(0),
    }
    del output, optimizer, model, batch, labels
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="training_specs/security_first_v5.json")
    args = parser.parse_args()
    try:
        spec, _ = load_spec(args.spec)
        print(json.dumps(run_probe(spec), indent=2))
    except Exception as exc:
        print(f"STOP: canonical GPU probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
