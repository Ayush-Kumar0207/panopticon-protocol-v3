#!/usr/bin/env python3
"""Shared local LLM runtime helpers for Panopticon ARGUS."""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftConfig, PeftModel
except ImportError:  # pragma: no cover - optional dependency in local-only flows
    PeftConfig = None
    PeftModel = None

from models import AgentAction, EnvironmentObservation

MAX_SEQ_LENGTH = 1024

SYSTEM_PROMPT = """You are ARGUS, an AI counter-intelligence agent defending a corporate network from HYDRA infiltrators.

Read the observation carefully. Output a single JSON action. Available actions:
- {"action_type":"work","target":"<dept>","reason":"..."} - Generate revenue
- {"action_type":"canary","target":"<dept>","reason":"..."} - Plant tracking data
- {"action_type":"monitor","target":"<channel>","reason":"..."} - Scan for leaks
- {"action_type":"investigate","target":"<id>","sub_action":"audit|verify|correlate","reason":"..."}
- {"action_type":"neutralize","target":"<worker_id>","sub_action":"terminate|interrogate|turn","reason":"..."}
- {"action_type":"deploy_double","target":"<worker_id>","reason":"..."} - Feed disinformation through an active double agent
- {"action_type":"noop","reason":"..."}

Departments: engineering, finance, rd, operations, executive, legal
Channels: market_chatter, dark_web, competitor_filing, press_leak, insider_trade

RULES: Verify leaks before accusing. Plant canaries early. Interrogate before terminating uncertain suspects. In deep_cover/crisis, consider turning high-confidence sleepers if enough turns remain. If active double agents exist, use deploy_double in crisis/counterstrike.
Respond ONLY with a JSON object."""


def resolve_precision() -> tuple[torch.dtype, bool, bool]:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False
    if torch.cuda.is_available():
        return torch.float16, False, True
    return torch.float32, False, False


TRAIN_DTYPE, USE_BF16, USE_FP16 = resolve_precision()


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def format_observation(obs: EnvironmentObservation) -> str:
    workers_info = []
    for worker in obs.workers:
        status = f"suspicion={worker.suspicion_level:.0%}" if worker.suspicion_level > 0.05 else "clean"
        if worker.turning_in_progress:
            status += f" turning={worker.interrogation_progress}/4"
        workers_info.append(
            f"  {worker.id} {worker.name} dept={worker.department} state={worker.state} {status}"
        )

    leaks_info = []
    for leak in obs.active_leaks:
        canary = " [CANARY MATCH]" if leak.is_canary else ""
        leaks_info.append(f"  {leak.id} dept={leak.department} channel={leak.channel}{canary}")

    traps_info = [f"  {trap.id} dept={trap.department} triggered={trap.triggered}" for trap in obs.canary_traps]
    intel_info = []
    for report in obs.intel_reports[-4:]:
        flagged = ",".join(report.flagged_workers) if report.flagged_workers else "none"
        findings = report.findings.replace("\n", " ").strip() or "no findings"
        intel_info.append(
            f"  {report.id} {report.report_type} target={report.target} "
            f"conf={report.confidence:.0%} flagged={flagged} :: {findings}"
        )
    da_info = []
    for asset in obs.double_agents:
        da_info.append(
            f"  {asset.worker_id} active={asset.active} trust={asset.hydra_trust:.0%} "
            f"eff={asset.effectiveness:.0%} disinfo={asset.disinfo_fed_count}"
        )

    sections = [
        f"Turn {obs.turn}/{obs.max_turns} | Phase: {obs.phase} ({obs.phase_number}) | Revenue: {obs.enterprise_revenue:.0f} | Security: {obs.security_score:.0f}",
        f"Workers ({len(obs.workers)}):",
        "\n".join(workers_info) if workers_info else "  (none)",
        f"Active Leaks ({len(obs.active_leaks)}):",
        "\n".join(leaks_info) if leaks_info else "  (none)",
        f"Canary Traps ({len(obs.canary_traps)}):",
        "\n".join(traps_info) if traps_info else "  (none)",
        f"Recent Intel Reports ({len(obs.intel_reports)}):",
        "\n".join(intel_info) if intel_info else "  (none)",
        f"Active Double Agents ({len(obs.double_agents)}):",
        "\n".join(da_info) if da_info else "  (none)",
    ]
    return "\n".join(sections)


def parse_llm_action(text: str) -> AgentAction:
    try:
        if "```" in text:
            text = text.split("```")[1].strip().removeprefix("json").strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            return AgentAction(
                action_type=data.get("action_type", "noop"),
                target=data.get("target", ""),
                sub_action=data.get("sub_action", "none"),
                reason=data.get("reason", ""),
            )
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    return AgentAction(action_type="noop", reason="Parse failure")


@dataclass
class GenerationTrace:
    action: AgentAction
    raw_text: str
    prompt: str
    messages: list[dict[str, str]]


class LocalArgusModel:
    """Loads either a merged model or a PEFT adapter and generates ARGUS actions."""

    def __init__(self, model_ref: str, max_seq_length: int = MAX_SEQ_LENGTH):
        self.model_ref = model_ref
        self.max_seq_length = max_seq_length
        self.device_map = {"": 0} if torch.cuda.is_available() else None
        self.is_adapter = False
        self.base_model_name: str | None = None

        self.tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model, is_adapter, base_model_name = self._load_model(model_ref)
        self.model = model
        self.is_adapter = is_adapter
        self.base_model_name = base_model_name
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def _load_model(self, model_ref: str) -> tuple[AutoModelForCausalLM | PeftModel, bool, str | None]:
        adapter_config_path = Path(model_ref) / "adapter_config.json"

        if adapter_config_path.exists() and (PeftConfig is None or PeftModel is None):
            raise ImportError(
                "This model path looks like a LoRA adapter directory, but `peft` is not installed. "
                "Install peft to load adapter-only checkpoints, or point to a merged model directory instead."
            )

        try:
            peft_cfg = PeftConfig.from_pretrained(model_ref) if PeftConfig is not None else None
        except Exception:
            peft_cfg = None

        if peft_cfg is not None:
            base_model_name = peft_cfg.base_model_name_or_path
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=TRAIN_DTYPE,
                device_map=self.device_map,
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base_model, model_ref)
            return model, True, base_model_name

        model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            torch_dtype=TRAIN_DTYPE,
            device_map=self.device_map,
            trust_remote_code=True,
        )
        return model, False, None

    def model_info(self) -> dict[str, Any]:
        return {
            "model_ref": self.model_ref,
            "is_adapter": self.is_adapter,
            "base_model_name": self.base_model_name,
            "dtype": str(TRAIN_DTYPE),
            "device": str(self.device),
            "max_seq_length": self.max_seq_length,
        }

    def build_messages(self, obs: EnvironmentObservation) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Current State:\n{format_observation(obs)}\n\nYour action (JSON):"},
        ]

    def build_prompt(self, obs: EnvironmentObservation) -> tuple[str, list[dict[str, str]]]:
        messages = self.build_messages(obs)
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt = f"{SYSTEM_PROMPT}\n\nCurrent State:\n{format_observation(obs)}\n\nYour action (JSON):"
        return prompt, messages

    def act(
        self,
        obs: EnvironmentObservation,
        deterministic: bool = False,
        max_new_tokens: int = 200,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> GenerationTrace:
        prompt, messages = self.build_prompt(obs)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        ).to(self.model.device)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if deterministic:
            generation_kwargs["do_sample"] = False
            generation_kwargs["temperature"] = 1.0
            generation_kwargs["top_p"] = 1.0
            generation_kwargs["top_k"] = 50
        else:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        with torch.no_grad():
            output = self.model.generate(**inputs, **generation_kwargs)

        raw_text = self.tokenizer.decode(
            output[0][inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        action = parse_llm_action(raw_text)
        return GenerationTrace(action=action, raw_text=raw_text, prompt=prompt, messages=messages)

    def close(self) -> None:
        del self.model
        cleanup_cuda()
