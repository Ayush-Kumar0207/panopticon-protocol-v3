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
PROMPT_INPUT_BUDGET = 880

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


def _take_priority(items: list[Any], limit: int | None, score_fn) -> tuple[list[Any], int]:
    if limit is None or len(items) <= limit:
        return items, 0
    ranked = sorted(items, key=score_fn, reverse=True)
    selected = ranked[:limit]
    return selected, max(0, len(items) - len(selected))


def format_observation(
    obs: EnvironmentObservation,
    *,
    max_workers: int | None = None,
    max_leaks: int | None = None,
    max_traps: int | None = None,
    max_reports: int | None = 4,
) -> str:
    workers, omitted_workers = _take_priority(
        list(obs.workers),
        max_workers,
        lambda worker: (
            worker.turning_in_progress,
            worker.state != "loyal",
            worker.suspicion_level,
            worker.performance,
        ),
    )
    workers_info = []
    for worker in workers:
        status = f"suspicion={worker.suspicion_level:.0%}" if worker.suspicion_level > 0.05 else "clean"
        if worker.turning_in_progress:
            status += f" turning={worker.interrogation_progress}/4"
        workers_info.append(
            f"  {worker.id} {worker.name} dept={worker.department} state={worker.state} {status}"
        )

    leaks, omitted_leaks = _take_priority(
        list(obs.active_leaks),
        max_leaks,
        lambda leak: (leak.is_canary, not leak.verified, leak.turn_detected),
    )
    leaks_info = []
    for leak in leaks:
        canary = " [CANARY MATCH]" if leak.is_canary else ""
        leaks_info.append(f"  {leak.id} dept={leak.department} channel={leak.channel}{canary}")

    traps, omitted_traps = _take_priority(
        list(obs.canary_traps),
        max_traps,
        lambda trap: (trap.triggered, trap.planted_turn),
    )
    traps_info = [f"  {trap.id} dept={trap.department} triggered={trap.triggered}" for trap in traps]

    reports_source = list(obs.intel_reports)
    if max_reports is not None and len(reports_source) > max_reports:
        reports_source = sorted(
            reports_source,
            key=lambda report: (bool(report.flagged_workers), report.confidence, report.turn),
            reverse=True,
        )[:max_reports]
    omitted_reports = max(0, len(obs.intel_reports) - len(reports_source))
    intel_info = []
    for report in reports_source:
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

    suspicious_workers = sum(1 for worker in obs.workers if worker.suspicion_level > 0.25)
    triggered_canaries = sum(1 for trap in obs.canary_traps if trap.triggered)
    sections = [
        f"Turn {obs.turn}/{obs.max_turns} | Phase: {obs.phase} ({obs.phase_number}) | Revenue: {obs.enterprise_revenue:.0f} | Security: {obs.security_score:.0f}",
        (
            f"Summary: suspicious={suspicious_workers} | leaks={len(obs.active_leaks)} | "
            f"triggered_canaries={triggered_canaries} | intel={len(obs.intel_reports)} | "
            f"double_agents={len(obs.double_agents)}"
        ),
        f"Workers ({len(obs.workers)}):",
        "\n".join(workers_info) if workers_info else "  (none)",
        f"  ... {omitted_workers} additional workers omitted" if omitted_workers else "",
        f"Active Leaks ({len(obs.active_leaks)}):",
        "\n".join(leaks_info) if leaks_info else "  (none)",
        f"  ... {omitted_leaks} additional leaks omitted" if omitted_leaks else "",
        f"Canary Traps ({len(obs.canary_traps)}):",
        "\n".join(traps_info) if traps_info else "  (none)",
        f"  ... {omitted_traps} additional canaries omitted" if omitted_traps else "",
        f"Recent Intel Reports ({len(obs.intel_reports)}):",
        "\n".join(intel_info) if intel_info else "  (none)",
        f"  ... {omitted_reports} additional reports omitted" if omitted_reports else "",
        f"Active Double Agents ({len(obs.double_agents)}):",
        "\n".join(da_info) if da_info else "  (none)",
    ]
    return "\n".join(section for section in sections if section)


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
        self.tokenizer.truncation_side = "left"

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
        return self._build_messages_from_text(format_observation(obs))

    def _build_messages_from_text(self, observation_text: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Current State:\n{observation_text}\n\nYour action (JSON):"},
        ]

    def _render_prompt(self, messages: list[dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return (
                f"{SYSTEM_PROMPT}\n\nCurrent State:\n{messages[1]['content'].replace('Current State:\\n', '', 1)}"
            )

    def _prompt_token_length(self, prompt: str) -> int:
        return len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])

    def build_prompt(self, obs: EnvironmentObservation) -> tuple[str, list[dict[str, str]]]:
        variants = [
            {},
            {"max_workers": 12, "max_leaks": 18, "max_traps": 10, "max_reports": 6},
            {"max_workers": 10, "max_leaks": 12, "max_traps": 8, "max_reports": 5},
            {"max_workers": 8, "max_leaks": 8, "max_traps": 6, "max_reports": 4},
            {"max_workers": 6, "max_leaks": 5, "max_traps": 4, "max_reports": 3},
        ]
        last_prompt = ""
        last_messages: list[dict[str, str]] = []
        for variant in variants:
            observation_text = format_observation(obs, **variant)
            messages = self._build_messages_from_text(observation_text)
            prompt = self._render_prompt(messages)
            last_prompt, last_messages = prompt, messages
            if self._prompt_token_length(prompt) <= PROMPT_INPUT_BUDGET:
                return prompt, messages
        return last_prompt, last_messages

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
