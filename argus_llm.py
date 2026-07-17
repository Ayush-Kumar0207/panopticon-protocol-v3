#!/usr/bin/env python3
"""Shared local LLM runtime helpers for Panopticon ARGUS."""

from __future__ import annotations

import copy
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

MAX_SEQ_LENGTH = 512
MAX_NEW_TOKENS = 128

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

PRIORITIES: Confirmed threats outrank all revenue and double-agent actions. Keep security at or above 90. Catch every sleeper with zero false accusations. Verify leaks before accusing. Plant canaries early. Interrogate uncertain suspects. Turn at most one confirmed sleeper in advanced levels, then terminate the remaining confirmed threats. Deploy a double agent only when no confirmed/high-suspicion threat is waiting and security is at least 95.
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
    max_double_agents: int | None = None,
    compact: bool = False,
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
    double_agents, omitted_double_agents = _take_priority(
        list(obs.double_agents),
        max_double_agents,
        lambda asset: (asset.active, asset.hydra_trust, asset.effectiveness),
    )
    da_info = []
    for asset in double_agents:
        da_info.append(
            f"  {asset.worker_id} active={asset.active} trust={asset.hydra_trust:.0%} "
            f"eff={asset.effectiveness:.0%} disinfo={asset.disinfo_fed_count}"
        )

    suspicious_workers = sum(1 for worker in obs.workers if worker.suspicion_level > 0.25)
    triggered_canaries = sum(1 for trap in obs.canary_traps if trap.triggered)
    active_departments = []
    for worker in obs.workers:
        if worker.department not in active_departments:
            active_departments.append(worker.department)
    if compact:
        def compact_section(
            label: str,
            items: list[str],
            omitted: int,
        ) -> str:
            selected = " | ".join(item.strip() for item in items) or "none"
            suffix = f" | +{omitted} omitted" if omitted else ""
            return f"{label}={selected}{suffix}"

        return "\n".join(
            [
                (
                    f"T={obs.turn}/{obs.max_turns} phase={obs.phase}:{obs.phase_number} "
                    f"revenue={obs.enterprise_revenue:.0f} security={obs.security_score:.0f}"
                ),
                (
                    f"Counts workers={len(obs.workers)} suspicious={suspicious_workers} "
                    f"leaks={len(obs.active_leaks)} triggered={triggered_canaries} "
                    f"intel={len(obs.intel_reports)} doubles={len(obs.double_agents)}"
                ),
                f"Departments={','.join(active_departments)}",
                compact_section("Workers", workers_info, omitted_workers),
                compact_section("Leaks", leaks_info, omitted_leaks),
                compact_section("Traps", traps_info, omitted_traps),
                compact_section("Intel", intel_info, omitted_reports),
                compact_section("Doubles", da_info, omitted_double_agents),
            ]
        )

    sections = [
        f"Turn {obs.turn}/{obs.max_turns} | Phase: {obs.phase} ({obs.phase_number}) | Revenue: {obs.enterprise_revenue:.0f} | Security: {obs.security_score:.0f}",
        (
            f"Summary: suspicious={suspicious_workers} | leaks={len(obs.active_leaks)} | "
            f"triggered_canaries={triggered_canaries} | intel={len(obs.intel_reports)} | "
            f"double_agents={len(obs.double_agents)}"
        ),
        f"Allowed Departments: {', '.join(active_departments)}",
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
        (
            f"  ... {omitted_double_agents} additional double agents omitted"
            if omitted_double_agents
            else ""
        ),
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
class PromptBuild:
    prompt: str
    messages: list[dict[str, str]]
    original_token_count: int
    final_token_count: int
    compaction_level: int


@dataclass
class GenerationTrace:
    action: AgentAction
    raw_text: str
    prompt: str
    messages: list[dict[str, str]]
    original_prompt_tokens: int
    prompt_tokens: int
    prompt_limit: int
    max_new_tokens: int
    compaction_level: int
    token_truncated: bool


class LocalArgusModel:
    """Loads either a merged model or a PEFT adapter and generates ARGUS actions."""

    def __init__(
        self,
        model_ref: str,
        max_seq_length: int = MAX_SEQ_LENGTH,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ):
        if max_seq_length < 128:
            raise ValueError("max_seq_length must be at least 128")
        if not 1 <= max_new_tokens <= max_seq_length:
            raise ValueError("max_new_tokens must be between 1 and max_seq_length")
        self.model_ref = model_ref
        self.max_seq_length = max_seq_length
        self.max_new_tokens = max_new_tokens
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
                dtype=TRAIN_DTYPE,
                device_map=self.device_map,
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base_model, model_ref)
            return model, True, base_model_name

        model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            dtype=TRAIN_DTYPE,
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
            "max_new_tokens": self.max_new_tokens,
            "prompt_fitting": "structured_compaction_no_token_truncation",
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
        return len(
            self.tokenizer(
                prompt,
                add_special_tokens=False,
                verbose=False,
            )["input_ids"]
        )

    def build_prompt(self, obs: EnvironmentObservation) -> PromptBuild:
        variants = [
            {},
            {"max_workers": 12, "max_leaks": 18, "max_traps": 10, "max_reports": 6},
            {"max_workers": 10, "max_leaks": 12, "max_traps": 8, "max_reports": 5},
            {"max_workers": 8, "max_leaks": 8, "max_traps": 6, "max_reports": 4},
            {"max_workers": 6, "max_leaks": 5, "max_traps": 4, "max_reports": 3},
            {"max_workers": 4, "max_leaks": 3, "max_traps": 2, "max_reports": 2},
            {"max_workers": 3, "max_leaks": 2, "max_traps": 1, "max_reports": 1},
            {
                "max_workers": 2,
                "max_leaks": 1,
                "max_traps": 1,
                "max_reports": 1,
                "max_double_agents": 1,
            },
            {
                "max_workers": 1,
                "max_leaks": 1,
                "max_traps": 1,
                "max_reports": 1,
                "max_double_agents": 1,
            },
            {
                "max_workers": 0,
                "max_leaks": 1,
                "max_traps": 0,
                "max_reports": 1,
                "max_double_agents": 1,
            },
            {
                "max_workers": 1,
                "max_leaks": 1,
                "max_traps": 0,
                "max_reports": 0,
                "max_double_agents": 1,
            },
            {
                "max_workers": 0,
                "max_leaks": 0,
                "max_traps": 0,
                "max_reports": 0,
                "max_double_agents": 1,
                "compact": True,
            },
        ]
        original_token_count = 0
        last_token_count = 0
        for compaction_level, variant in enumerate(variants):
            observation_text = format_observation(obs, **variant)
            messages = self._build_messages_from_text(observation_text)
            prompt = self._render_prompt(messages)
            token_count = self._prompt_token_length(prompt)
            if compaction_level == 0:
                original_token_count = token_count
            last_token_count = token_count
            if token_count <= self.max_seq_length:
                return PromptBuild(
                    prompt=prompt,
                    messages=messages,
                    original_token_count=original_token_count,
                    final_token_count=token_count,
                    compaction_level=compaction_level,
                )
        raise RuntimeError(
            "Structured prompt compaction could not satisfy the frozen prompt limit: "
            f"original={original_token_count}, compact={last_token_count}, "
            f"limit={self.max_seq_length}. Refusing silent token truncation."
        )

    def act(
        self,
        obs: EnvironmentObservation,
        deterministic: bool = False,
        max_new_tokens: int | None = None,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> GenerationTrace:
        prompt_build = self.build_prompt(obs)
        effective_max_new_tokens = (
            self.max_new_tokens if max_new_tokens is None else max_new_tokens
        )
        if not 1 <= effective_max_new_tokens <= self.max_seq_length:
            raise ValueError("max_new_tokens is outside the frozen generation contract")
        inputs = self.tokenizer(
            prompt_build.prompt,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False,
        ).to(self.model.device)
        input_token_count = int(inputs["input_ids"].shape[1])
        if input_token_count != prompt_build.final_token_count:
            raise RuntimeError("Prompt token accounting changed between fitting and inference")
        if input_token_count > self.max_seq_length:
            raise RuntimeError("Prompt exceeded the frozen limit after structured fitting")

        generation_config = copy.deepcopy(self.model.generation_config)
        generation_config.max_new_tokens = effective_max_new_tokens
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        if deterministic:
            generation_config.do_sample = False
            generation_config.temperature = None
            generation_config.top_p = None
            generation_config.top_k = None
        else:
            generation_config.do_sample = True
            generation_config.temperature = temperature
            generation_config.top_p = top_p

        with torch.no_grad():
            output = self.model.generate(**inputs, generation_config=generation_config)

        raw_text = self.tokenizer.decode(
            output[0][inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        action = parse_llm_action(raw_text)
        return GenerationTrace(
            action=action,
            raw_text=raw_text,
            prompt=prompt_build.prompt,
            messages=prompt_build.messages,
            original_prompt_tokens=prompt_build.original_token_count,
            prompt_tokens=input_token_count,
            prompt_limit=self.max_seq_length,
            max_new_tokens=effective_max_new_tokens,
            compaction_level=prompt_build.compaction_level,
            token_truncated=False,
        )

    def close(self) -> None:
        del self.model
        cleanup_cuda()
