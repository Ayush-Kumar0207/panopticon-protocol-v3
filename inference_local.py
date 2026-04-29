#!/usr/bin/env python3
"""Local inference and showcase export for the fine-tuned ARGUS model."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from environment import Environment
from grader import grade_episode
from models import (
    ActionType,
    AgentAction,
    Department,
    EnvironmentObservation,
    LeakChannel,
    SubAction,
    validate_action,
)

if TYPE_CHECKING:
    from argus_llm import LocalArgusModel

DEFAULT_MODEL = "Ayush-Kumar0207/panopticon-argus-qwen-1.5B"
LEVELS = ["easy", "medium", "hard", "level_4", "level_5"]


@dataclass
class PolicyDecision:
    action: AgentAction
    raw_text: str = ""
    prompt: str = ""
    messages: list[dict[str, str]] = field(default_factory=list)
    policy_name: str = ""


class EpisodePolicy(Protocol):
    policy_name: str

    def reset(self) -> None:
        """Reset any per-episode internal state."""

    def act(self, obs: EnvironmentObservation) -> PolicyDecision:
        """Produce the next action for the current observation."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.pstdev(values)


def state_success(state: dict[str, Any]) -> bool:
    return state.get("security_score", 0.0) > 20 and state.get("enterprise_revenue", 0.0) > 20


def compact_observation(obs: EnvironmentObservation) -> dict[str, Any]:
    return obs.model_dump()


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return to_builtin(value.tolist())
        except TypeError:
            pass
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, Path):
        return str(value)
    return value


def serialize_action(action: AgentAction) -> dict[str, Any]:
    return action.model_dump()


def make_policy_decision(
    action: AgentAction,
    policy_name: str,
    raw_text: str = "",
    prompt: str = "",
    messages: list[dict[str, str]] | None = None,
) -> PolicyDecision:
    return PolicyDecision(
        action=action,
        raw_text=raw_text,
        prompt=prompt,
        messages=messages or [],
        policy_name=policy_name,
    )


def enumerate_valid_actions(obs: EnvironmentObservation) -> list[AgentAction]:
    actions: list[AgentAction] = [AgentAction(action_type=ActionType.NOOP.value, reason="Baseline NOOP")]

    departments = [dept.value for dept in Department]
    channels = [channel.value for channel in LeakChannel]
    live_workers = [worker for worker in obs.workers if worker.state != "terminated"]
    active_double_agents = [asset.worker_id for asset in obs.double_agents if asset.active]

    for dept in departments:
        actions.append(AgentAction(action_type=ActionType.WORK.value, target=dept, reason="Random baseline"))
        actions.append(AgentAction(action_type=ActionType.HIRE.value, target=dept, reason="Random baseline"))
        actions.append(AgentAction(action_type=ActionType.CANARY.value, target=dept, reason="Random baseline"))
        actions.append(
            AgentAction(
                action_type=ActionType.INVESTIGATE.value,
                target=dept,
                sub_action=SubAction.CORRELATE.value,
                reason="Random baseline",
            )
        )

    for channel in channels:
        actions.append(AgentAction(action_type=ActionType.MONITOR.value, target=channel, reason="Random baseline"))

    for worker in live_workers:
        actions.append(
            AgentAction(
                action_type=ActionType.INVESTIGATE.value,
                target=worker.id,
                sub_action=SubAction.AUDIT.value,
                reason="Random baseline",
            )
        )
        for sub_action in (SubAction.TERMINATE, SubAction.INTERROGATE, SubAction.TURN):
            actions.append(
                AgentAction(
                    action_type=ActionType.NEUTRALIZE.value,
                    target=worker.id,
                    sub_action=sub_action.value,
                    reason="Random baseline",
                )
            )

    for leak in obs.active_leaks:
        actions.append(
            AgentAction(
                action_type=ActionType.INVESTIGATE.value,
                target=leak.id,
                sub_action=SubAction.VERIFY.value,
                reason="Random baseline",
            )
        )

    for worker_id in active_double_agents:
        actions.append(
            AgentAction(
                action_type=ActionType.DEPLOY_DOUBLE.value,
                target=worker_id,
                reason="Random baseline",
            )
        )

    return [action for action in actions if validate_action(action, obs)[0]]


def _safe_fallback_action(obs: EnvironmentObservation) -> AgentAction:
    canary_leak = next((leak for leak in obs.active_leaks if leak.is_canary and not leak.verified), None)
    if canary_leak:
        return AgentAction(
            action_type=ActionType.INVESTIGATE.value,
            target=canary_leak.id,
            sub_action=SubAction.VERIFY.value,
            reason="Recovered: verify triggered canary leak",
        )

    suspicious_worker = max(
        obs.workers,
        key=lambda worker: worker.suspicion_level,
        default=None,
    )
    if suspicious_worker and suspicious_worker.suspicion_level > 0.2:
        return AgentAction(
            action_type=ActionType.INVESTIGATE.value,
            target=suspicious_worker.id,
            sub_action=SubAction.AUDIT.value,
            reason="Recovered: audit highest-suspicion worker",
        )

    planted_departments = {trap.department for trap in obs.canary_traps if trap.active}
    for dept in [dept.value for dept in Department]:
        if dept not in planted_departments:
            return AgentAction(
                action_type=ActionType.CANARY.value,
                target=dept,
                reason="Recovered: expand canary coverage",
            )

    if obs.active_leaks:
        most_recent = max(obs.active_leaks, key=lambda leak: leak.turn_detected)
        return AgentAction(
            action_type=ActionType.MONITOR.value,
            target=most_recent.channel,
            reason="Recovered: monitor latest active leak channel",
        )

    return AgentAction(
        action_type=ActionType.WORK.value,
        target=Department.ENGINEERING.value,
        reason="Recovered: maintain revenue while awaiting stronger signals",
    )


def _recover_action_from_text(raw_text: str, obs: EnvironmentObservation) -> AgentAction | None:
    text = (raw_text or "").lower()
    departments = {dept.value for dept in Department}
    worker_ids = {worker.id for worker in obs.workers if worker.state != "terminated"}
    leak_ids = {leak.id for leak in obs.active_leaks}
    double_agents = [asset.worker_id for asset in obs.double_agents if asset.active]

    worker_match = re.search(r"w-\d{3}", text)
    leak_match = re.search(r"leak-\d{3}", text)
    department_match = next((dept for dept in departments if dept in text), "")

    if "verify" in text and leak_match and leak_match.group(0) in leak_ids:
        return AgentAction(
            action_type=ActionType.INVESTIGATE.value,
            target=leak_match.group(0),
            sub_action=SubAction.VERIFY.value,
            reason="Recovered verify action from model trace",
        )

    if "audit" in text and worker_match and worker_match.group(0) in worker_ids:
        return AgentAction(
            action_type=ActionType.INVESTIGATE.value,
            target=worker_match.group(0),
            sub_action=SubAction.AUDIT.value,
            reason="Recovered audit action from model trace",
        )

    if "correlate" in text and department_match:
        return AgentAction(
            action_type=ActionType.INVESTIGATE.value,
            target=department_match,
            sub_action=SubAction.CORRELATE.value,
            reason="Recovered correlate action from model trace",
        )

    if "terminate" in text and worker_match and worker_match.group(0) in worker_ids:
        return AgentAction(
            action_type=ActionType.NEUTRALIZE.value,
            target=worker_match.group(0),
            sub_action=SubAction.TERMINATE.value,
            reason="Recovered terminate action from model trace",
        )

    if "interrogate" in text and worker_match and worker_match.group(0) in worker_ids:
        return AgentAction(
            action_type=ActionType.NEUTRALIZE.value,
            target=worker_match.group(0),
            sub_action=SubAction.INTERROGATE.value,
            reason="Recovered interrogate action from model trace",
        )

    if "turn" in text and worker_match and worker_match.group(0) in worker_ids:
        return AgentAction(
            action_type=ActionType.NEUTRALIZE.value,
            target=worker_match.group(0),
            sub_action=SubAction.TURN.value,
            reason="Recovered turn action from model trace",
        )

    if "deploy_double" in text and double_agents:
        return AgentAction(
            action_type=ActionType.DEPLOY_DOUBLE.value,
            target=double_agents[0],
            reason="Recovered deploy_double action from model trace",
        )

    return None


def repair_trained_action(action: AgentAction, raw_text: str, obs: EnvironmentObservation) -> AgentAction:
    departments = {dept.value for dept in Department}
    channels = {channel.value for channel in LeakChannel}
    worker_ids = {worker.id for worker in obs.workers if worker.state != "terminated"}
    leak_ids = {leak.id for leak in obs.active_leaks}
    active_double_agents = [asset.worker_id for asset in obs.double_agents if asset.active]

    normalized = AgentAction(
        action_type=(action.action_type or "noop").strip().lower(),
        target=(action.target or "").strip().lower(),
        sub_action=(action.sub_action or "none").strip().lower(),
        reason=action.reason or "",
    )

    action_aliases = {
        "analyse": ActionType.INVESTIGATE.value,
        "analyze": ActionType.INVESTIGATE.value,
        "scan": ActionType.MONITOR.value,
        "scan_for_leaks": ActionType.MONITOR.value,
        "generate_revenue": ActionType.WORK.value,
    }
    normalized.action_type = action_aliases.get(normalized.action_type, normalized.action_type)

    if normalized.action_type == ActionType.NOOP.value and (raw_text.strip() or "parse failure" in normalized.reason.lower()):
        recovered = _recover_action_from_text(raw_text, obs)
        if recovered is not None and validate_action(recovered, obs)[0]:
            return recovered
        return _safe_fallback_action(obs)

    if normalized.action_type == ActionType.MONITOR.value:
        if normalized.target in leak_ids:
            normalized = AgentAction(
                action_type=ActionType.INVESTIGATE.value,
                target=normalized.target,
                sub_action=SubAction.VERIFY.value,
                reason=normalized.reason or "Repair: verify referenced leak",
            )
        elif normalized.target in departments:
            normalized = AgentAction(
                action_type=ActionType.INVESTIGATE.value,
                target=normalized.target,
                sub_action=SubAction.CORRELATE.value,
                reason=normalized.reason or "Repair: correlate on referenced department",
            )

    if normalized.action_type == ActionType.INVESTIGATE.value:
        if normalized.sub_action not in {
            SubAction.AUDIT.value,
            SubAction.VERIFY.value,
            SubAction.CORRELATE.value,
        }:
            if normalized.target in leak_ids:
                normalized.sub_action = SubAction.VERIFY.value
            elif normalized.target in worker_ids:
                normalized.sub_action = SubAction.AUDIT.value
            elif normalized.target in departments:
                normalized.sub_action = SubAction.CORRELATE.value

    if normalized.action_type == ActionType.NEUTRALIZE.value:
        if normalized.sub_action not in {
            SubAction.TERMINATE.value,
            SubAction.INTERROGATE.value,
            SubAction.TURN.value,
        } and normalized.target in worker_ids:
            normalized.sub_action = SubAction.INTERROGATE.value

    if normalized.action_type == ActionType.DEPLOY_DOUBLE.value and normalized.target not in active_double_agents:
        if active_double_agents:
            normalized.target = active_double_agents[0]

    if normalized.action_type not in {action_type.value for action_type in ActionType}:
        recovered = _recover_action_from_text(raw_text, obs)
        if recovered is not None:
            normalized = recovered

    valid, _ = validate_action(normalized, obs)
    if valid:
        return normalized

    recovered = _recover_action_from_text(raw_text, obs)
    if recovered is not None and validate_action(recovered, obs)[0]:
        return recovered

    return _safe_fallback_action(obs)


class LocalModelPolicy:
    def __init__(
        self,
        model_ref: str,
        deterministic: bool = False,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ):
        from argus_llm import LocalArgusModel

        self.runtime = LocalArgusModel(model_ref)
        self.deterministic = deterministic
        self.temperature = temperature
        self.top_p = top_p
        self.policy_name = "trained"

    def reset(self) -> None:
        return None

    def act(self, obs: EnvironmentObservation) -> PolicyDecision:
        trace = self.runtime.act(
            obs,
            deterministic=self.deterministic,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        repaired_action = repair_trained_action(trace.action, trace.raw_text, obs)
        return make_policy_decision(
            action=repaired_action,
            policy_name=self.policy_name,
            raw_text=trace.raw_text,
            prompt=trace.prompt,
            messages=trace.messages,
        )

    def model_info(self) -> dict[str, Any]:
        return self.runtime.model_info()

    def close(self) -> None:
        self.runtime.close()


class RandomPolicy:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.policy_name = "random"

    def reset(self) -> None:
        return None

    def act(self, obs: EnvironmentObservation) -> PolicyDecision:
        valid_actions = enumerate_valid_actions(obs)
        action = self.rng.choice(valid_actions) if valid_actions else AgentAction(action_type=ActionType.NOOP.value)
        return make_policy_decision(
            action=action,
            policy_name=self.policy_name,
            raw_text=json.dumps(serialize_action(action)),
        )


class HeuristicPolicy:
    def __init__(self):
        self.policy_name = "heuristic"
        self._depts = [dept.value for dept in Department]
        self._channels = [channel.value for channel in LeakChannel]
        self.reset()

    def reset(self) -> None:
        self._turn = 0
        self._canary_phase_done = False
        self._canary_idx = 0
        self._monitor_idx = 0
        self._interrogated_ids: set[str] = set()

    def act(self, obs: EnvironmentObservation) -> PolicyDecision:
        self._turn += 1
        action = AgentAction(action_type=ActionType.NOOP.value, reason="Heuristic fallback")

        confirmed = next(
            (
                worker
                for worker in obs.workers
                if worker.suspicion_level >= 0.9 and worker.state == "suspected"
            ),
            None,
        )
        if confirmed:
            action = AgentAction(
                action_type=ActionType.NEUTRALIZE.value,
                target=confirmed.id,
                sub_action=SubAction.TERMINATE.value,
                reason=f"Confirmed threat: {confirmed.name}",
            )
        elif any(
            worker.suspicion_level > 0.5
            and worker.state != "terminated"
            and worker.id not in self._interrogated_ids
            for worker in obs.workers
        ):
            target = max(
                (
                    worker
                    for worker in obs.workers
                    if worker.suspicion_level > 0.5
                    and worker.state != "terminated"
                    and worker.id not in self._interrogated_ids
                ),
                key=lambda worker: worker.suspicion_level,
            )
            self._interrogated_ids.add(target.id)
            action = AgentAction(
                action_type=ActionType.NEUTRALIZE.value,
                target=target.id,
                sub_action=SubAction.INTERROGATE.value,
                reason=f"Interrogating {target.name}",
            )
        elif any(leak.is_canary and not leak.verified for leak in obs.active_leaks):
            leak = next(leak for leak in obs.active_leaks if leak.is_canary and not leak.verified)
            action = AgentAction(
                action_type=ActionType.INVESTIGATE.value,
                target=leak.id,
                sub_action=SubAction.VERIFY.value,
                reason="Verify canary-matched leak",
            )
        elif not self._canary_phase_done:
            if self._canary_idx < min(len(self._depts), 4):
                dept = self._depts[self._canary_idx]
                action = AgentAction(
                    action_type=ActionType.CANARY.value,
                    target=dept,
                    reason="Plant canary trap",
                )
                self._canary_idx += 1
                if self._canary_idx >= min(4, len(self._depts)):
                    self._canary_phase_done = True
            else:
                self._canary_phase_done = True
        elif self._turn % 4 == 0:
            channel = self._channels[self._monitor_idx % len(self._channels)]
            action = AgentAction(
                action_type=ActionType.MONITOR.value,
                target=channel,
                reason="Scan for leaks",
            )
            self._monitor_idx += 1
        elif self._turn % 4 == 1:
            if obs.active_leaks:
                leak_departments: dict[str, int] = {}
                for leak in obs.active_leaks:
                    leak_departments[leak.department] = leak_departments.get(leak.department, 0) + 1
                target_dept = max(leak_departments, key=leak_departments.get)
                action = AgentAction(
                    action_type=ActionType.INVESTIGATE.value,
                    target=target_dept,
                    sub_action=SubAction.CORRELATE.value,
                    reason=f"Correlating signals in {target_dept}",
                )
            else:
                recent_hires = sorted(
                    [
                        worker
                        for worker in obs.workers
                        if worker.state not in ("terminated", "double_agent", "compromised")
                    ],
                    key=lambda worker: worker.hire_turn,
                    reverse=True,
                )
                if recent_hires:
                    target = recent_hires[0]
                    action = AgentAction(
                        action_type=ActionType.INVESTIGATE.value,
                        target=target.id,
                        sub_action=SubAction.AUDIT.value,
                        reason=f"Auditing recent hire {target.name}",
                    )
                else:
                    dept = self._depts[self._turn % len(self._depts)]
                    action = AgentAction(
                        action_type=ActionType.WORK.value,
                        target=dept,
                        reason="No actionable intel",
                    )
        elif self._turn % 4 == 2:
            suspicious = [
                worker
                for worker in obs.workers
                if worker.suspicion_level > 0.1
                and worker.state not in ("terminated", "double_agent", "compromised")
            ]
            if suspicious:
                target = max(suspicious, key=lambda worker: worker.suspicion_level)
                action = AgentAction(
                    action_type=ActionType.INVESTIGATE.value,
                    target=target.id,
                    sub_action=SubAction.AUDIT.value,
                    reason=f"Auditing {target.name}",
                )
            elif self._turn % 20 == 2 and self._canary_idx < len(self._depts):
                dept = self._depts[self._canary_idx % len(self._depts)]
                action = AgentAction(
                    action_type=ActionType.CANARY.value,
                    target=dept,
                    reason="Replant canary",
                )
                self._canary_idx += 1
            else:
                dept = self._depts[self._turn % len(self._depts)]
                action = AgentAction(
                    action_type=ActionType.WORK.value,
                    target=dept,
                    reason="Maintain revenue",
                )
        else:
            dept = self._depts[self._turn % len(self._depts)]
            action = AgentAction(
                action_type=ActionType.WORK.value,
                target=dept,
                reason="Maintain revenue",
            )

        return make_policy_decision(
            action=action,
            policy_name=self.policy_name,
            raw_text=json.dumps(serialize_action(action)),
        )


def build_grade_payload(state: dict[str, Any], rewards: list[float], steps: int) -> dict[str, Any]:
    total_reward = sum(rewards)
    return {
        "total_reward": total_reward,
        "rewards": rewards,
        "success": state_success(state),
        "steps": steps,
        "state": state,
        "cascade_failures": 0,
        "invalid_actions": state.get("invalid_actions", 0),
    }


def summarize_level_results(level: str, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    grades = [episode["grade"]["score"] for episode in episodes]
    rewards = [episode["total_reward"] for episode in episodes]
    revenues = [episode["final_state"]["enterprise_revenue"] for episode in episodes]
    securities = [episode["final_state"]["security_score"] for episode in episodes]
    caught = [episode["final_state"]["sleepers_caught"] for episode in episodes]
    missed = [episode["final_state"]["sleepers_missed"] for episode in episodes]
    invalid = [episode["final_state"]["invalid_actions"] for episode in episodes]
    steps = [episode["steps"] for episode in episodes]
    pass_rate = sum(1 for episode in episodes if episode["grade"]["passed"]) / max(len(episodes), 1)

    dimension_names = ["security", "revenue", "intelligence", "adaptability", "efficiency"]
    dimension_summary = {}
    for name in dimension_names:
        values = [episode["grade"]["dimensions"][name] for episode in episodes]
        value_mean, value_std = mean_std(values)
        dimension_summary[name] = {"mean": value_mean, "std": value_std}

    grade_mean, grade_std = mean_std(grades)
    reward_mean, reward_std = mean_std(rewards)
    revenue_mean, revenue_std = mean_std(revenues)
    security_mean, security_std = mean_std(securities)
    caught_mean, caught_std = mean_std(caught)
    missed_mean, missed_std = mean_std(missed)
    invalid_mean, invalid_std = mean_std(invalid)
    steps_mean, steps_std = mean_std(steps)

    return {
        "level": level,
        "episodes": len(episodes),
        "pass_rate": pass_rate,
        "grade_mean": grade_mean,
        "grade_std": grade_std,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "revenue_mean": revenue_mean,
        "revenue_std": revenue_std,
        "security_mean": security_mean,
        "security_std": security_std,
        "sleepers_caught_mean": caught_mean,
        "sleepers_caught_std": caught_std,
        "sleepers_missed_mean": missed_mean,
        "sleepers_missed_std": missed_std,
        "invalid_actions_mean": invalid_mean,
        "invalid_actions_std": invalid_std,
        "steps_mean": steps_mean,
        "steps_std": steps_std,
        "grader_dimensions": dimension_summary,
    }


def select_representative_episode(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        return {}
    return max(episodes, key=lambda episode: episode["grade"]["score"])


def run_episode(
    policy: EpisodePolicy,
    task_level: str,
    seed: int,
    max_steps: int = 300,
    verbose: bool = True,
) -> dict[str, Any]:
    policy.reset()
    env = Environment(seed=seed)
    obs = env.reset(task_level=task_level, seed=seed)

    rewards: list[float] = []
    timeline: list[dict[str, Any]] = []
    steps = 0

    while not env.state.done and steps < max_steps:
        before = compact_observation(obs)
        decision = policy.act(obs)
        result = env.step(decision.action)
        after = compact_observation(result.observation)

        step_record = {
            "step_index": steps,
            "turn": obs.turn,
            "phase": obs.phase,
            "policy": decision.policy_name,
            "action": serialize_action(decision.action),
            "raw_text": decision.raw_text,
            "prompt": decision.prompt,
            "messages": decision.messages,
            "reward": result.reward,
            "done": result.done,
            "truncated": result.truncated,
            "info": result.info,
            "observation_before": before,
            "observation_after": after,
            "metrics_after": {
                "enterprise_revenue": after["enterprise_revenue"],
                "security_score": after["security_score"],
                "active_leaks": len(after["active_leaks"]),
                "double_agents": len(after["double_agents"]),
            },
        }
        timeline.append(step_record)

        if verbose and (steps == 0 or (steps + 1) % 10 == 0 or not result.info.get("valid", True) or result.done):
            print(
                f"  T{obs.turn:>3} | {decision.action.action_type:<13} {decision.action.target:<16} "
                f"| Reward={result.reward:>6.2f} | Rev={after['enterprise_revenue']:>7.2f} "
                f"| Sec={after['security_score']:>6.2f}"
            )

        rewards.append(result.reward)
        obs = result.observation
        steps += 1

    state = env.state.model_dump()
    grade = grade_episode(task_level, build_grade_payload(state, rewards, steps))

    return {
        "level": task_level,
        "seed": seed,
        "policy": getattr(policy, "policy_name", "unknown"),
        "steps": steps,
        "total_reward": sum(rewards),
        "reward_history": rewards,
        "grade": grade.to_dict(),
        "timeline": timeline,
        "final_state": state,
    }


def print_level_summary(level: str, summary: dict[str, Any]) -> None:
    print(
        f"  {level:<8} | grade={summary['grade_mean']:.3f} +/- {summary['grade_std']:.3f} "
        f"| reward={summary['reward_mean']:.2f} | rev={summary['revenue_mean']:.1f} "
        f"| sec={summary['security_mean']:.1f} | caught={summary['sleepers_caught_mean']:.2f} "
        f"| pass={summary['pass_rate']:.0%}"
    )


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local inference for the Panopticon ARGUS model")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF repo, local merged model, or adapter directory")
    parser.add_argument("--level", default="all", choices=["all", *LEVELS], help="Difficulty level to evaluate")
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per level")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument("--output", default="", help="Optional JSON output path")
    parser.add_argument("--deterministic", action="store_true", help="Use greedy decoding")
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum steps per episode")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature when not deterministic")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter when not deterministic")
    parser.add_argument("--quiet", action="store_true", help="Reduce per-turn console output")
    return parser


def main() -> None:
    args = build_cli().parse_args()
    levels = LEVELS if args.level == "all" else [args.level]
    seed_rng = random.Random(args.seed)

    policy = LocalModelPolicy(
        args.model,
        deterministic=args.deterministic,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    try:
        level_results: dict[str, list[dict[str, Any]]] = {}
        level_summaries: dict[str, dict[str, Any]] = {}
        showcase: dict[str, dict[str, Any]] = {}

        print(f"[*] Loaded model: {args.model}")
        print(f"[*] Levels: {levels} | Episodes per level: {args.episodes}")

        for level in levels:
            print(f"\n[Level] {level}")
            episodes: list[dict[str, Any]] = []
            for episode_idx in range(args.episodes):
                episode_seed = seed_rng.randint(0, 999999)
                print(f"  Episode {episode_idx + 1}/{args.episodes} | seed={episode_seed}")
                episode = run_episode(
                    policy,
                    task_level=level,
                    seed=episode_seed,
                    max_steps=args.max_steps,
                    verbose=not args.quiet,
                )
                episodes.append(episode)
                print(
                    f"    -> grade={episode['grade']['score']:.3f} | passed={episode['grade']['passed']} "
                    f"| reward={episode['total_reward']:.2f} | rev={episode['final_state']['enterprise_revenue']:.1f} "
                    f"| sec={episode['final_state']['security_score']:.1f}"
                )

            level_results[level] = episodes
            level_summaries[level] = summarize_level_results(level, episodes)
            showcase[level] = select_representative_episode(episodes)
            print_level_summary(level, level_summaries[level])

        payload = {
            "schema_version": 1,
            "created_at": utc_now_iso(),
            "model_info": policy.model_info(),
            "config": {
                "model": args.model,
                "levels": levels,
                "episodes_per_level": args.episodes,
                "seed": args.seed,
                "deterministic": args.deterministic,
                "max_steps": args.max_steps,
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
            "summary": level_summaries,
            "representative_runs": showcase,
            "episodes": level_results,
        }

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(to_builtin(payload), indent=2), encoding="utf-8")
            print(f"\n[*] Wrote inference results to {output_path}")
    finally:
        policy.close()


if __name__ == "__main__":
    main()
