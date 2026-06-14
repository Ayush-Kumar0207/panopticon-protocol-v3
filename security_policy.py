"""Security-first expert policy used to generate safe curriculum demonstrations."""

from __future__ import annotations

from models import ActionType, AgentAction, EnvironmentObservation, LeakChannel, SubAction

ADVANCED_LEVELS = {"level_4", "level_5"}
RESOLVED_WORKER_STATES = {"terminated", "double_agent", "compromised"}


def new_security_expert_state() -> dict:
    return {
        "audit_idx": 0,
        "monitor_idx": 0,
        "canaried_departments": set(),
        "turned_sleeper": False,
    }


def active_departments(obs: EnvironmentObservation) -> list[str]:
    departments: list[str] = []
    for worker in obs.workers:
        if worker.department not in departments:
            departments.append(worker.department)
    for trap in obs.canary_traps:
        if trap.department not in departments:
            departments.append(trap.department)
    return departments


def choose_security_first_action(
    obs: EnvironmentObservation,
    task_level: str,
    expert_state: dict,
) -> AgentAction:
    """Choose an observable, security-first action for expert SFT data."""
    departments = active_departments(obs)
    channels = [channel.value for channel in LeakChannel]

    confirmed = next(
        (
            worker
            for worker in obs.workers
            if worker.suspicion_level >= 0.9
            and worker.state == "suspected"
            and not worker.turning_in_progress
        ),
        None,
    )
    if confirmed:
        enough_runway = (obs.max_turns - obs.turn) >= 8
        should_turn = (
            task_level in ADVANCED_LEVELS
            and not expert_state["turned_sleeper"]
            and enough_runway
        )
        if should_turn:
            expert_state["turned_sleeper"] = True
            return AgentAction(
                action_type=ActionType.NEUTRALIZE.value,
                target=confirmed.id,
                sub_action=SubAction.TURN.value,
                reason=f"Turn confirmed sleeper {confirmed.name} for controlled counterstrike",
            )
        return AgentAction(
            action_type=ActionType.NEUTRALIZE.value,
            target=confirmed.id,
            sub_action=SubAction.TERMINATE.value,
            reason=f"Immediately remove confirmed threat {confirmed.name}",
        )

    suspicious = max(
        (
            worker
            for worker in obs.workers
            if worker.suspicion_level > 0.5
            and worker.state not in RESOLVED_WORKER_STATES
            and not worker.turning_in_progress
        ),
        key=lambda worker: worker.suspicion_level,
        default=None,
    )
    if suspicious:
        return AgentAction(
            action_type=ActionType.NEUTRALIZE.value,
            target=suspicious.id,
            sub_action=SubAction.INTERROGATE.value,
            reason=f"Interrogate high-confidence suspect {suspicious.name}",
        )

    canary_leak = next(
        (leak for leak in obs.active_leaks if leak.is_canary and not leak.verified),
        None,
    )
    if canary_leak:
        return AgentAction(
            action_type=ActionType.INVESTIGATE.value,
            target=canary_leak.id,
            sub_action=SubAction.VERIFY.value,
            reason="Verify triggered canary before enforcement",
        )

    planted = expert_state["canaried_departments"]
    for department in departments:
        if department not in planted:
            planted.add(department)
            return AgentAction(
                action_type=ActionType.CANARY.value,
                target=department,
                reason="Establish complete early-warning coverage",
            )

    active_double_agents = [asset for asset in obs.double_agents if asset.active]
    if (
        active_double_agents
        and obs.phase_number >= 5
        and obs.security_score >= 95
        and obs.turn % 10 == 0
    ):
        asset = max(
            active_double_agents,
            key=lambda item: (item.hydra_trust, item.effectiveness, -item.disinfo_fed_count),
        )
        return AgentAction(
            action_type=ActionType.DEPLOY_DOUBLE.value,
            target=asset.worker_id,
            reason="Deploy disinformation only while security is under control",
        )

    if obs.active_leaks and obs.turn % 3 != 2:
        latest_leak = max(obs.active_leaks, key=lambda leak: leak.turn_detected)
        return AgentAction(
            action_type=ActionType.INVESTIGATE.value,
            target=latest_leak.department,
            sub_action=SubAction.CORRELATE.value,
            reason=f"Correlate the latest unresolved leak in {latest_leak.department}",
        )

    if obs.turn % 4 == 0:
        recent_workers = sorted(
            (
                worker
                for worker in obs.workers
                if worker.state not in RESOLVED_WORKER_STATES
                and not worker.turning_in_progress
            ),
            key=lambda worker: (worker.hire_turn, worker.suspicion_level),
            reverse=True,
        )
        if recent_workers:
            audit_pool = recent_workers[: min(4, len(recent_workers))]
            target = audit_pool[expert_state["audit_idx"] % len(audit_pool)]
            expert_state["audit_idx"] += 1
            return AgentAction(
                action_type=ActionType.INVESTIGATE.value,
                target=target.id,
                sub_action=SubAction.AUDIT.value,
                reason=f"Audit recent high-risk hire {target.name}",
            )

    if obs.turn % 4 == 1:
        channel = channels[expert_state["monitor_idx"] % len(channels)]
        expert_state["monitor_idx"] += 1
        return AgentAction(
            action_type=ActionType.MONITOR.value,
            target=channel,
            reason="Maintain rotating external-channel coverage",
        )

    return AgentAction(
        action_type=ActionType.WORK.value,
        target=departments[obs.turn % len(departments)],
        reason="Maintain revenue after security priorities are handled",
    )


def expert_episode_meets_security_gate(metrics: dict) -> bool:
    return (
        metrics.get("security", 0.0) >= 90.0
        and metrics.get("sleepers_missed", 1) == 0
        and metrics.get("false_accusations", 1) == 0
        and metrics.get("sleepers_caught", 0) >= metrics.get("sleepers_spawned", 1)
    )

