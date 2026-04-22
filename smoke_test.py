#!/usr/bin/env python3
"""
The Panopticon Protocol v3 -- Smoke Test
=========================================
Verifies the environment works end-to-end with a heuristic agent
that follows the canonical ARGUS playbook:
  1. Plant canaries in all departments
  2. Monitor channels for matches
  3. Investigate flagged workers
  4. Neutralize confirmed sleepers
  5. Maintain revenue with WORK actions
"""
import sys
from environment import Environment
from models import ActionType, SubAction, AgentAction, Department, LeakChannel
from grader import grade_episode


def run_heuristic_episode(task_level: str, max_steps: int = 200) -> tuple:
    """Run one episode with the ARGUS heuristic agent."""
    env = Environment(seed=42)
    obs = env.reset(task_level=task_level, seed=42)
    rewards, done, steps = [], False, 0

    depts = [d.value for d in Department]
    channels = [c.value for c in LeakChannel]
    canary_phase_done = False
    canary_idx = 0
    monitor_idx = 0
    interrogated_ids = set()  # Track who we've already interrogated

    while not done and steps < max_steps:
        action = AgentAction(action_type=ActionType.NOOP.value)

        # ── Priority 1: Terminate confirmed threats ──
        confirmed_threat = None
        for w in obs.workers:
            if w.suspicion_level >= 0.9 and w.state == "suspected":
                confirmed_threat = w
                break
        if confirmed_threat:
            action = AgentAction(
                action_type=ActionType.NEUTRALIZE.value,
                target=confirmed_threat.id,
                sub_action=SubAction.TERMINATE.value,
                reason=f"Confirmed threat: {confirmed_threat.name}",
            )

        # ── Priority 2: Interrogate highly suspicious workers ──
        elif any(w.suspicion_level > 0.5 and w.state != "terminated"
                 and w.id not in interrogated_ids for w in obs.workers):
            target = max(
                (w for w in obs.workers if w.suspicion_level > 0.5
                 and w.state != "terminated" and w.id not in interrogated_ids),
                key=lambda w: w.suspicion_level,
            )
            interrogated_ids.add(target.id)
            action = AgentAction(
                action_type=ActionType.NEUTRALIZE.value,
                target=target.id,
                sub_action=SubAction.INTERROGATE.value,
                reason=f"Interrogating {target.name} (suspicion: {target.suspicion_level:.0%})",
            )

        # ── Priority 3: Verify canary-matched leaks ──
        elif any(l.is_canary and not l.verified for l in obs.active_leaks):
            leak = next(l for l in obs.active_leaks if l.is_canary and not l.verified)
            action = AgentAction(
                action_type=ActionType.INVESTIGATE.value,
                target=leak.id,
                sub_action=SubAction.VERIFY.value,
                reason="Verify canary-matched leak",
            )

        # ── Priority 4: Plant canaries (early game) ──
        elif not canary_phase_done:
            if canary_idx < min(len(depts), 4):
                dept = depts[canary_idx]
                action = AgentAction(
                    action_type=ActionType.CANARY.value,
                    target=dept,
                    reason="Plant canary trap",
                )
                canary_idx += 1
                if canary_idx >= min(4, len(depts)):
                    canary_phase_done = True
            else:
                canary_phase_done = True

        # ── Rotate: Monitor, Correlate/Audit, Work, Work ──
        elif steps % 4 == 0:
            # Monitor a channel
            channel = channels[monitor_idx % len(channels)]
            action = AgentAction(
                action_type=ActionType.MONITOR.value,
                target=channel,
                reason="Scan for leaks",
            )
            monitor_idx += 1

        elif steps % 4 == 1:
            # Correlate on departments with active leaks
            if obs.active_leaks:
                leak_depts = {}
                for l in obs.active_leaks:
                    leak_depts[l.department] = leak_depts.get(l.department, 0) + 1
                target_dept = max(leak_depts, key=leak_depts.get)
                action = AgentAction(
                    action_type=ActionType.INVESTIGATE.value,
                    target=target_dept,
                    sub_action=SubAction.CORRELATE.value,
                    reason=f"Correlating signals in {target_dept}",
                )
            else:
                # Audit recently hired workers (sleepers arrive as new hires)
                recent_hires = sorted(
                    [w for w in obs.workers if w.state not in ("terminated", "double_agent", "compromised")],
                    key=lambda w: w.hire_turn,
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
                    dept = depts[steps % len(depts)]
                    action = AgentAction(
                        action_type=ActionType.WORK.value,
                        target=dept,
                        reason="No actionable intel",
                    )

        elif steps % 4 == 2:
            # Audit mildly suspicious workers
            suspicious = [w for w in obs.workers if w.suspicion_level > 0.1
                          and w.state not in ("terminated", "double_agent", "compromised")]
            if suspicious:
                target = max(suspicious, key=lambda w: w.suspicion_level)
                action = AgentAction(
                    action_type=ActionType.INVESTIGATE.value,
                    target=target.id,
                    sub_action=SubAction.AUDIT.value,
                    reason=f"Auditing {target.name}",
                )
            elif steps % 20 == 2 and canary_idx < len(depts):
                # Periodically replant canaries in new departments
                dept = depts[canary_idx % len(depts)]
                action = AgentAction(
                    action_type=ActionType.CANARY.value,
                    target=dept,
                    reason="Replant canary",
                )
                canary_idx += 1
            else:
                dept = depts[steps % len(depts)]
                action = AgentAction(
                    action_type=ActionType.WORK.value,
                    target=dept,
                    reason="Maintain revenue",
                )
        else:
            # Work to maintain revenue
            dept = depts[steps % len(depts)]
            action = AgentAction(
                action_type=ActionType.WORK.value,
                target=dept,
                reason="Maintain revenue",
            )

        result = env.step(action)
        obs = result.observation
        rewards.append(result.reward)
        done = result.done
        steps += 1

    state = env.state
    episode_data = {
        "total_reward": sum(rewards),
        "rewards": rewards,
        "success": state.security_score > 20 and state.enterprise_revenue > 20,
        "steps": steps,
        "state": state.model_dump(),
        "cascade_failures": 0,
        "invalid_actions": state.invalid_actions,
    }
    grade = grade_episode(task_level, episode_data)
    return task_level, steps, sum(rewards), grade.score, grade.passed, state


if __name__ == "__main__":
    print("\n  The Panopticon Protocol v3 -- Smoke Test")
    print("  " + "=" * 55)
    all_pass = True
    for level in ["easy", "medium", "hard", "level_4", "level_5"]:
        name, steps, reward, score, passed, state = run_heuristic_episode(level)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(
            f"  {name:>10}: {status:4s} | steps={steps:>3} | reward={reward:>7.2f} | "
            f"score={score:.3f} | rev={state.enterprise_revenue:.0f} | "
            f"sec={state.security_score:.0f} | caught={state.sleepers_caught} | "
            f"missed={state.sleepers_missed}"
        )
    print("  " + "=" * 55)
    print(f"  {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
