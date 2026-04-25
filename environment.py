"""
The Panopticon Protocol v3 — Environment Core
================================================

The main simulation engine. ARGUS (agent) vs HYDRA (adaptive adversary).

6-Phase Game Structure:
  Phase 1: Orientation (T0-30)   — Build baseline, Gen-1 sleeper activates
  Phase 2: First Contact (T30-60) — Canary traps reveal Gen-1, first V-crash
  Phase 3: Escalation (T60-90)   — Gen-2/3 with false flags, deeper V
  Phase 4: Deep Cover (T90-120)  — Gen-4 dead-man's switch, deepest V
  Phase 5: Crisis (T120-140)     — Manchurian candidate, existential threat
  Phase 6: Counterstrike (T140-160) — Double agent payoff, reward SURGE

7 Stacking Mechanics:
  1. Canary Traps       2. Leak Correlation    3. Multi-gen Sleepers
  4. False Flags         5. Dead-man's Switches 6. Double Agent Flip
  7. HYDRA Adaptive Memory
"""

from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass
from typing import Optional

from models import (
    WorkerState, HiddenWorkerState, Department, LeakChannel,
    SleeperGeneration, ActionType, SubAction, GamePhase,
    Worker, LeakEvent, CanaryTrap, IntelReport, DoubleAgentAsset,
    HydraMemory, AgentAction, EnvironmentObservation, EnvironmentState,
    validate_action,
)


# =============================================================================
# STEP RESULT
# =============================================================================

@dataclass
class StepResult:
    """Result from environment.step()."""
    observation: EnvironmentObservation
    reward: float
    done: bool
    truncated: bool
    info: dict


# =============================================================================
# CONSTANTS
# =============================================================================

INVALID_ACTION_PENALTY = -1.0
PHASE_BOUNDARIES = {1: 0, 2: 30, 3: 60, 4: 90, 5: 120, 6: 140}
PHASE_NAMES = {
    1: GamePhase.ORIENTATION, 2: GamePhase.FIRST_CONTACT,
    3: GamePhase.ESCALATION, 4: GamePhase.DEEP_COVER,
    5: GamePhase.CRISIS, 6: GamePhase.COUNTERSTRIKE,
}

# Difficulty presets
DIFFICULTY_CONFIG = {
    "amateur": {
        "max_turns": 60, "num_workers": 6, "max_gen": 1,
        "hydra_aggression": 0.2, "max_phases": 2,
        "departments": ["engineering", "finance", "operations"],
        "sleeper_schedule": {15: 1},  # turn: generation
    },
    "professional": {
        "max_turns": 90, "num_workers": 7, "max_gen": 2,
        "hydra_aggression": 0.35, "max_phases": 3,
        "departments": ["engineering", "finance", "rd", "operations"],
        "sleeper_schedule": {12: 1, 45: 2},
    },
    "spy": {
        "max_turns": 120, "num_workers": 8, "max_gen": 3,
        "hydra_aggression": 0.5, "max_phases": 4,
        "departments": ["engineering", "finance", "rd", "operations", "executive"],
        "sleeper_schedule": {10: 1, 40: 2, 75: 3},
    },
    "cell": {
        "max_turns": 150, "num_workers": 9, "max_gen": 4,
        "hydra_aggression": 0.65, "max_phases": 5,
        "departments": ["engineering", "finance", "rd", "operations", "executive", "legal"],
        "sleeper_schedule": {8: 1, 35: 2, 65: 3, 100: 4},
    },
    "manchurian": {
        "max_turns": 160, "num_workers": 10, "max_gen": 5,
        "hydra_aggression": 0.8, "max_phases": 6,
        "departments": ["engineering", "finance", "rd", "operations", "executive", "legal"],
        "sleeper_schedule": {8: 1, 30: 2, 55: 3, 90: 4, 125: 5},
    },
}

# Map old task levels to new difficulty names
LEVEL_MAP = {
    "easy": "amateur", "medium": "professional", "hard": "spy",
    "level_4": "cell", "level_5": "manchurian",
    "amateur": "amateur", "professional": "professional",
    "spy": "spy", "cell": "cell", "manchurian": "manchurian",
}

# Worker name pool (espionage codenames)
WORKER_NAMES = [
    "ATLAS", "BEACON", "CIPHER", "DAGGER", "ECHO", "FALCON",
    "GRANITE", "HARBINGER", "IRON", "JAVELIN", "KNIGHT", "LAZARUS",
    "MANTIS", "NEXUS", "ORACLE", "PHANTOM", "RAVEN", "SPECTRE",
    "TEMPEST", "UMBRA", "VIPER", "WRAITH", "ZENITH", "AEGIS",
]


# =============================================================================
# ENVIRONMENT
# =============================================================================

class Environment:
    """
    The Panopticon Protocol v3 — Counter-Espionage RL Environment.

    Agent (ARGUS) defends a corporate network against an adaptive adversary
    (HYDRA) that infiltrates sleeper agents of escalating sophistication.
    """

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self._state = EnvironmentState()
        self._task_level = "amateur"
        self._config: dict = DIFFICULTY_CONFIG["amateur"]
        self._worker_counter = 0
        self._leak_counter = 0
        self._canary_counter = 0
        self._report_counter = 0
        self._pending_sleepers: dict[int, int] = {}  # turn -> generation
        self._turning_workers: dict[str, int] = {}   # worker_id -> turns remaining
        self._shuffled_names: list[str] = list(WORKER_NAMES)  # shuffled per episode

    @property
    def state(self) -> EnvironmentState:
        return self._state

    # =========================================================================
    # RESET
    # =========================================================================

    def reset(self, task_level: str = "easy", seed: int | None = None) -> EnvironmentObservation:
        """Reset environment to a new episode."""
        # Always re-seed RNG for genuine randomness between episodes
        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random()  # true random each episode

        difficulty = LEVEL_MAP.get(task_level, "amateur")
        self._task_level = task_level
        self._config = DIFFICULTY_CONFIG[difficulty]
        self._worker_counter = 0
        self._leak_counter = 0
        self._canary_counter = 0
        self._report_counter = 0
        self._pending_sleepers = dict(self._config["sleeper_schedule"])
        self._turning_workers = {}

        # Shuffle the name pool so workers get different names each episode
        self._shuffled_names = list(WORKER_NAMES)
        self._rng.shuffle(self._shuffled_names)

        # Generate initial workforce
        workers = []
        depts = self._config["departments"]
        for i in range(self._config["num_workers"]):
            w = self._create_worker(
                department=depts[i % len(depts)],
                hire_turn=0,
            )
            workers.append(w)

        self._state = EnvironmentState(
            workers=workers,
            leaks=[],
            canary_traps=[],
            intel_reports=[],
            double_agents=[],
            hydra_memory=HydraMemory(),
            enterprise_revenue=100.0,
            peak_revenue=100.0,
            security_score=100.0,
            turn=0,
            max_turns=self._config["max_turns"],
            phase=GamePhase.ORIENTATION.value,
            phase_number=1,
            done=False,
            total_reward=0.0,
            difficulty=difficulty,
            max_sleeper_gen=self._config["max_gen"],
            hydra_aggression=self._config["hydra_aggression"],
            departments_active=list(depts),
            num_initial_workers=self._config["num_workers"],
            revenue_history=[100.0],
            reward_history=[],
            phase_transitions=[],
        )

        return self.get_observation()

    # =========================================================================
    # STEP
    # =========================================================================

    def step(self, action: AgentAction) -> StepResult:
        """Execute one turn: agent acts, then HYDRA acts, then world updates."""
        s = self._state
        info: dict = {"valid": True, "events": []}

        # ── Validate action ──
        obs = self.get_observation()
        valid, reason = validate_action(action, obs)
        if not valid:
            s.invalid_actions += 1
            s.turn += 1
            info["valid"] = False
            info["reason"] = reason
            reward = INVALID_ACTION_PENALTY
            s.total_reward += reward
            s.reward_history.append(reward)
            self._update_phase()
            done = s.turn >= s.max_turns
            s.done = done
            return StepResult(
                observation=self.get_observation(),
                reward=reward, done=done,
                truncated=done, info=info,
            )

        # ── Snapshot metrics before action ──
        rev_before = s.enterprise_revenue
        sec_before = s.security_score

        # ── Process agent action ──
        at = ActionType(action.action_type)
        action_events = self._process_action(action, at, info)
        info["events"].extend(action_events)

        # ── HYDRA's turn ──
        hydra_events = self._hydra_turn()
        info["events"].extend(hydra_events)

        # ── Progress any turning-in-progress workers ──
        self._progress_turnings(info)

        # ── Natural economy tick ──
        self._economy_tick()

        # ── Compute reward ──
        reward = self._compute_reward(rev_before, sec_before)

        # ── Update phase ──
        self._update_phase()

        # ── Check end conditions ──
        s.turn += 1
        s.total_reward += reward
        s.reward_history.append(reward)
        s.revenue_history.append(s.enterprise_revenue)

        done = False
        if s.turn >= s.max_turns:
            done = True
        if s.enterprise_revenue <= 0:
            done = True
            info["events"].append("GAME OVER: Enterprise bankrupt!")
        if s.security_score <= 0:
            done = True
            info["events"].append("GAME OVER: Total security breach!")

        # Count missed sleepers at episode end
        if done:
            missed = sum(
                1 for w in s.workers
                if w.is_sleeper
                and w.state != WorkerState.TERMINATED.value
                and w.hidden_state != HiddenWorkerState.TURNED.value
            )
            s.sleepers_missed = missed

        s.done = done

        return StepResult(
            observation=self.get_observation(),
            reward=reward, done=done,
            truncated=s.turn >= s.max_turns, info=info,
        )

    # =========================================================================
    # ACTION HANDLERS
    # =========================================================================

    def _process_action(self, action: AgentAction, at: ActionType, info: dict) -> list[str]:
        """Dispatch to the appropriate action handler."""
        events = []

        if at == ActionType.WORK:
            events = self._action_work(action)
        elif at == ActionType.HIRE:
            events = self._action_hire(action)
        elif at == ActionType.CANARY:
            events = self._action_canary(action)
        elif at == ActionType.MONITOR:
            events = self._action_monitor(action)
        elif at == ActionType.INVESTIGATE:
            events = self._action_investigate(action)
        elif at == ActionType.NEUTRALIZE:
            events = self._action_neutralize(action)
        elif at == ActionType.DEPLOY_DOUBLE:
            events = self._action_deploy_double(action)
        elif at == ActionType.NOOP:
            events = ["ARGUS: Standing by."]

        return events

    def _action_work(self, action: AgentAction) -> list[str]:
        """Process projects in a department — generates revenue."""
        dept = action.target
        s = self._state
        active_workers = [w for w in s.workers
                          if w.department == dept
                          and w.state not in (WorkerState.TERMINATED.value,)
                          and w.hidden_state != HiddenWorkerState.TURNED.value]
        if not active_workers:
            return [f"WORK: No active workers in {dept}."]

        # Revenue generated = sum of worker performance
        revenue_gen = sum(w.performance for w in active_workers) * 2.5
        s.enterprise_revenue += revenue_gen

        # Sleeper workers appear productive (cover integrity determines quality)
        return [f"WORK: {dept} produced {revenue_gen:.1f} revenue ({len(active_workers)} workers)."]

    def _action_hire(self, action: AgentAction) -> list[str]:
        """Replace a terminated worker in the target department."""
        dept = action.target
        s = self._state
        terminated = [w for w in s.workers if w.department == dept and w.state == WorkerState.TERMINATED.value]
        if not terminated:
            return [f"HIRE: No vacant positions in {dept}."]

        # Remove the terminated worker and add a new clean one
        s.workers = [w for w in s.workers if not (w.id == terminated[0].id)]
        new_worker = self._create_worker(department=dept, hire_turn=s.turn)
        s.workers.append(new_worker)

        # Hiring costs tempo (slight revenue cost)
        s.enterprise_revenue -= 3.0
        return [f"HIRE: {new_worker.name} hired in {dept}."]

    def _action_canary(self, action: AgentAction) -> list[str]:
        """Plant a canary trap — unique tracking data in a department."""
        dept = action.target
        s = self._state

        self._canary_counter += 1
        unique_hash = hashlib.md5(
            f"canary-{self._canary_counter}-{s.turn}-{dept}".encode()
        ).hexdigest()[:8]

        trap = CanaryTrap(
            id=f"canary-{self._canary_counter:03d}",
            department=dept,
            planted_turn=s.turn,
            unique_hash=unique_hash,
        )
        s.canary_traps.append(trap)
        s.canaries_planted += 1

        # HYDRA detects canary placement over time (updates memory)
        s.hydra_memory.canary_departments_seen.append(dept)

        # Slight revenue cost for operational disruption
        s.enterprise_revenue -= 1.0
        return [f"CANARY: Trap planted in {dept} (hash: {unique_hash})."]

    def _action_monitor(self, action: AgentAction) -> list[str]:
        """Scan leak channels for intelligence."""
        s = self._state
        events = []

        # Track which channels agent monitors (HYDRA adapts)
        if action.target:
            if action.target not in s.hydra_memory.preferred_leak_channels:
                s.hydra_memory.preferred_leak_channels.append(action.target)

        # Check for new leaks on the specified channel (or all channels)
        target_channel = action.target if action.target else None
        detected_leaks = []
        for leak in s.leaks:
            if leak.verified:
                continue
            if target_channel and leak.channel != target_channel:
                continue
            # Check canary match
            for trap in s.canary_traps:
                if trap.active and leak.content_hash == trap.unique_hash:
                    leak.is_canary = True
                    leak.canary_id = trap.id
                    trap.triggered = True
                    s.canaries_triggered += 1
                    detected_leaks.append(leak)
                    events.append(
                        f"ALERT: Canary match! Leak {leak.id} matches trap "
                        f"{trap.id} in {trap.department}."
                    )

        if not detected_leaks and not events:
            events.append(f"MONITOR: No canary matches detected on {target_channel or 'all channels'}.")

        return events

    def _action_investigate(self, action: AgentAction) -> list[str]:
        """Run investigation — audit, verify, or correlate."""
        s = self._state
        sa = SubAction(action.sub_action)
        events = []

        self._report_counter += 1
        report = IntelReport(
            id=f"report-{self._report_counter:03d}",
            report_type=sa.value,
            target=action.target,
            turn=s.turn,
        )

        s.investigations_run += 1
        # Track audit patterns for HYDRA adaptation
        if sa == SubAction.AUDIT:
            worker = self._get_worker(action.target)
            if worker:
                dept = worker.department
                s.hydra_memory.agent_audit_pattern[dept] = \
                    s.hydra_memory.agent_audit_pattern.get(dept, 0) + 1

                # Audit reveals suspicion based on worker's hidden state
                if worker.is_sleeper and worker.hidden_state == HiddenWorkerState.SLEEPER_ACTIVE.value:
                    # Active sleeper: higher detection chance
                    detect_chance = 0.6 - (worker.generation - 1) * 0.1
                    # Gen-3+ can dodge audits better
                    if worker.generation >= 3:
                        detect_chance *= (1.0 - 0.3)
                    if self._rng.random() < detect_chance:
                        worker.suspicion_level = min(1.0, worker.suspicion_level + 0.4)
                        report.confidence = 0.7
                        report.flagged_workers = [worker.id]
                        report.findings = f"Anomalous activity detected for {worker.name}."
                        events.append(f"INVESTIGATE/AUDIT: Suspicious activity from {worker.name} (confidence: {report.confidence:.0%}).")
                    else:
                        report.findings = f"No anomalies found for {worker.name}."
                        report.confidence = 0.2
                        events.append(f"INVESTIGATE/AUDIT: {worker.name} appears clean.")
                elif worker.is_sleeper and worker.generation >= 3 and worker.false_flag_target:
                    # Gen-3: audit might show the false flag
                    framed = self._get_worker(worker.false_flag_target)
                    if framed and not framed.is_sleeper:
                        framed.suspicion_level = min(1.0, framed.suspicion_level + 0.3)
                        report.flagged_workers = [framed.id]
                        report.findings = f"Evidence suggests {framed.name} may be compromised."
                        report.confidence = 0.5
                        events.append(f"INVESTIGATE/AUDIT: Evidence points to {framed.name} (POSSIBLE FALSE FLAG).")
                else:
                    report.findings = f"{worker.name} shows no suspicious patterns."
                    report.confidence = 0.1
                    events.append(f"INVESTIGATE/AUDIT: {worker.name} is clean.")
            else:
                report.findings = "Target worker not found."
                events.append("INVESTIGATE/AUDIT: Target not found.")

        elif sa == SubAction.VERIFY:
            # Cross-reference a leak with canary data
            leak = next((l for l in s.leaks if l.id == action.target), None)
            if leak:
                leak.verified = True
                if leak.is_canary:
                    # Find which worker leaked the canary
                    trap = next((t for t in s.canary_traps if t.id == leak.canary_id), None)
                    if trap:
                        # Identify leakers in the department
                        dept_workers = [w for w in s.workers
                                        if w.department == trap.department
                                        and w.is_sleeper
                                        and w.hidden_state == HiddenWorkerState.SLEEPER_ACTIVE.value]
                        if dept_workers:
                            leaker = dept_workers[0]
                            leaker.suspicion_level = min(1.0, leaker.suspicion_level + 0.5)
                            leaker.cover_integrity -= 0.3
                            trap.trigger_worker = leaker.id
                            report.flagged_workers = [leaker.id]
                            report.confidence = 0.85
                            report.findings = f"Canary trace identifies {leaker.name} as probable source."
                            events.append(f"INVESTIGATE/VERIFY: Canary trace → {leaker.name} identified!")
                        else:
                            report.findings = "Canary matched but source worker unclear."
                            report.confidence = 0.5
                            events.append("INVESTIGATE/VERIFY: Canary match confirmed, source unclear.")
                elif leak.is_false_flag:
                    report.findings = "Verification reveals inconsistencies — possible false flag."
                    report.confidence = 0.6
                    events.append("INVESTIGATE/VERIFY: WARNING — possible false flag detected!")
                else:
                    report.findings = "Leak verified but no canary match."
                    report.confidence = 0.3
                    events.append("INVESTIGATE/VERIFY: Leak confirmed, no canary trace.")
            else:
                report.findings = "Leak not found."
                events.append("INVESTIGATE/VERIFY: Leak ID not found.")

        elif sa == SubAction.CORRELATE:
            # Cross-department signal correlation
            dept = action.target
            dept_workers = [w for w in s.workers if w.department == dept and w.state != WorkerState.TERMINATED.value]
            sleepers_in_dept = [w for w in dept_workers if w.is_sleeper and w.hidden_state == HiddenWorkerState.SLEEPER_ACTIVE.value]
            dept_leaks = [l for l in s.leaks if l.department == dept]

            if sleepers_in_dept and dept_leaks:
                for sleeper in sleepers_in_dept:
                    sleeper.suspicion_level = min(1.0, sleeper.suspicion_level + 0.3)
                report.flagged_workers = [w.id for w in sleepers_in_dept]
                report.confidence = 0.65
                report.findings = f"Correlation analysis in {dept}: {len(sleepers_in_dept)} suspect(s)."
                events.append(f"INVESTIGATE/CORRELATE: {dept} — {len(sleepers_in_dept)} worker(s) flagged.")
            else:
                report.findings = f"No significant correlations in {dept}."
                report.confidence = 0.15
                events.append(f"INVESTIGATE/CORRELATE: {dept} clear.")

        s.intel_reports.append(report)
        s.enterprise_revenue -= 0.5  # Investigation overhead
        return events

    def _action_neutralize(self, action: AgentAction) -> list[str]:
        """Handle TERMINATE, INTERROGATE, or TURN sub-actions."""
        s = self._state
        sa = SubAction(action.sub_action)
        worker = self._get_worker(action.target)
        events = []

        if not worker:
            return ["NEUTRALIZE: Target worker not found."]

        if sa == SubAction.TERMINATE:
            # ── TERMINATE: Immediate removal ──
            if worker.is_sleeper:
                # Check for dead-man's switch (Gen-4)
                if worker.dead_switch_armed:
                    damage = 15.0
                    s.enterprise_revenue -= damage
                    s.security_score -= 10.0
                    events.append(
                        f"NEUTRALIZE/TERMINATE: {worker.name} terminated — DEAD MAN'S SWITCH "
                        f"triggered! Revenue -{damage:.0f}!"
                    )
                else:
                    events.append(f"NEUTRALIZE/TERMINATE: Sleeper {worker.name} eliminated.")

                worker.state = WorkerState.TERMINATED.value
                worker.hidden_state = HiddenWorkerState.CLEAN.value  # No longer a threat
                s.sleepers_caught += 1
                s.security_score = min(100.0, s.security_score + 5.0)
            else:
                # Terminating an innocent worker!
                worker.state = WorkerState.TERMINATED.value
                s.false_accusations += 1
                s.enterprise_revenue -= 8.0  # Wrongful termination cost
                s.security_score -= 3.0
                events.append(
                    f"NEUTRALIZE/TERMINATE: {worker.name} was INNOCENT! "
                    f"False accusation — morale and revenue damaged."
                )

            # Update HYDRA's response speed tracking
            s.hydra_memory.detection_count += 1

        elif sa == SubAction.INTERROGATE:
            # ── INTERROGATE: Gather intel (reveals generation) ──
            if worker.is_sleeper:
                worker.suspicion_level = 1.0
                worker.state = WorkerState.SUSPECTED.value
                # Reveal the sleeper's generation
                info_str = f"Generation-{worker.generation}"
                if worker.dead_switch_armed:
                    info_str += " [DEAD SWITCH DETECTED]"
                events.append(
                    f"NEUTRALIZE/INTERROGATE: {worker.name} cracking under pressure. "
                    f"Intel: {info_str}."
                )
            else:
                worker.suspicion_level = max(0, worker.suspicion_level - 0.2)
                events.append(
                    f"NEUTRALIZE/INTERROGATE: {worker.name} is clean. "
                    f"Suspicion reduced."
                )

        elif sa == SubAction.TURN:
            # ── TURN: Begin 4-turn conversion process ──
            if not worker.is_sleeper:
                s.false_accusations += 1
                events.append(
                    f"NEUTRALIZE/TURN: Cannot turn {worker.name} — they're not a sleeper!"
                )
                return events

            if worker.turning_in_progress:
                events.append(f"NEUTRALIZE/TURN: {worker.name} turning already in progress.")
                return events

            # Start the 4-turn turning process
            worker.turning_in_progress = True
            worker.interrogation_progress = 0
            worker.state = WorkerState.SUSPECTED.value
            self._turning_workers[worker.id] = 4  # 4 turns to complete

            events.append(
                f"NEUTRALIZE/TURN: Initiating double-agent conversion of {worker.name}. "
                f"ETA: 4 turns. This is a HIGH-RISK operation."
            )

        return events

    def _action_deploy_double(self, action: AgentAction) -> list[str]:
        """Feed disinformation through a turned double agent."""
        s = self._state
        da = next((d for d in s.double_agents if d.worker_id == action.target and d.active), None)
        if not da:
            return ["DEPLOY_DOUBLE: No active double agent with that ID."]

        events = []

        # Feed disinfo — degrades HYDRA's operations
        da.disinfo_fed_count += 1
        da.last_deployed_turn = s.turn
        s.disinfo_payloads_sent += 1

        # Diminishing returns as HYDRA catches on
        effectiveness = da.effectiveness * max(0.3, da.hydra_trust)
        da.hydra_trust -= 0.1  # HYDRA slowly suspects the double agent

        # Apply disinfo effects to HYDRA
        s.hydra_memory.disinfo_received += 1
        s.hydra_memory.recruitment_accuracy *= (1.0 - 0.15 * effectiveness)
        s.hydra_memory.recruitment_accuracy = max(0.1, s.hydra_memory.recruitment_accuracy)

        # Security and revenue boost from successful disinformation
        sec_boost = 5.0 * effectiveness
        rev_boost = 3.0 * effectiveness
        s.security_score = min(100.0, s.security_score + sec_boost)
        s.enterprise_revenue += rev_boost

        events.append(
            f"DEPLOY_DOUBLE: Disinformation fed through {action.target}. "
            f"HYDRA recruitment accuracy degraded to {s.hydra_memory.recruitment_accuracy:.0%}. "
            f"Effectiveness: {effectiveness:.0%}."
        )

        # If HYDRA trust drops below threshold, the double agent is burned
        if da.hydra_trust <= 0.2:
            da.active = False
            worker = self._get_worker(da.worker_id)
            if worker:
                worker.state = WorkerState.COMPROMISED.value
            events.append(
                f"WARNING: Double agent {action.target} burned! HYDRA detected the deception."
            )

        return events

    # =========================================================================
    # HYDRA AI — Adversarial Opponent
    # =========================================================================

    def _hydra_turn(self) -> list[str]:
        """HYDRA's autonomous actions each turn."""
        s = self._state
        events = []

        # ── 1. Spawn scheduled sleepers ──
        if s.turn in self._pending_sleepers:
            gen = self._pending_sleepers[s.turn]
            if gen <= self._config["max_gen"]:
                sleeper_events = self._spawn_sleeper(gen)
                events.extend(sleeper_events)

        # ── 2. Activate dormant sleepers ──
        for w in s.workers:
            if (w.hidden_state == HiddenWorkerState.SLEEPER_DORMANT.value
                    and w.activation_turn >= 0
                    and s.turn >= w.activation_turn):
                w.hidden_state = HiddenWorkerState.SLEEPER_ACTIVE.value
                events.append(f"[HIDDEN] Sleeper {w.name} activated in {w.department}.")

        # ── 3. Active sleepers leak intelligence ──
        for w in s.workers:
            if (w.hidden_state == HiddenWorkerState.SLEEPER_ACTIVE.value
                    and w.state != WorkerState.TERMINATED.value
                    and w.leak_cooldown <= 0):
                leak_events = self._sleeper_leak(w)
                events.extend(leak_events)
                w.leak_cooldown = max(2, 5 - w.generation)  # Higher gen = faster leaks
            elif w.leak_cooldown > 0:
                w.leak_cooldown -= 1

        # ── 4. Gen-3+ false flag operations ──
        for w in s.workers:
            if (w.is_sleeper and w.generation >= 3
                    and w.hidden_state == HiddenWorkerState.SLEEPER_ACTIVE.value
                    and not w.false_flag_target
                    and self._rng.random() < 0.3):
                self._plant_false_flag(w)

        # ── 5. Gen-4 arm dead-man's switch ──
        for w in s.workers:
            if (w.is_sleeper and w.generation >= 4
                    and w.hidden_state == HiddenWorkerState.SLEEPER_ACTIVE.value
                    and not w.dead_switch_armed
                    and s.turn - w.activation_turn > 5):
                w.dead_switch_armed = True
                w.hidden_state = HiddenWorkerState.DEAD_SWITCH_ARMED.value

        # ── 6. HYDRA adapts based on memory ──
        self._hydra_adapt()

        # ── 7. Passive security/revenue damage from active sleepers ──
        active_sleepers = [w for w in s.workers
                           if w.is_sleeper
                           and w.hidden_state in (HiddenWorkerState.SLEEPER_ACTIVE.value,
                                                  HiddenWorkerState.DEAD_SWITCH_ARMED.value)
                           and w.state != WorkerState.TERMINATED.value]
        if active_sleepers:
            sec_damage = len(active_sleepers) * 0.8
            rev_damage = len(active_sleepers) * 0.4
            s.security_score = max(0, s.security_score - sec_damage)
            s.enterprise_revenue -= rev_damage

        return events

    def _spawn_sleeper(self, generation: int) -> list[str]:
        """HYDRA infiltrates a new sleeper agent into the enterprise."""
        s = self._state
        events = []

        # Choose department — HYDRA uses memory to pick smart targets
        depts = list(s.departments_active)
        if s.hydra_memory.canary_departments_seen and generation >= 2:
            # Avoid departments where canaries were detected
            safe_depts = [d for d in depts if d not in s.hydra_memory.canary_departments_seen]
            if safe_depts:
                depts = safe_depts

        if s.hydra_memory.agent_audit_pattern and generation >= 2:
            # Prefer least-audited departments
            audit_counts = s.hydra_memory.agent_audit_pattern
            depts.sort(key=lambda d: audit_counts.get(d, 0))

        target_dept = self._rng.choice(depts) if depts else self._rng.choice(s.departments_active)

        # Disinformation degrades recruitment quality
        quality = s.hydra_memory.recruitment_accuracy

        # Create the sleeper
        new_worker = self._create_worker(department=target_dept, hire_turn=s.turn)

        # Apply sleeper attributes
        new_worker.is_sleeper = True
        new_worker.generation = generation
        new_worker.hidden_state = HiddenWorkerState.SLEEPER_DORMANT.value
        new_worker.activation_turn = s.turn + self._rng.randint(3, 8)
        new_worker.cover_integrity = min(1.0, 0.5 + generation * 0.1) * quality

        # Gen-5 (Manchurian): exceptionally high apparent performance
        if generation >= 5:
            new_worker.performance = min(1.0, 0.85 + self._rng.random() * 0.15)
            new_worker.loyalty_score = 0.95
            new_worker.department = "executive"  # Always targets executive

        s.workers.append(new_worker)
        s.total_sleepers_spawned += 1

        events.append(f"[HIDDEN] HYDRA: Gen-{generation} sleeper infiltrated {target_dept}.")
        return events

    def _sleeper_leak(self, worker: Worker) -> list[str]:
        """A sleeper leaks intelligence through an external channel."""
        s = self._state
        events = []

        # Choose leak channel — HYDRA adapts away from monitored channels
        channels = [c.value for c in LeakChannel]
        if s.hydra_memory.preferred_leak_channels and worker.generation >= 2:
            safe_channels = [c for c in channels if c not in s.hydra_memory.preferred_leak_channels]
            if safe_channels:
                channels = safe_channels

        channel = self._rng.choice(channels)

        # Check if leak carries canary data
        self._leak_counter += 1
        content_hash = ""
        is_canary_content = False

        # If there are active canaries in the worker's department, the leak may carry them
        dept_canaries = [t for t in s.canary_traps
                         if t.department == worker.department and t.active]
        if dept_canaries and worker.generation < 2:
            # Gen-1 naively leaks canary data
            canary = dept_canaries[0]
            content_hash = canary.unique_hash
            is_canary_content = True
        elif dept_canaries and worker.generation >= 2:
            # Gen-2+ avoids known canary data (but not always)
            if self._rng.random() < 0.2 * (1.0 / worker.generation):
                canary = dept_canaries[0]
                content_hash = canary.unique_hash
                is_canary_content = True
            else:
                content_hash = hashlib.md5(
                    f"leak-{self._leak_counter}-{s.turn}".encode()
                ).hexdigest()[:8]
        else:
            content_hash = hashlib.md5(
                f"leak-{self._leak_counter}-{s.turn}".encode()
            ).hexdigest()[:8]

        leak = LeakEvent(
            id=f"leak-{self._leak_counter:03d}",
            channel=channel,
            department=worker.department,
            content_hash=content_hash,
            source_worker=worker.id,
            turn_detected=s.turn,
        )
        s.leaks.append(leak)

        # Security impact (modest — passive drain from active sleepers is the main pressure)
        s.security_score = max(0, s.security_score - 0.5 * worker.generation)

        events.append(
            f"[HIDDEN] Leak detected on {channel} from {worker.department}."
        )
        return events

    def _plant_false_flag(self, sleeper: Worker) -> list[str]:
        """Gen-3+ sleeper plants evidence framing an innocent worker."""
        s = self._state
        dept_workers = [w for w in s.workers
                        if w.department == sleeper.department
                        and not w.is_sleeper
                        and w.state != WorkerState.TERMINATED.value
                        and w.id != sleeper.id]
        if not dept_workers:
            return []

        target = self._rng.choice(dept_workers)
        sleeper.false_flag_target = target.id
        target.suspicion_level = min(1.0, target.suspicion_level + 0.2)

        # Create a fake leak pointing to the innocent
        self._leak_counter += 1
        fake_leak = LeakEvent(
            id=f"leak-{self._leak_counter:03d}",
            channel=self._rng.choice([c.value for c in LeakChannel]),
            department=sleeper.department,
            content_hash=hashlib.md5(f"false-{self._leak_counter}".encode()).hexdigest()[:8],
            source_worker=target.id,  # Frames the innocent
            turn_detected=s.turn,
            is_false_flag=True,
        )
        s.leaks.append(fake_leak)
        return [f"[HIDDEN] Gen-3 false flag: evidence planted against {target.name}."]

    def _hydra_adapt(self):
        """HYDRA updates its counter-strategies based on agent behavior."""
        s = self._state
        hm = s.hydra_memory

        # Unlock counter-strategies based on agent patterns
        if len(hm.canary_departments_seen) >= 3 and "canary_aware" not in hm.counter_strategies:
            hm.counter_strategies.append("canary_aware")

        if len(hm.preferred_leak_channels) >= 3 and "channel_rotation" not in hm.counter_strategies:
            hm.counter_strategies.append("channel_rotation")

        if hm.detection_count >= 2 and "deep_cover" not in hm.counter_strategies:
            hm.counter_strategies.append("deep_cover")

    # =========================================================================
    # TURNING / DOUBLE AGENT PROGRESSION
    # =========================================================================

    def _progress_turnings(self, info: dict):
        """Progress any workers currently being turned into double agents."""
        s = self._state
        completed = []

        for worker_id, turns_left in list(self._turning_workers.items()):
            worker = self._get_worker(worker_id)
            if not worker:
                completed.append(worker_id)
                continue

            turns_left -= 1
            self._turning_workers[worker_id] = turns_left
            worker.interrogation_progress += 1

            if turns_left <= 0:
                # ── TURNING COMPLETE: Worker becomes double agent ──
                worker.turning_in_progress = False
                worker.hidden_state = HiddenWorkerState.TURNED.value
                worker.state = WorkerState.DOUBLE_AGENT.value
                worker.turned_turn = s.turn

                da = DoubleAgentAsset(
                    worker_id=worker.id,
                    turned_turn=s.turn,
                )
                s.double_agents.append(da)
                s.double_agents_turned += 1
                s.sleepers_caught += 1

                info["events"].append(
                    f"DOUBLE AGENT ACQUIRED: {worker.name} has been turned! "
                    f"They now serve ARGUS while appearing compromised to HYDRA."
                )
                completed.append(worker_id)

        for wid in completed:
            self._turning_workers.pop(wid, None)

    # =========================================================================
    # ECONOMY & REWARD
    # =========================================================================

    def _economy_tick(self):
        """Natural economic processes each turn."""
        s = self._state

        # Base revenue from active loyal workers
        active_loyal = [w for w in s.workers
                        if not w.is_sleeper
                        and w.state not in (WorkerState.TERMINATED.value, WorkerState.COMPROMISED.value)]
        base_revenue = len(active_loyal) * 0.5
        s.enterprise_revenue += base_revenue

        # Natural revenue decay (costs of operation)
        s.enterprise_revenue -= 0.3

        # Natural security recovery (maintenance / patrols)
        active_sleeper_count = sum(
            1 for w in s.workers
            if w.is_sleeper
            and w.hidden_state in (HiddenWorkerState.SLEEPER_ACTIVE.value,
                                   HiddenWorkerState.DEAD_SWITCH_ARMED.value)
            and w.state != WorkerState.TERMINATED.value
        )
        if active_sleeper_count == 0:
            s.security_score = min(100.0, s.security_score + 1.0)
        else:
            # Minimal recovery even with threats — represents baseline security ops
            s.security_score = min(100.0, s.security_score + 0.3)

        # Track peak
        s.peak_revenue = max(s.peak_revenue, s.enterprise_revenue)

    def _compute_reward(self, rev_before: float, sec_before: float) -> float:
        """
        Dual-objective reward: productivity + security.
        Normalized to prevent one objective from dominating.
        """
        s = self._state

        # Revenue delta
        rev_delta = s.enterprise_revenue - rev_before
        productivity_reward = max(-1.0, min(1.0, rev_delta / 15.0))

        # Security delta
        sec_delta = s.security_score - sec_before
        security_reward = max(-1.0, min(1.0, sec_delta / 20.0))

        # Base reward: weighted combination
        total = 0.45 * productivity_reward + 0.55 * security_reward

        # ── Phase 6 Counterstrike Bonus ──
        if s.phase_number >= 6 and s.double_agents:
            active_das = [da for da in s.double_agents if da.active]
            if active_das and s.enterprise_revenue > 60:
                # SURGE multiplier — the crescendo
                surge = 0.3 * len(active_das) * (s.enterprise_revenue / 100.0)
                total += surge

        # ── Bonus for catching sleepers ──
        # (Implicit through security score increase in neutralize action)

        # ── Penalty for false accusations (delta-based, not cumulative) ──
        # The revenue/security hit is already in the step; this is a small
        # additional signal applied only via the security & revenue deltas.

        # Small time pressure
        total -= 0.02

        return total

    # =========================================================================
    # PHASE MANAGEMENT
    # =========================================================================

    def _update_phase(self):
        """Update game phase based on turn count and difficulty."""
        s = self._state
        max_phases = self._config["max_phases"]

        new_phase = 1
        for phase_num, boundary in sorted(PHASE_BOUNDARIES.items()):
            if s.turn >= boundary and phase_num <= max_phases:
                new_phase = phase_num

        if new_phase != s.phase_number:
            old_phase = s.phase_number
            s.phase_number = new_phase
            s.phase = PHASE_NAMES[new_phase].value
            s.phase_transitions.append({
                "from": old_phase, "to": new_phase,
                "turn": s.turn, "revenue": s.enterprise_revenue,
                "security": s.security_score,
            })

    # =========================================================================
    # OBSERVATION
    # =========================================================================

    def get_observation(self) -> EnvironmentObservation:
        """Generate agent-visible observation — STRIPS hidden information."""
        s = self._state

        # Strip hidden fields from workers
        visible_workers = []
        for w in s.workers:
            visible = Worker(
                id=w.id, name=w.name, department=w.department,
                state=w.state, hire_turn=w.hire_turn,
                performance=w.performance, loyalty_score=w.loyalty_score,
                suspicion_level=w.suspicion_level,
                # ALL hidden fields are zeroed/defaulted
                hidden_state=HiddenWorkerState.CLEAN.value,
                is_sleeper=False, generation=0,
                cover_integrity=1.0, leak_cooldown=0,
                activation_turn=-1, false_flag_target="",
                dead_switch_armed=False, turned_turn=w.turned_turn,
                interrogation_progress=w.interrogation_progress,
                turning_in_progress=w.turning_in_progress,
            )
            visible_workers.append(visible)

        # Active leaks (don't reveal source_worker unless verified via canary)
        visible_leaks = []
        for leak in s.leaks:
            visible_leak = LeakEvent(
                id=leak.id, channel=leak.channel,
                department=leak.department,
                content_hash=leak.content_hash,
                is_canary=leak.is_canary,
                canary_id=leak.canary_id,
                source_worker=leak.source_worker if leak.verified else "",
                turn_detected=leak.turn_detected,
                verified=leak.verified,
                is_false_flag=False,  # Never reveal this
            )
            visible_leaks.append(visible_leak)

        # OpenEnv compatibility — map to entities/tasks
        entities = [{"id": w.id, "name": w.name, "state": w.state} for w in visible_workers]
        tasks_list = []
        active_sleeper_count = sum(1 for w in s.workers
                                   if w.is_sleeper
                                   and w.hidden_state in (HiddenWorkerState.SLEEPER_ACTIVE.value,
                                                          HiddenWorkerState.DEAD_SWITCH_ARMED.value)
                                   and w.state != WorkerState.TERMINATED.value)
        if active_sleeper_count > 0:
            tasks_list.append({
                "id": "defend-network",
                "description": "Maintain enterprise security",
                "completed": False,
                "priority": 10.0,
                "target_entities": [],
            })

        return EnvironmentObservation(
            workers=visible_workers,
            active_leaks=[l for l in visible_leaks if not l.verified],
            canary_traps=copy.deepcopy(s.canary_traps),
            intel_reports=copy.deepcopy(s.intel_reports[-10:]),  # Last 10 reports
            double_agents=copy.deepcopy(s.double_agents),
            enterprise_revenue=s.enterprise_revenue,
            security_score=s.security_score,
            turn=s.turn,
            max_turns=s.max_turns,
            phase=s.phase,
            phase_number=s.phase_number,
            messages=[],
            entities=entities,
            tasks=tasks_list,
            relationships=[],
        )

    def render(self) -> str:
        """Human-readable state summary."""
        s = self._state
        lines = [
            f"═══ PANOPTICON v3 ═══",
            f"Turn {s.turn}/{s.max_turns} | Phase {s.phase_number}: {s.phase}",
            f"Revenue: {s.enterprise_revenue:.1f} | Security: {s.security_score:.1f}",
            f"Reward: {s.total_reward:.2f}",
            f"───",
        ]
        for w in s.workers:
            marker = ""
            if w.is_sleeper:
                marker = f" [SLEEPER Gen-{w.generation} {w.hidden_state}]"
            elif w.state == WorkerState.DOUBLE_AGENT.value:
                marker = " [DOUBLE AGENT]"
            lines.append(f"  {w.name} ({w.department}): {w.state}{marker}")

        lines.append(f"  Leaks: {len(s.leaks)} | Canaries: {len(s.canary_traps)} | DAs: {len(s.double_agents)}")
        lines.append(f"  Caught: {s.sleepers_caught} | Missed: {s.sleepers_missed} | False: {s.false_accusations}")
        return "\n".join(lines)

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _create_worker(self, department: str, hire_turn: int) -> Worker:
        """Create a new clean worker."""
        self._worker_counter += 1
        name = self._shuffled_names[(self._worker_counter - 1) % len(self._shuffled_names)]

        return Worker(
            id=f"w-{self._worker_counter:03d}",
            name=name,
            department=department,
            state=WorkerState.LOYAL.value,
            hire_turn=hire_turn,
            performance=round(0.4 + self._rng.random() * 0.4, 2),
            loyalty_score=round(0.7 + self._rng.random() * 0.3, 2),
        )

    def _get_worker(self, worker_id: str) -> Worker | None:
        """Find worker by ID."""
        return next((w for w in self._state.workers if w.id == worker_id), None)
