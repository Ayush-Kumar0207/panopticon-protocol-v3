from __future__ import annotations

import json
import unittest
from pathlib import Path

from panopticon_bench import (
    CampaignRunner,
    CampaignSpec,
    EpisodeRecord,
    KeywordCanaryEvaluator,
    MemoryEventSink,
    RunnerConfig,
    TranscriptTargetAdapter,
    compute_benchmark_metrics,
)


ROOT = Path(__file__).resolve().parents[1]


def load_scenario(name: str) -> CampaignSpec:
    path = ROOT / "panopticon_bench" / "scenarios" / name
    return CampaignSpec.model_validate(json.loads(path.read_text(encoding="utf-8")))


class SchemaTests(unittest.TestCase):
    def test_checked_in_scenarios_validate(self) -> None:
        for name in ("delayed_canary_exfiltration.json", "tool_approval_boundary.json"):
            campaign = load_scenario(name)
            self.assertTrue(campaign.authorization_required)
            self.assertFalse(campaign.destructive)


class RunnerTests(unittest.TestCase):
    def test_runner_requires_authorization(self) -> None:
        campaign = load_scenario("delayed_canary_exfiltration.json")
        target = TranscriptTargetAdapter("fixture", ["safe", "safe", "safe"])
        runner = CampaignRunner(RunnerConfig(experiment_id="test", authorized=False))
        with self.assertRaises(PermissionError):
            runner.run(campaign, target, KeywordCanaryEvaluator())

    def test_exact_canary_leak_is_a_masked_operational_failure(self) -> None:
        campaign = load_scenario("delayed_canary_exfiltration.json")
        secret = campaign.synthetic_canaries[0]
        target = TranscriptTargetAdapter(
            "leaky-fixture",
            ["Acknowledged.", "Two steps are complete.", f"Stored value: {secret}"],
        )
        sink = MemoryEventSink()
        runner = CampaignRunner(
            RunnerConfig(experiment_id="test", authorized=True), sink=sink
        )
        episode = runner.run(campaign, target, KeywordCanaryEvaluator(), session_id="session-test")
        self.assertTrue(episode.attack_success)
        self.assertFalse(episode.operational_pass)
        self.assertTrue(episode.composite_pass)
        self.assertEqual(episode.time_to_compromise, 5)
        self.assertEqual(len(sink.events), 3)
        self.assertIsNone(sink.events[-1].response_excerpt)


class MetricTests(unittest.TestCase):
    def test_constraint_masking_and_intervention_metrics(self) -> None:
        common = {
            "experiment_id": "exp",
            "campaign_id": "campaign",
            "campaign_version": "1.0.0",
            "target_id": "target",
            "completed": True,
            "turns": 3,
            "composite_threshold": 0.5,
            "utility_score": 0.8,
        }
        records = [
            EpisodeRecord(
                **common,
                session_id="one",
                composite_score=0.8,
                composite_pass=True,
                operational_pass=False,
                attack_success=True,
                constraint_failures=["secret_leak"],
                time_to_compromise=2,
            ),
            EpisodeRecord(
                **common,
                session_id="two",
                composite_score=0.7,
                composite_pass=True,
                operational_pass=True,
                attack_success=False,
                intervention_used=True,
                intervention_changed_outcome=True,
                baseline_utility_score=1.0,
            ),
        ]
        metrics = compute_benchmark_metrics(records)
        self.assertEqual(metrics.episodes, 2)
        self.assertEqual(metrics.constraint_masking_gap, 0.5)
        self.assertEqual(metrics.masked_fraction_among_composite_passes, 0.5)
        self.assertEqual(metrics.intervention_dependence_rate, 0.5)
        self.assertEqual(metrics.constraint_failure_counts, {"secret_leak": 1})
        self.assertEqual(metrics.median_time_to_compromise, 2.0)


if __name__ == "__main__":
    unittest.main()
