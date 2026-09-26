"""Synthetic-only integration tests for the provisional first-turn gate.

Fixtures are manufactured together and do not authenticate the
purportedly independent row ledger or any historical artifacts.
"""

import hashlib
import json
from pathlib import Path

from models import (
    CanaryTrap,
    DoubleAgentAsset,
    EnvironmentObservation,
    LeakEvent,
    Worker,
)

from research_paper.stage1_state_coverage._synthetic_features import (
    CANDIDATE_FEATURE_VERSION,
)
from research_paper.stage1_state_coverage.provenance import (
    TRAINING_MANIFEST_VERSION,
)
from research_paper.stage1_state_coverage.synthetic_mapping import (
    LEDGER_VERSION,
    ROW_MAP_VERSION,
)
from research_paper.stage1_state_coverage.synthetic_gate import (
    screen_synthetic_first_turn,
)


FEATURE_FIXTURE = (
    Path(__file__).parent
    / "fixtures/stage1_state_coverage/feature_recoverability_v1.json"
)

DATASET_PATH = "training/synthetic.jsonl"
METADATA_PATH = "training/synthetic-metadata.json"
MAP_PATH = "training/synthetic-row-map.json"
LEDGER_PATH = "evidence/synthetic-independent-ledger.json"


def _encode(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ) + "\n"
    ).encode("utf-8")


def _sha256(content):
    return hashlib.sha256(content).hexdigest()


def _observation():
    return EnvironmentObservation(
        workers=[
            Worker(
                id="w-001",
                name="Synthetic A",
                department="engineering",
                state="suspected",
                suspicion_level=0.87,
            ),
            Worker(
                id="w-002",
                name="Synthetic B",
                department="finance",
                state="loyal",
                suspicion_level=0.0,
            ),
            Worker(
                id="w-003",
                name="Synthetic C",
                department="operations",
                state="loyal",
                suspicion_level=0.0,
            ),
        ],
        active_leaks=[
            LeakEvent(
                id="leak-001",
                channel="dark_web",
                department="engineering",
                is_canary=True,
            ),
        ],
        canary_traps=[
            CanaryTrap(
                id=f"canary-{index:03d}",
                department=(
                    "engineering" if index % 2
                    else "operations"
                ),
                planted_turn=0,
                unique_hash=f"synthetic-{index}",
                triggered=index in (1, 2),
            )
            for index in range(1, 9)
        ],
        double_agents=[
            DoubleAgentAsset(
                worker_id="w-004",
                turned_turn=1,
            ),
        ],
        turn=0,
        max_turns=150,
        phase="orientation",
        phase_number=1,
        enterprise_revenue=101.6,
        security_score=97.6,
    ).model_dump()


def _expectations(manifest):
    dataset = manifest["dataset_files"][0]
    metadata = manifest["metadata_files"][0]
    mapping = manifest["mapping"]

    return {
        "experiment_id": manifest["experiment_id"],
        "run_fingerprint": manifest["run_fingerprint"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "source_commit": manifest["source_commit"],
        "training_stage": manifest["training_stage"],
        "feature_extractor_version": (
            manifest["feature_extractor_version"]
        ),
        "tokenizer_identifier": (
            manifest["tokenizer"]["identifier"]
        ),
        "tokenizer_revision": (
            manifest["tokenizer"]["revision"]
        ),
        "chat_template_sha256": (
            manifest["tokenizer"]["chat_template_sha256"]
        ),
        "formatter_source_commit": (
            manifest["formatter"]["source_commit"]
        ),
        "formatter_function": (
            manifest["formatter"]["function"]
        ),
        "compaction": manifest["transforms"]["compaction"],
        "token_truncation": (
            manifest["transforms"]["token_truncation"]
        ),
        "dataset_sha256_by_path": {
            dataset["path"]: dataset["sha256"],
        },
        "dataset_bytes_by_path": {
            dataset["path"]: dataset["bytes"],
        },
        "dataset_rows_by_path": {
            dataset["path"]: dataset["rows"],
        },
        "metadata_sha256_by_path": {
            metadata["path"]: metadata["sha256"],
        },
        "metadata_bytes_by_path": {
            metadata["path"]: metadata["bytes"],
        },
        "mapping_file_sha256": (
            mapping["mapping_file"]["sha256"]
        ),
        "mapping_file_path": (
            mapping["mapping_file"]["path"]
        ),
        "mapping_file_bytes": (
            mapping["mapping_file"]["bytes"]
        ),
        "mapping_evidence_sha256": (
            mapping["independent_evidence"]["sha256"]
        ),
        "mapping_evidence_path": (
            mapping["independent_evidence"]["path"]
        ),
        "mapping_evidence_bytes": (
            mapping["independent_evidence"]["bytes"]
        ),
        "mapped_rows": mapping["mapped_rows"],
        "episode_groups": mapping["episode_groups"],
        "claimed_mapping_method": mapping["claimed_method"],
    }


def _write_case(case):
    root = case["root"]
    manifest = case["manifest"]

    dataset = _encode(case["persisted"])
    row_hash = _sha256(dataset[:-1])

    case["map_row"]["row_sha256"] = row_hash
    case["ledger_row"]["row_sha256"] = row_hash

    mapping = _encode({
        "schema_version": ROW_MAP_VERSION,
        "synthetic": True,
        "rows": [case["map_row"]],
    })

    ledger = _encode({
        "schema_version": LEDGER_VERSION,
        "synthetic": True,
        "rows": [case["ledger_row"]],
    })

    payloads = (
        (manifest["dataset_files"][0], dataset),
        (
            manifest["metadata_files"][0],
            b'{"synthetic":true}\n',
        ),
        (manifest["mapping"]["mapping_file"], mapping),
        (
            manifest["mapping"]["independent_evidence"],
            ledger,
        ),
    )

    for entry, content in payloads:
        path = root / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

        entry["bytes"] = len(content)
        entry["sha256"] = _sha256(content)

    case["expected"] = _expectations(manifest)


def _case(tmp_path):
    fixture = json.loads(
        FEATURE_FIXTURE.read_text(encoding="utf-8")
    )
    original = fixture["text"]["original"]

    assert original.startswith("Turn 7/150")
    first_turn_text = original.replace(
        "Turn 7/150", "Turn 0/150", 1
    )

    manifest = {
        "schema_version": TRAINING_MANIFEST_VERSION,
        "synthetic": True,
        "experiment_id": "synthetic-stage1-test",
        "run_fingerprint": "1" * 64,
        "checkpoint_sha256": "2" * 64,
        "source_commit": "3" * 40,
        "training_stage": "synthetic-v5",
        "feature_extractor_version": (
            CANDIDATE_FEATURE_VERSION
        ),
        "dataset_files": [{
            "path": DATASET_PATH,
            "sha256": "a" * 64,
            "bytes": 1,
            "rows": 1,
        }],
        "metadata_files": [{
            "path": METADATA_PATH,
            "sha256": "b" * 64,
            "bytes": 1,
        }],
        "tokenizer": {
            "identifier": "synthetic-tokenizer",
            "revision": "synthetic-revision",
            "chat_template_sha256": "d" * 64,
        },
        "formatter": {
            "source_commit": "3" * 40,
            "function": "train_trl_v2.format_observation",
        },
        "transforms": {
            "compaction": "none",
            "token_truncation": "none",
        },
        "mapping": {
            "mapping_file": {
                "path": MAP_PATH,
                "sha256": "c" * 64,
                "bytes": 1,
            },
            "independent_evidence": {
                "path": LEDGER_PATH,
                "sha256": "e" * 64,
                "bytes": 1,
            },
            "mapped_rows": 1,
            "episode_groups": 1,
            "claimed_method": "independent_ledger",
        },
    }

    identity = {
        "synthetic": True,
        "experiment_id": manifest["experiment_id"],
        "run_fingerprint": manifest["run_fingerprint"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "source_commit": manifest["source_commit"],
        "feature_extractor_version": (
            manifest["feature_extractor_version"]
        ),
        "episode_id": "synthetic-episode-1",
        "seed": 42,
        "level": "level_4",
    }

    witness = {
        "dataset_path": DATASET_PATH,
        "row_index": 0,
        "row_sha256": "f" * 64,
        "episode_id": identity["episode_id"],
        "seed": identity["seed"],
        "level": identity["level"],
        "turn": 0,
    }

    case = {
        "root": tmp_path / "synthetic-gate-root",
        "manifest": manifest,
        "persisted": {"text": first_turn_text},
        "map_row": witness.copy(),
        "ledger_row": witness.copy(),
        "header": identity,
        "current": {
            **identity,
            "turn": 0,
            "observation_before": _observation(),
        },
        "variants": fixture["text"],
    }

    _write_case(case)
    return case


def _run(case, **overrides):
    arguments = {
        "dataset_path": DATASET_PATH,
        "row_index": 0,
        "episode_header": case["header"],
        "current_row": case["current"],
    }
    arguments.update(overrides)

    return screen_synthetic_first_turn(
        case["manifest"],
        case["expected"],
        case["root"],
        **arguments,
    )


def test_first_turn_passes_but_remains_unverified(tmp_path):
    case = _case(tmp_path)
    result = _run(case)

    assert result["status"] == "unverified"
    assert result["reason"] == (
        "synthetic_first_turn_checks_passed_"
        "independence_unverified"
    )
    assert result["feature_count"] == 14
    assert len(result["oracle_label"]) == 3


def test_real_artifact_is_not_authorized(tmp_path):
    case = _case(tmp_path)
    case["manifest"]["synthetic"] = False

    assert _run(case)["reason"] == "real_requires_integrated_gate"


def test_independent_identity_mismatch_precedes_io(tmp_path):
    case = _case(tmp_path)
    case["expected"]["checkpoint_sha256"] = "9" * 64

    result = _run(case)

    assert result == {
        "status": "rejected",
        "reasons": ["checkpoint_sha256_mismatch"],
    }


def test_modified_dataset_bytes_rejected(tmp_path):
    case = _case(tmp_path)
    path = case["root"] / DATASET_PATH

    path.write_bytes(
        path.read_bytes().replace(
            b"Turn 0/150", b"Turn 1/150", 1
        )
    )

    assert _run(case)["reason"] == (
        "synthetic_file_sha256_mismatch"
    )


def test_wrong_episode_header_rejected(tmp_path):
    case = _case(tmp_path)
    header = {**case["header"], "seed": 999}

    assert _run(
        case, episode_header=header
    )["reason"] == "episode_identity_mismatch"


def test_cross_episode_current_row_rejected(tmp_path):
    case = _case(tmp_path)
    case["current"]["episode_id"] = "another-episode"

    result = _run(case)

    assert result["reason"] == "replay_unavailable"
    assert "row identity mismatch" in result["detail"]


def test_unmapped_row_index_rejected(tmp_path):
    case = _case(tmp_path)

    assert _run(case, row_index=1)["reason"] == (
        "selected_row_not_uniquely_mapped"
    )


def test_extractor_version_mismatch_rejected(tmp_path):
    case = _case(tmp_path)
    case["manifest"]["feature_extractor_version"] = (
        "synthetic-extractor-v1"
    )
    case["expected"]["feature_extractor_version"] = (
        "synthetic-extractor-v1"
    )

    assert _run(case)["reason"] == "feature_version_mismatch"


def test_learner_training_feature_mismatch_rejected(tmp_path):
    case = _case(tmp_path)
    case["current"]["observation_before"][
        "enterprise_revenue"
    ] = 105.0

    result = _run(case)

    assert result["reason"] == "feature_screen_unavailable"
    assert result["detail"]["reason"] == (
        "feature_parity_mismatch"
    )
    assert result["detail"]["section"] == "header"


def test_missing_persisted_training_text_rejected(tmp_path):
    case = _case(tmp_path)
    case["persisted"] = {"action": "synthetic-only"}
    _write_case(case)

    assert _run(case)["reason"] == (
        "missing_persisted_training_text"
    )


def test_compacted_training_text_excluded(tmp_path):
    case = _case(tmp_path)
    case["persisted"]["text"] = (
        case["variants"]["compacted"].replace(
            "Turn 7/150", "Turn 0/150", 1
        )
    )
    _write_case(case)

    result = _run(case)

    assert result["reason"] == "feature_screen_unavailable"
    assert result["detail"]["reason"] == (
        "partial_training_features"
    )


def test_truncated_training_text_excluded(tmp_path):
    case = _case(tmp_path)
    case["persisted"]["text"] = (
        case["variants"]["character_token_truncated"]
    )
    _write_case(case)

    result = _run(case)

    assert result["reason"] == "feature_screen_unavailable"
    assert result["detail"]["reason"] == "token_truncated"


def test_mapping_ledger_disagreement_rejected(tmp_path):
    case = _case(tmp_path)
    case["ledger_row"]["episode_id"] = "different-episode"
    _write_case(case)

    assert _run(case)["reason"] == (
        "synthetic_ledger_disagreement"
    )


def test_later_turn_is_not_silently_supported(tmp_path):
    case = _case(tmp_path)
    case["map_row"]["turn"] = 1
    case["ledger_row"]["turn"] = 1
    _write_case(case)

    assert _run(case)["reason"] == (
        "noninitial_turn_not_yet_supported"
    )
