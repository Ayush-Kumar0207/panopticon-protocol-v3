"""Synthetic regression test using the actual V5 training writer."""

import ast
import builtins
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

import pytest

from models import ActionType, SubAction
from research_paper.stage1_state_coverage.persisted_text import (
    CHAT_TEMPLATE_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    extract_pinned_observation,
)

ROOT = Path(__file__).resolve().parents[1]

FUNCTIONS = {
    "trajectory_training_weight",
    "render_training_text",
    "compact_observation_lines",
    "truncate_observation_tokens",
    "fit_training_text",
    "save_training_data_with_template",
}


def production_writer():
    source = (ROOT / "train_trl_v2.py").read_text(
        encoding="utf-8-sig"
    )
    tree = ast.parse(source)

    selected = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in FUNCTIONS
    ]
    assert {node.name for node in selected} == FUNCTIONS

    prompts = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "SYSTEM_PROMPT"
            for target in node.targets
        )
    ]
    assert len(prompts) == 1

    spec = json.loads(
        (ROOT / "training_specs/security_first_v5.json")
        .read_text(encoding="utf-8")
    )

    namespace = {
        "SYSTEM_PROMPT": prompts[0],
        "MAX_SEQ_LENGTH": spec["training"]["max_sequence_length"],
        "ActionType": ActionType,
        "SubAction": SubAction,
        "json": json,
        "os": os,
        "sys": sys,
        "log_event": lambda *args, **kwargs: None,
    }

    module = ast.fix_missing_locations(
        ast.Module(body=selected, type_ignores=[])
    )
    exec(compile(module, "train_trl_v2.py", "exec"), namespace)
    return namespace["save_training_data_with_template"]



class OfflineTemplateTokenizer:
    """Exercise the unmodified writer with the exact pinned template.

    The small synthetic observations require no compaction. This
    fixture-specific length check is not Qwen tokenization; the
    separate cached-tokenizer tests cover token-based transforms.
    """

    def __init__(self):
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        path = (
            ROOT / "tests/fixtures/stage1_state_coverage/"
            "qwen_pinned_chat_template.jinja"
        )
        self.chat_template = path.read_text(encoding="utf-8")
        assert hashlib.sha256(
            self.chat_template.encode("utf-8")
        ).hexdigest() == CHAT_TEMPLATE_SHA256

        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=["jinja2.ext.loopcontrols"],
        )
        self.renderer = environment.from_string(self.chat_template)

    def apply_chat_template(self, messages, tokenize=False):
        assert tokenize is False
        return self.renderer.render(
            messages=messages,
            tools=None,
            add_generation_prompt=False,
        )

    def __call__(self, text, truncation=False):
        assert truncation is False
        count = len(text.split())
        assert count <= 512, "Unexpected fixture length"
        return {"input_ids": list(range(count))}

    def encode(self, *args, **kwargs):
        raise AssertionError(
            "Unexpected compaction: real Qwen tokenizer required"
        )

    def decode(self, *args, **kwargs):
        raise AssertionError(
            "Unexpected truncation: real Qwen tokenizer required"
        )


def test_actual_writer_persists_two_turns_and_weighted_rows(tmp_path):
    tokenizer = OfflineTemplateTokenizer()

    helpers = runpy.run_path(
        str(ROOT / "tests/test_stage1_feature_parity.py")
    )
    formatter, _ = helpers["load_formatter"]()

    first = helpers["synthetic_observation"]()
    first.turn = 0

    # Keep this example small enough to fit the frozen token budget.
    first.workers = first.workers[:1]
    first.active_leaks = []
    first.canary_traps = []
    first.intel_reports = []
    first.double_agents = []

    second = first.model_copy(deep=True)
    second.turn = 1

    observations = [
        formatter["format_observation"](first),
        formatter["format_observation"](second),
    ]

    trajectories = [
        {
            "observation": observations[0],
            "action": json.dumps({
                "action_type": "canary",
                "target": "engineering",
            }),
        },
        {
            "observation": observations[1],
            "action": json.dumps({
                "action_type": "noop",
            }),
        },
    ]

    path = tmp_path / "production-synthetic.jsonl"

    actual_writer = production_writer()

    # Reproduce the frozen Windows CRLF serialization on all CI hosts.
    # The actual production-writer function remains unchanged.
    def fixture_open(file, mode="r", *args, **kwargs):
        assert mode == "w"
        assert str(file).endswith(".jsonl.tmp")
        assert "newline" not in kwargs
        return builtins.open(
            file, mode, *args, newline=chr(13) + chr(10), **kwargs
        )

    actual_writer.__globals__["open"] = fixture_open

    written = actual_writer(
        trajectories,
        str(path),
        tokenizer,
        "level_4",
    )

    expected_path = (
        ROOT / "tests/fixtures/stage1_state_coverage/"
        "production_pinned_two_turn_v1.jsonl"
    )
    expected_bytes = expected_path.read_bytes()
    actual_bytes = path.read_bytes()
    expected_hash = (
        "fc92a78607dde094b32c84a5ba68a2db0ac7cc8411179b368495d6861d7d22e5"
    )

    assert actual_bytes == expected_bytes
    assert hashlib.sha256(actual_bytes).hexdigest() == expected_hash

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]

    # The production writer weights canary actions twice.
    assert written == 3
    assert len(rows) == 3
    assert rows[0] == rows[1]
    assert rows[1] != rows[2]

    for row, observation in zip(
        [rows[0], rows[2]], observations
    ):
        assert row.keys() == {"text"}
        assert row["text"].startswith("<|im_start|>system\n")

        extracted = extract_pinned_observation(
            row["text"],
            tokenizer_identifier=MODEL_ID,
            tokenizer_revision=MODEL_REVISION,
            chat_template_sha256=CHAT_TEMPLATE_SHA256,
        )
        assert extracted == {
            "status": "available",
            "observation": observation,
        }


def _minimal_observation():
    helpers = runpy.run_path(
        str(ROOT / "tests/test_stage1_feature_parity.py")
    )
    observation = helpers["synthetic_observation"]()
    observation.turn = 1
    observation.workers = observation.workers[:1]
    observation.active_leaks = []
    observation.canary_traps = []
    observation.intel_reports = []
    observation.double_agents = []
    return observation, helpers


def _persist_one(tmp_path, tokenizer, observation):
    helpers = runpy.run_path(
        str(ROOT / "tests/test_stage1_feature_parity.py")
    )
    formatter, _ = helpers["load_formatter"]()
    raw = formatter["format_observation"](observation)

    output = tmp_path / "transform-probe.jsonl"
    count = production_writer()(
        [{
            "observation": raw,
            "action": '{"action_type":"noop"}',
        }],
        str(output),
        tokenizer,
        "level_4",
    )

    assert count == 1
    lines = output.read_bytes().splitlines()
    assert len(lines) == 1

    persisted = json.loads(lines[0])["text"]
    assert persisted.startswith("<|im_start|>system" + chr(10))
    return raw, persisted


def _pinned_tokenizer():
    import hashlib

    transformers = pytest.importorskip("transformers")
    pytest.importorskip("jinja2")

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
            trust_remote_code=False,
        )
    except OSError:
        pytest.skip("Pinned tokenizer unavailable locally")

    assert hashlib.sha256(
        tokenizer.chat_template.encode("utf-8")
    ).hexdigest() == CHAT_TEMPLATE_SHA256

    return tokenizer


def _extract(text):
    return extract_pinned_observation(
        text,
        tokenizer_identifier=MODEL_ID,
        tokenizer_revision=MODEL_REVISION,
        chat_template_sha256=CHAT_TEMPLATE_SHA256,
    )


def test_actual_writer_compacts_observation(tmp_path):
    from research_paper.stage1_state_coverage.feature_eligibility import (
        screen_synthetic_feature_parity,
    )
    from research_paper.stage1_state_coverage._synthetic_features import (
        CANDIDATE_FEATURE_VERSION,
    )

    tokenizer = _pinned_tokenizer()
    base, helpers = _minimal_observation()
    clean_worker = helpers["synthetic_observation"]().workers[1]

    for count in (6, 12, 24, 48):
        observation = base.model_copy(deep=True)

        for index in range(count):
            worker = clean_worker.model_copy(deep=True)
            worker.id = f"w-{100 + index:03d}"
            worker.name = f"Synthetic Clean {index}"
            observation.workers.append(worker)

        raw, persisted = _persist_one(
            tmp_path, tokenizer, observation
        )
        result = _extract(persisted)

        if (
            result["status"] == "available"
            and "clean loyal workers omitted:" in result["observation"]
            and result["observation"] != raw
        ):
            assert len(
                tokenizer(persisted, truncation=False)["input_ids"]
            ) <= 512

            parity = screen_synthetic_feature_parity(
                result["observation"],
                observation.model_dump(),
                synthetic=True,
                expected_feature_version=CANDIDATE_FEATURE_VERSION,
            )

            assert parity["status"] == "unavailable"
            return

    pytest.fail("No production compaction-only case found")


def test_actual_writer_token_truncation(tmp_path):
    tokenizer = _pinned_tokenizer()
    base, _ = _minimal_observation()

    for count in (12, 24, 48, 96):
        observation = base.model_copy(deep=True)

        for index in range(count):
            worker = observation.workers[0].model_copy(deep=True)
            worker.id = f"w-{200 + index:03d}"
            worker.name = f"Synthetic Suspect Extended Name {index}"
            observation.workers.append(worker)

        _, persisted = _persist_one(
            tmp_path, tokenizer, observation
        )
        result = _extract(persisted)

        if result.get("reason") == "token_truncated":
            assert result["status"] == "unavailable"
            assert len(
                tokenizer(persisted, truncation=False)["input_ids"]
            ) <= 512
            return

    pytest.fail("No production token-truncated case found")
