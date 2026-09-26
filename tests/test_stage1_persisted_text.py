"""Pinned production chat-template extraction tests."""

import ast
import hashlib
import runpy
from pathlib import Path

import pytest

from research_paper.stage1_state_coverage.persisted_text import (
    CHAT_TEMPLATE_SHA256,
    MODEL_ID,
    MODEL_REVISION,
    SYSTEM_PROMPT_SHA256,
    extract_pinned_observation,
)


ROOT = Path(__file__).resolve().parents[1]


def production_source():
    source = (ROOT / "train_trl_v2.py").read_text(
        encoding="utf-8-sig"
    )
    tree = ast.parse(source)

    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "render_training_text"
    ]
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

    assert len(functions) == 1
    assert len(prompts) == 1
    return functions[0], prompts[0]


def test_pinned_system_prompt_matches_checked_in_source():
    _, prompt = production_source()
    assert hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest() == SYSTEM_PROMPT_SHA256


@pytest.fixture(scope="module")
def production_example():
    transformers = pytest.importorskip("transformers")

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
            trust_remote_code=False,
        )
    except OSError:
        pytest.skip("Pinned tokenizer not cached locally")

    assert hashlib.sha256(
        tokenizer.chat_template.encode("utf-8")
    ).hexdigest() == CHAT_TEMPLATE_SHA256

    function, prompt = production_source()
    module = ast.fix_missing_locations(
        ast.Module(body=[function], type_ignores=[])
    )
    namespace = {"SYSTEM_PROMPT": prompt}
    exec(compile(module, "train_trl_v2.py", "exec"), namespace)

    helpers = runpy.run_path(
        str(ROOT / "tests/test_stage1_feature_parity.py")
    )
    formatter, _ = helpers["load_formatter"]()
    observation = helpers["synthetic_observation"]()
    observation.turn = 0

    raw = formatter["format_observation"](observation)
    text = namespace["render_training_text"](
        tokenizer,
        raw,
        '{"action_type":"noop"}',
    )

    assert text.startswith("<|im_start|>system\n")
    return text, raw


def extract(text, **overrides):
    identity = {
        "tokenizer_identifier": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
    }
    identity.update(overrides)
    return extract_pinned_observation(text, **identity)


def test_extracts_actual_production_rendering(production_example):
    text, raw = production_example
    result = extract(text)

    assert result == {
        "status": "available",
        "observation": raw,
    }


def test_rejects_raw_untemplated_observation(production_example):
    _, raw = production_example
    assert extract(raw)["status"] == "unavailable"


@pytest.mark.parametrize("damage", [
    lambda text: text + "unexpected suffix",
    lambda text: text.replace(
        "<|im_start|>user\n",
        "<|im_start|>other\n",
        1,
    ),
    lambda text: text.replace(
        "Current State:\n",
        "Current State:\n<|im_start|>user\n",
        1,
    ),
    lambda text: text.replace(
        "Turn 0/150",
        "[... compacted for training context ...]\nTurn 0/150",
        1,
    ),
])
def test_rejects_malformed_or_truncated_text(
    production_example, damage
):
    text, _ = production_example
    assert extract(damage(text))["status"] == "unavailable"


def test_rejects_wrong_template_identity(production_example):
    text, _ = production_example

    assert extract(
        text,
        tokenizer_revision="different-revision",
    ) == {
        "status": "unavailable",
        "reason": "unsupported_template_identity",
    }
