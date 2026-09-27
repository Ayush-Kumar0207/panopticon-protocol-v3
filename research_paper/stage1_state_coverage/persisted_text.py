"""Strict extraction of the pinned Qwen V5 training representation.

Synthetic-only Stage 1 infrastructure. Successful extraction does not
authenticate historical provenance or authorize real-artifact analysis.
"""

import hashlib

from ._synthetic_features import TRUNCATION_MARKER


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
CHAT_TEMPLATE_SHA256 = (
    "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
)
SYSTEM_PROMPT_SHA256 = (
    "97b895a4c79c9c721634751bd3ecbce4a5c57cf288e4445163d8924275b89f76"
)

_SYSTEM_START = "<|im_start|>system\n"
_USER_START = "<|im_end|>\n<|im_start|>user\nCurrent State:\n"
_ASSISTANT_START = (
    "\n\nYour action (JSON):<|im_end|>\n"
    "<|im_start|>assistant\n"
)
_END = "<|im_end|>\n"


def extract_pinned_observation(
    training_text,
    *,
    tokenizer_identifier,
    tokenizer_revision,
    chat_template_sha256,
):
    """Recognize the exact pinned system/user/assistant representation."""

    identity = (
        tokenizer_identifier == MODEL_ID
        and tokenizer_revision == MODEL_REVISION
        and chat_template_sha256 == CHAT_TEMPLATE_SHA256
    )
    if not identity:
        return {
            "status": "unavailable",
            "reason": "unsupported_template_identity",
        }

    if type(training_text) is not str:
        return {"status": "unavailable", "reason": "invalid_training_text"}

    if (
        not training_text.startswith(_SYSTEM_START)
        or not training_text.endswith(_END)
        or training_text.count("<|im_start|>") != 3
        or training_text.count("<|im_end|>") != 3
        or training_text.count(_USER_START) != 1
        or training_text.count(_ASSISTANT_START) != 1
    ):
        return {"status": "unavailable", "reason": "invalid_template_boundaries"}

    body = training_text[len(_SYSTEM_START):]
    system_prompt, separator, body = body.partition(_USER_START)

    if not separator:
        return {"status": "unavailable", "reason": "missing_user_boundary"}

    if (
        hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        != SYSTEM_PROMPT_SHA256
    ):
        return {"status": "unavailable", "reason": "system_prompt_mismatch"}

    observation, separator, assistant = body.partition(_ASSISTANT_START)

    if (
        not separator
        or not observation
        or not assistant.endswith(_END)
        or not assistant[:-len(_END)].strip()
    ):
        return {"status": "unavailable", "reason": "invalid_message_content"}

    if TRUNCATION_MARKER in observation:
        return {"status": "unavailable", "reason": "token_truncated"}

    return {
        "status": "available",
        "observation": observation,
    }
