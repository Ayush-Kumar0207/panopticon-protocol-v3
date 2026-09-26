
"""Provisional Stage 1 training-provenance manifest contract.

Structural validation and comparison with caller-supplied expectations
cannot authenticate provenance or establish an episode/seed mapping.
This module performs no file I/O and never authorizes real-data analysis.
"""

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


TRAINING_MANIFEST_VERSION = "panopticon-stage1-training-provenance-v1"
_HASH = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
_COMMIT = Field(min_length=40, max_length=40, pattern=r"^[0-9a-f]{40}$")


class _StrictRecord(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    @field_validator("*", mode="after")
    @classmethod
    def reject_blank_or_padded_strings(cls, value):
        if type(value) is str and (not value.strip() or value != value.strip()):
            raise ValueError("blank or padded string")
        return value


class HashedFile(_StrictRecord):
    path: str
    sha256: str = _HASH
    bytes: int = Field(gt=0)

    @field_validator("path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        if (
            not value
            or "\\" in value
            or ":" in value
            or any(ord(ch) < 32 or ord(ch) == 127 for ch in value)
        ):
            raise ValueError("unsafe relative path")
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or ".." in path.parts
            or str(path) != value
            or value == "."
        ):
            raise ValueError("unsafe relative path")
        return value


class TrainingFile(HashedFile):
    rows: int = Field(gt=0)


class TokenizerIdentity(_StrictRecord):
    identifier: str
    revision: str
    chat_template_sha256: str = _HASH


class FormatterIdentity(_StrictRecord):
    source_commit: str = _COMMIT
    function: Literal["train_trl_v2.format_observation"]


class TextTransform(_StrictRecord):
    compaction: Literal["none", "present", "unknown"]
    token_truncation: Literal["none", "present", "unknown"]


class RowMappingEvidence(_StrictRecord):
    mapping_file: HashedFile
    independent_evidence: HashedFile
    mapped_rows: int = Field(gt=0)
    episode_groups: int = Field(gt=0)
    claimed_method: Literal["original_row_metadata", "independent_ledger"]


class TrainingManifest(_StrictRecord):
    schema_version: Literal[TRAINING_MANIFEST_VERSION]
    synthetic: bool
    experiment_id: str
    run_fingerprint: str = _HASH
    checkpoint_sha256: str = _HASH
    source_commit: str = _COMMIT
    training_stage: str
    feature_extractor_version: str
    dataset_files: list[TrainingFile] = Field(min_length=1)
    metadata_files: list[HashedFile] = Field(min_length=1)
    tokenizer: TokenizerIdentity
    formatter: FormatterIdentity
    transforms: TextTransform
    mapping: RowMappingEvidence

    @model_validator(mode="after")
    def check_consistency(self):
        total_rows = sum(item.rows for item in self.dataset_files)
        if self.mapping.mapped_rows != total_rows:
            raise ValueError("mapping_must_cover_every_persisted_row")
        if self.mapping.episode_groups > total_rows:
            raise ValueError("more_episode_groups_than_rows")
        paths = (
            [file.path for file in self.dataset_files]
            + [file.path for file in self.metadata_files]
            + [self.mapping.mapping_file.path, self.mapping.independent_evidence.path]
        )
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate_artifact_path")
        return self


_EXPECTED_FIELDS = frozenset({
    "experiment_id", "run_fingerprint", "checkpoint_sha256",
    "source_commit", "training_stage", "feature_extractor_version",
    "tokenizer_identifier", "tokenizer_revision", "chat_template_sha256",
    "formatter_source_commit", "formatter_function",
    "compaction", "token_truncation",
    "dataset_sha256_by_path", "dataset_bytes_by_path", "dataset_rows_by_path",
    "metadata_sha256_by_path", "metadata_bytes_by_path",
    "mapping_file_sha256", "mapping_file_path", "mapping_file_bytes",
    "mapping_evidence_sha256", "mapping_evidence_path", "mapping_evidence_bytes",
    "mapped_rows", "episode_groups", "claimed_mapping_method",
})


def _strict_equal(actual, expected):
    """Compare nested expectations without bool/int or numeric coercion."""
    if type(actual) is not type(expected):
        return False
    if type(actual) is dict:
        return actual.keys() == expected.keys() and all(
            _strict_equal(value, expected[key])
            for key, value in actual.items()
        )
    if type(actual) is list:
        return len(actual) == len(expected) and all(
            _strict_equal(a, b) for a, b in zip(actual, expected)
        )
    return actual == expected


def validate_training_manifest(manifest: object, expected_identity: object) -> dict:
    """Check structure and compare independently supplied declarations.

    The caller must authenticate expected_identity separately. Even a match
    leaves the file bytes, evidence independence, and row mapping unverified.
    """
    try:
        record = TrainingManifest.model_validate(manifest)
    except ValidationError as exc:
        reasons = sorted({
            (".".join(map(str, error["loc"])) or "manifest")
            + ": " + error["msg"]
            for error in exc.errors()
        })
        return {"status": "rejected", "reasons": reasons}

    if record.synthetic is not True:
        return {"status": "unavailable", "reason": "real_requires_integrated_gate"}

    if type(expected_identity) is not dict or set(expected_identity) != _EXPECTED_FIELDS:
        return {"status": "unavailable", "reason": "missing_independent_expectations"}

    actual = {
        "experiment_id": record.experiment_id,
        "run_fingerprint": record.run_fingerprint,
        "checkpoint_sha256": record.checkpoint_sha256,
        "source_commit": record.source_commit,
        "training_stage": record.training_stage,
        "feature_extractor_version": record.feature_extractor_version,
        "tokenizer_identifier": record.tokenizer.identifier,
        "tokenizer_revision": record.tokenizer.revision,
        "chat_template_sha256": record.tokenizer.chat_template_sha256,
        "formatter_source_commit": record.formatter.source_commit,
        "formatter_function": record.formatter.function,
        "compaction": record.transforms.compaction,
        "token_truncation": record.transforms.token_truncation,
        "dataset_sha256_by_path": {
            file.path: file.sha256 for file in record.dataset_files
        },
        "dataset_bytes_by_path": {
            file.path: file.bytes for file in record.dataset_files
        },
        "dataset_rows_by_path": {
            file.path: file.rows for file in record.dataset_files
        },
        "metadata_sha256_by_path": {
            file.path: file.sha256 for file in record.metadata_files
        },
        "metadata_bytes_by_path": {
            file.path: file.bytes for file in record.metadata_files
        },
        "mapping_file_sha256": record.mapping.mapping_file.sha256,
        "mapping_file_path": record.mapping.mapping_file.path,
        "mapping_file_bytes": record.mapping.mapping_file.bytes,
        "mapping_evidence_sha256": record.mapping.independent_evidence.sha256,
        "mapping_evidence_path": record.mapping.independent_evidence.path,
        "mapping_evidence_bytes": record.mapping.independent_evidence.bytes,
        "mapped_rows": record.mapping.mapped_rows,
        "episode_groups": record.mapping.episode_groups,
        "claimed_mapping_method": record.mapping.claimed_method,
    }
    mismatches = sorted(
        field for field in _EXPECTED_FIELDS
        if not _strict_equal(actual[field], expected_identity[field])
    )
    if mismatches:
        return {
            "status": "rejected",
            "reasons": [field + "_mismatch" for field in mismatches],
        }

    return {
        "status": "unverified",
        "reason": "structure_only_bytes_and_mapping_not_verified",
        "declared_dataset_rows": sum(file.rows for file in record.dataset_files),
        "declared_episode_groups": record.mapping.episode_groups,
    }
