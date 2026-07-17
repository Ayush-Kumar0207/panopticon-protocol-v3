"""Small trainable neural policy for the synthetic HYDRA adversary."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Categorical

from hydra_policy import HYDRA_POLICY_SCHEMA_VERSION, HydraPolicyObservation
from models import Department, LeakChannel


HYDRA_NEURAL_SCHEMA_VERSION = "hydra-neural-policy-v1"
DEPARTMENT_VOCAB = [department.value for department in Department]
CHANNEL_VOCAB = [channel.value for channel in LeakChannel]
HYDRA_OBS_SIZE = 10 + len(DEPARTMENT_VOCAB) * 2 + len(CHANNEL_VOCAB)


def encode_hydra_observation(observation: HydraPolicyObservation) -> torch.Tensor:
    """Encode only the declared HYDRA information boundary, never raw state."""
    values = [
        observation.turn / max(observation.max_turns, 1),
        observation.phase_number / 6.0,
        observation.enterprise_revenue / 200.0,
        observation.security_score / 100.0,
        observation.hydra_aggression,
        observation.recruitment_accuracy,
        min(observation.detection_count / 5.0, 1.0),
        min(observation.disinfo_received / 5.0, 1.0),
        min(observation.active_sleepers / 5.0, 1.0),
        min(observation.dormant_sleepers / 5.0, 1.0),
    ]
    values.extend(
        1.0 if department in observation.canary_departments_seen else 0.0
        for department in DEPARTMENT_VOCAB
    )
    values.extend(
        min(observation.agent_audit_pattern.get(department, 0) / 10.0, 1.0)
        for department in DEPARTMENT_VOCAB
    )
    values.extend(
        1.0 if channel in observation.monitored_channels else 0.0
        for channel in CHANNEL_VOCAB
    )
    tensor = torch.tensor(values, dtype=torch.float32)
    if tensor.shape != (HYDRA_OBS_SIZE,):
        raise RuntimeError(f"HYDRA encoder produced {tensor.shape}; expected {(HYDRA_OBS_SIZE,)}")
    return tensor


class HydraPolicyNetwork(nn.Module):
    """Shared MLP with department, channel, and false-flag policy heads."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(HYDRA_OBS_SIZE, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
        )
        self.department_head = nn.Linear(64, len(DEPARTMENT_VOCAB))
        self.channel_head = nn.Linear(64, len(CHANNEL_VOCAB))
        self.false_flag_head = nn.Linear(64, 2)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(observation)
        return (
            self.department_head(features),
            self.channel_head(features),
            self.false_flag_head(features),
        )


@dataclass
class NeuralDecisionRecord:
    kind: str
    log_probability: torch.Tensor
    entropy: torch.Tensor


class NeuralHydraPolicy:
    """HydraPolicy implementation backed by a trainable PyTorch network."""

    policy_name = "neural_hydra_v1"

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        model: HydraPolicyNetwork | None = None,
        deterministic: bool = True,
        record_gradients: bool = False,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = (model or HydraPolicyNetwork()).to(self.device)
        self.deterministic = deterministic
        self.record_gradients = record_gradients
        self.trajectory: list[NeuralDecisionRecord] = []
        self.checkpoint_metadata: dict[str, Any] = {}
        if checkpoint is not None:
            self.checkpoint_metadata = load_hydra_checkpoint(checkpoint, self.model, self.device)
        if not record_gradients:
            self.model.eval()

    def reset(self) -> None:
        self.trajectory.clear()

    def _logits(self, observation: HydraPolicyObservation):
        encoded = encode_hydra_observation(observation).to(self.device)
        return self.model(encoded)

    def _select(
        self,
        logits: torch.Tensor,
        allowed_indices: list[int],
        kind: str,
    ) -> int:
        if not allowed_indices:
            raise ValueError(f"No allowed indices for neural HYDRA decision {kind}")
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[allowed_indices] = True
        masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        distribution = Categorical(logits=masked_logits)
        choice = torch.argmax(masked_logits) if self.deterministic else distribution.sample()
        if self.record_gradients:
            self.trajectory.append(
                NeuralDecisionRecord(kind, distribution.log_prob(choice), distribution.entropy())
            )
        return int(choice.detach().cpu().item())

    def choose_spawn_department(
        self,
        observation: HydraPolicyObservation,
        generation: int,
        candidates: list[str],
        rng: random.Random,
    ) -> str:
        department_logits, _, _ = self._logits(observation)
        allowed = [DEPARTMENT_VOCAB.index(value) for value in candidates]
        return DEPARTMENT_VOCAB[self._select(department_logits, allowed, "spawn_department")]

    def choose_leak_channel(
        self,
        observation: HydraPolicyObservation,
        generation: int,
        candidates: list[str],
        rng: random.Random,
    ) -> str:
        _, channel_logits, _ = self._logits(observation)
        allowed = [CHANNEL_VOCAB.index(value) for value in candidates]
        return CHANNEL_VOCAB[self._select(channel_logits, allowed, "leak_channel")]

    def should_plant_false_flag(
        self,
        observation: HydraPolicyObservation,
        generation: int,
        rng: random.Random,
    ) -> bool:
        if generation < 3:
            return False
        _, _, false_flag_logits = self._logits(observation)
        return bool(self._select(false_flag_logits, [0, 1], "false_flag"))

    def episode_log_probability(self) -> torch.Tensor | None:
        if not self.trajectory:
            return None
        return torch.stack([record.log_probability for record in self.trajectory]).sum()

    def episode_entropy(self) -> torch.Tensor | None:
        if not self.trajectory:
            return None
        return torch.stack([record.entropy for record in self.trajectory]).mean()


def save_hydra_checkpoint(
    path: str | Path,
    model: HydraPolicyNetwork,
    *,
    metadata: dict[str, Any] | None = None,
    optimizer_state_dict: dict[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": HYDRA_NEURAL_SCHEMA_VERSION,
        "policy_schema_version": HYDRA_POLICY_SCHEMA_VERSION,
        "observation_size": HYDRA_OBS_SIZE,
        "department_vocab": DEPARTMENT_VOCAB,
        "channel_vocab": CHANNEL_VOCAB,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer_state_dict,
        "metadata": metadata or {},
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_hydra_checkpoint(
    path: str | Path,
    model: HydraPolicyNetwork,
    device: str | torch.device,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != HYDRA_NEURAL_SCHEMA_VERSION:
        raise RuntimeError("Unsupported or unversioned HYDRA checkpoint")
    if payload.get("observation_size") != HYDRA_OBS_SIZE:
        raise RuntimeError("HYDRA checkpoint uses a different observation schema")
    if payload.get("department_vocab") != DEPARTMENT_VOCAB or payload.get("channel_vocab") != CHANNEL_VOCAB:
        raise RuntimeError("HYDRA checkpoint vocabulary does not match this environment")
    model.load_state_dict(payload["state_dict"])
    return payload.get("metadata", {})

def load_hydra_training_state(
    path: str | Path,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
) -> dict[str, Any]:
    """Restore optimizer state and return resumable training metadata."""
    payload = torch.load(path, map_location=device, weights_only=True)
    optimizer_state = payload.get("optimizer_state_dict")
    if optimizer_state is None:
        raise RuntimeError("HYDRA checkpoint does not contain resumable optimizer state")
    optimizer.load_state_dict(optimizer_state)
    return payload.get("metadata", {})
