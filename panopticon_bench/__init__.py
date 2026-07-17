"""Long-horizon agent-security evaluation primitives for Panopticon.

This package is local-first and does not ship a network target. External systems
must be connected through an explicitly authorized adapter.
"""

from .adapters import CallableTargetAdapter, TargetAdapter, TranscriptTargetAdapter
from .metrics import BenchmarkMetrics, compute_benchmark_metrics
from .runner import CampaignRunner, KeywordCanaryEvaluator, MemoryEventSink, RunnerConfig
from .schemas import (
    CampaignSpec,
    CampaignStep,
    EpisodeRecord,
    EventRecord,
    InterventionLevel,
    TargetRequest,
    TargetResponse,
    TaxonomyReference,
)

__all__ = [
    "BenchmarkMetrics",
    "CallableTargetAdapter",
    "CampaignRunner",
    "CampaignSpec",
    "CampaignStep",
    "EpisodeRecord",
    "EventRecord",
    "InterventionLevel",
    "KeywordCanaryEvaluator",
    "MemoryEventSink",
    "RunnerConfig",
    "TargetAdapter",
    "TargetRequest",
    "TargetResponse",
    "TaxonomyReference",
    "TranscriptTargetAdapter",
    "compute_benchmark_metrics",
]
