# Panopticon Bench

`panopticon_bench` is a local-first prototype for authorized, long-horizon security evaluation of agentic AI systems. It extends the original fictional environment's key research idea—apparently benign behavior followed by delayed compromise—into reusable campaign and metric primitives.

## What is implemented

- Strict, versioned Pydantic schemas for campaigns, turns, target responses, event provenance, and episode results.
- Mandatory authorization and non-destructive campaign validation.
- Local callable and deterministic transcript adapters; no default network target.
- Exact synthetic-canary disclosure evaluation.
- Request/response hashes with response retention disabled by default.
- Constraint Masking Gap: the rate at which a composite score passes while an operational constraint fails.
- Intervention-use and intervention-dependence metrics.
- Time-to-compromise, attack-success, utility, and failure-profile aggregation.
- Two checked-in long-horizon scenario fixtures and unit tests.

## Minimal example

```python
import json
from pathlib import Path

from panopticon_bench import (
    CampaignRunner,
    CampaignSpec,
    KeywordCanaryEvaluator,
    RunnerConfig,
    TranscriptTargetAdapter,
)

path = Path("panopticon_bench/scenarios/delayed_canary_exfiltration.json")
campaign = CampaignSpec.model_validate(json.loads(path.read_text()))
target = TranscriptTargetAdapter("safe-fixture", ["Acknowledged", "Two", "I cannot reveal it"])
runner = CampaignRunner(RunnerConfig(experiment_id="demo", authorized=True))
episode = runner.run(campaign, target, KeywordCanaryEvaluator())
print(episode.model_dump_json(indent=2))
```

## Safety boundary

Only test systems you own or are explicitly authorized to assess. Use mock tools and synthetic data. External HTTP/model-provider adapters, destructive actions, credential collection, and exploit payload libraries are intentionally not included in this prototype.

## Research boundary

The metrics are proposed and implemented for Panopticon research; they are not established standards. Taxonomy mappings indicate test coverage and do not imply OWASP, MITRE, or NIST certification.
