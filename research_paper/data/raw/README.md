# Raw Evidence Index

Large files remain at the repository root to avoid duplicating them in the paper package.

| Repository file | Approximate size | Role |
|---|---:|---|
| `evaluationResults.json` | 238 MB | Large raw/historical evaluation artifact; inspect schema/version before analysis. |
| `training_events_fixed_ep20.jsonl` | 9.4 MB | Training-event stream for the fixed episode-20 run. |
| `evaluation_comparison_latest.json` | 9.3 KB | Compact current V5 comparison used by the draft. |
| `evaluation_snapshot_apr26.json` | 6.9 KB | Older April snapshot; not interchangeable with the current V5 comparison. |
| `plots/training_statistics.json` | compact | Training and comparison summary supporting existing plots. |
| `research_paper/data/raw/v5_drive_seed_evidence.json` | compact | Read-only extraction of 250 ordered seeds and source identities from the five original Drive expert-metrics artifacts. |

`scripts/extract_metrics.py` writes `SHA256SUMS.txt` for repository-root evidence files. The compact Drive seed snapshot is integrity-bound inside `training_seed_ledger.drive_verified.json` through a canonical SHA-256 digest and per-level ordered-seed digests. A paper release should archive raw logs in an immutable repository and record a DOI/URL. Do not combine the older April snapshot with current V5 numbers without labeling the experiment/version change.
