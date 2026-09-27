# Panopticon V5 training provenance audit

Audit date: 2026-09-27  
Tracked decision: [Ayush-Kumar0207/panopticon-protocol-v3#8](https://github.com/Ayush-Kumar0207/panopticon-protocol-v3/issues/8)  
Drive run: `panopticon-security-v5-ep50` (`1OXQgw1OA-cAAXxLp9t8YbOsjv9dIBB_-`)

## Decision

**BLOCKED. Do not authorize Stage 1 real-artifact analysis, retraining, new rollouts, GPU work, or held-out/final evaluation.**

The corpus identity and episode/seed mapping are verified, but the historical run does not authenticate the exact base-model revision and the persisted representation cannot supply the complete Stage 1 candidate feature set for any episode.

## Gate results

| Gate | Result | Evidence |
|---|---|---|
| Exact corpus bytes and row counts | PASS | All five original Drive JSONL files parsed, matched their declared counts, and have recorded SHA-256 digests. |
| Episode-to-row and seed mapping | PASS | All 88,896 physical weighted rows map to 29,000 ordered logical turns across 250 episodes. Turn sequences, duplicate weights, action payloads, and ordered seed hashes match the expert metrics. Combined row-map hash: `9165245171110c4969b16d653230e9c3efe958e4db2a8e4724847d68576575aa`. |
| Template and system prompt identity | PASS | Every recovered message boundary uses the expected system prompt hash `97b895a4c79c9c721634751bd3ecbce4a5c57cf288e4445163d8924275b89f76`. All saved tokenizer configs use chat-template hash `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`, equal to the current Stage 1 pin. |
| Exact base-model revision | FAIL | All five adapter configs identify `Qwen/Qwen2.5-1.5B-Instruct` but contain `"revision": null`. The current Stage 1 pin is `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; no historical artifact proves that this revision was loaded. |
| Source checkout identity | INCOMPLETE | The later run config records commit `4c1f2db6a7a3d3c88136d5b51fbcdfe4058ad818`. That commit exists, but its timestamp (`2026-06-15T15:40:45Z`) follows the first run and easy corpus generation (`09:13Z`/`09:14Z`). Its `train_trl_v2.py` is byte-identical to parent `fc737fcc5654d548ed75a764fd376c49d376fee6`, which predates the run, but the actual checkout was not logged contemporaneously. |
| Final checkpoint identity | INCOMPLETE | The final level-5 adapter is hashed: SHA-256 `66606d1a60cfca2a189eab6a55b4b2b492d358bfac9a021f408aff3c8af2aec4`, 73,911,112 bytes, Drive ID `1knAxoGR_uudgJlicw92pmZhveuopTzhZ`. The merged model is 3,087,467,144 bytes, Drive ID `10XNLjK06sGWc7SLOmUM6z22ZsHjCu5Rb`; the connector refuses downloads above 256 MiB, so no content digest was obtained. |
| Representation/schema compatibility | FAIL | 28,150 of 29,000 logical turns contain the explicit token-truncation marker. The other 850 turns omit clean-worker detail and expose only partial workforce features. Therefore 0 logical turns and 0 of 250 episodes have the complete candidate feature set required by the current Stage 1 gate. |
| Event-log integrity | WARNING | The log contains the expected five dataset-write and five training-complete events, followed by final level-5 completion and merges. It also contains one malformed JSONL line at line 42,506 (SHA-256 `871e604bc6ee8b01c1f8fd7c68690919ea81d1b0f7b65497f77a732926c01df0`) and 70 run starts caused by retries/resumes. |

## Corpus verification

| Level | Drive file ID | SHA-256 | Bytes | Physical rows | Logical turns | Truncated turns | Complete turns | Complete episodes |
|---|---|---|---:|---:|---:|---:|---:|---:|
| easy | `1PUSLgjC3SqeOh7Kz1BQ8IBL4HT9Dsv3q` | `a7681c548f308480767f0b61959cc9f87c2fe915433b11015b591b7ff2c34757` | 16,832,673 | 7,430 | 3,000 | 2,800 | 0 | 0 |
| medium | `1IeQTUwUtqW6cdfDEXvQ7SWyS-zCtaxOg` | `81419946862687f5f1a1b8d0ce61727d8a1b01fc14e2e38c4761d1a23b09f4ab` | 29,739,154 | 13,166 | 4,500 | 4,300 | 0 | 0 |
| hard | `1PMHV0-mUECJ0S4mtTKx7rCHOPyCwLfhj` | `92ba629d7b581da454193d78ff4e74fa83f3c0415f05c71fb516f55afefe8b81` | 41,652,003 | 18,414 | 6,000 | 5,850 | 0 | 0 |
| level_4 | `11AntK3ih3dZnYcZqux1CxCLEj3u8aDxm` | `8b641e8028ddb29c52bfcc49841b70d40d925d78b4d15eaf1cb5b9c2df5c2a60` | 54,056,494 | 23,889 | 7,500 | 7,350 | 0 | 0 |
| level_5 | `1CsurFnPXDYs-udj-zx7o32ABkogvFSEY` | `4b20f6bf3a6622de029fe6609bc82cdd982abb731fc56d0ea9f7ca3adadef9ff` | 58,775,714 | 25,997 | 8,000 | 7,850 | 0 | 0 |
| **Total** |  |  | **201,056,038** | **88,896** | **29,000** | **28,150** | **0** | **0** |

## Row-map evidence

| Level | Row-map SHA-256 | Ordered-seed SHA-256 |
|---|---|---|
| easy | `96128b3afcf02c4570d4293257874f7d42531618eb1e347695e158af3b9a6482` | `7455f94736900fedc5a59bb9801ae43c8fef087afb8f0d20617fe036214ca3d7` |
| medium | `269ad93f74023edf013e6cee4b96797a7979414dc22151fae64fd7d469c1159d` | `f0d68faa1d1286fa22ab5b90888a17a1fc74e578d290b226999ffe7320623ddf` |
| hard | `c4f2100c4a5e9eeb1f41cbded892e688f255608b3830ea88973caf3b2d0338aa` | `f43b7fddbd3612e6183361facc2cbb7ee12c4e1c399fd809143ffe298a0ac59c` |
| level_4 | `d02eed7f5b3070f9404feea47cdd62459bc44b8920a3994dd113137a0746faf3` | `b5cafb98110a7efba7e6e0ea2ad75bf4258cb349e022516cfade4bc6e31e4f19` |
| level_5 | `df2a72e6ac3eaef8d27338806b38fc2b58134b78a140f6bec378ea5ae98855c7` | `269466fc100a85ce40fd276d5d0d30cac45b308e22822137edbc74b82be36cc8` |

The mapping reconstructs logical episodes from turn resets, checks exact `0..steps-1` sequences against each metrics record, verifies every physical duplicate against the historical action-weight function, and binds each physical row hash to level, episode, seed, and turn.

## Training lineage

- The event log SHA-256 is `8401536f2af2dd9ef865f6df3594545fd6e4ef8ff434ba38822fc875cd51c756`.
- Dataset writes occurred from 2026-06-15 through 2026-06-27 and declared 29,000 raw examples, all compacted, producing 88,896 weighted rows.
- Training completion events exist once for each curriculum level. Level 5 completed at `2026-07-03T04:38:57.003Z` with two epochs and train loss `0.0002514301348768837`.
- Three merge-complete events target the same level-5 adapter and merged-model path. The latest completed at `2026-07-04T03:35:43.313Z`.
- `colab_run_config.json` later records seed 42, 50 episodes per level, the base model name, and source commit, but it does not repair the missing contemporaneous model revision or checkout identity.

## Required remediation

1. Keep Issue #8 open and preserve the current research boundary.
2. If complete Stage 1 state coverage is still required, regenerate a provenance-complete development corpus from an explicitly pinned source commit and exact model/tokenizer revision, without lossy observation compaction for the approved feature allowlist.
3. Record SHA-256 digests for every generated corpus, adapter/checkpoint, merged model, configuration, and event log in a manifest created during the run.
4. Run the existing provenance and feature-completeness gate on that manifest. Only a separate, explicit maintainer decision after all checks pass may authorize development-artifact analysis.

