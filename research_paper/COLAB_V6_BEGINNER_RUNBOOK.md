# Panopticon V6 Colab Runbook

This guide accompanies `Panopticon_V6_Research_Colab.ipynb`. The notebook contains
all executable code required for the remaining GPU-heavy research work.

## 1. What the IDE error means

The message:

> Failed to save `Panopticon_Explained.md`: The content of the file is newer.

is a **save conflict**, not a lost Git commit. The editor kept an older unsaved
copy in memory while the repository file was updated on disk.

Do not click **Overwrite**. That would replace the newer 5,000-line disk file with
the older editor buffer.

Use this safe procedure:

1. Click **Compare**.
2. Treat the disk version as authoritative.
3. If the editor side contains a small manual note you still need, copy only that
   note into a new temporary file.
4. Close the old `Panopticon_Explained.md` editor tab.
5. Choose **Don't Save**, **Revert**, or **Discard Editor Changes** when prompted.
6. Reopen `Panopticon_Explained.md` from Explorer.

The repository commit preserves the complete disk version, so it can also be
recovered through Git.

## 2. Open the notebook in Google Colab

Use either method:

### Method A — upload the local file

1. Open <https://colab.research.google.com/>.
2. Sign in with the Google account that can access the Panopticon Drive folder.
3. Select **File → Upload notebook**.
4. Upload `Panopticon_V6_Research_Colab.ipynb` from the repository root.

### Method B — open it from GitHub

After the research tag is pushed:

1. Open the repository on GitHub.
2. Select `Panopticon_V6_Research_Colab.ipynb`.
3. Click **Open in Colab**, if shown.

The notebook checks out the immutable tag
`research-v6-pilot-2026-07-17-r3`, so later changes to `main` cannot silently alter
the experiment.

### Why Colab shows a crossed-out save/cloud icon

Opening a notebook from GitHub does not automatically save the notebook document
or its cell outputs to MyDrive. The icon is unrelated to `drive.mount()`.

- Click **Copy to Drive** if you want an editable notebook copy saved in MyDrive.
- The experiment files are written directly under `DRIVE_ROOT` even when the
  notebook document is not saved.
- Cell 9 writes, atomically renames, reads, and removes a probe file to prove that
  mounted-Drive persistence is working.

## 3. Select the runtime

1. In Colab select **Runtime → Change runtime type**.
2. Choose **T4 GPU** or another CUDA GPU.
3. Save.
4. Run Cell 1.

Cell 1 deliberately stops if a GPU is unavailable.

## 4. Run cells in order

Run Cells 1–15 in order.

- Cells 1–9 perform setup, validation, model hashing, configuration locking, and
  resume-state inspection.
- Cell 10 runs the scripted-HYDRA pilot.
- Cell 11 trains five independent neural-HYDRA models.
- Cell 12 plots training progress.
- Cell 13 evaluates all five neural models on unseen pilot seeds.
- Cell 14 produces episode-level, grouped, and paired CSV files.
- Cell 15 checks that the complete pilot matrix exists.
- Cell 16 is the locked final evaluation.
- Cell 17 creates an integrity manifest without copying the large trace files.

Do not skip validation cells. A deliberate red error prevents an invalid or mixed
experiment.

## 5. Checkpoint and Drive-space behavior

Neural-HYDRA checkpoints use the following design:

- one checkpoint file per neural initialization seed;
- one checkpoint after every completed training episode;
- optimizer state, episode number, moving baseline, recent scores, configuration,
  PyTorch RNG state, and CUDA RNG states are included;
- a new checkpoint is first written to `.pt.tmp`;
- only after the temporary file is complete does it atomically replace `.pt`;
- the previous checkpoint therefore remains usable during the write; and
- old checkpoints do not accumulate.

Five neural seeds therefore produce five rolling `.pt` files, not thousands of
checkpoint directories.

The training log stores one compact JSON line per episode. V6 evaluation stores
one verified record per completed episode. These are evidence, not redundant
checkpoints, and must not be deleted during the study.

## 6. Resume after a disconnect

If Colab disconnects:

1. Reopen the same notebook.
2. Select a GPU runtime again.
3. Run Cells 1–9.
4. Rerun the interrupted long cell.

Use:

- Cell 10 to resume the scripted pilot;
- Cell 11 to resume neural training;
- Cell 13 to resume neural pilot evaluation; or
- Cell 16 to resume a legitimately unlocked final evaluation.

Do not rename the Drive root, output directories, checkpoint files, model folder,
or research tag. The scripts compare frozen configuration hashes and refuse to
mix incompatible runs.

Completed neural-training seeds immediately report that the target is already
satisfied. Incomplete seeds restore the exact checkpoint RNG state. Evaluation
verifies existing episode records and skips their unique episode keys.

At most the single environment episode executing at the instant of a hard
disconnect must run again. Every completed checkpoint and evaluation record
remains on Drive.

## 7. Do not unlock the final split immediately

The final matrix contains 30,000 episodes:

- five ARGUS policies;
- five task levels;
- 200 frozen seeds per level;
- one scripted-HYDRA condition; and
- five independently trained neural-HYDRA conditions.

First complete the pilot and inspect:

- `analysis/pilot_episode_metrics.csv`;
- `analysis/pilot_group_summary.csv`;
- `analysis/pilot_paired_differences.csv`; and
- `analysis/hydra_training_curves.png`.

Only software/data errors may be repaired after viewing pilot results. Freeze the
policies, checkpoints, metrics, and analysis rules before unlocking final.

To unlock Cell 16:

```python
RUN_FINAL = True
FINAL_AUTHORIZATION_PHRASE = "I_UNDERSTAND_FINAL_SPLIT_IS_SINGLE_USE"
```

Never use final results to select the best neural seed, change a threshold, alter
the repair policy, or retrain a model. Report variability across all five neural
initializations.

## 8. Where results are saved

All persistent outputs are placed under:

```text
MyDrive/
└── panopticon-v6-research-20260717/
    ├── frozen_run_config.json
    ├── argus_model_manifest.json
    ├── runtime_sessions.jsonl
    ├── runtime_patch_history.jsonl   # present when a diagnostic patch was applied
    ├── console_logs/
    ├── checkpoints/
    ├── training_logs/
    ├── pilot/
    ├── final/
    ├── analysis/
    └── result_inventory.json
```

Do not copy the whole folder for every session. Resume in place. This avoids
duplicating model weights and episode traces.

## 9. Common errors

### “GPU is not enabled”

Select **Runtime → Change runtime type → T4 GPU** and rerun Cell 1.

### “Merged ARGUS model not found”

Confirm that Drive was mounted using the account that owns or can read:

```text
MyDrive/panopticon-security-v5-ep50/merged_model
```

### “Frozen configuration mismatch”

One of the experimental constants changed. Restore the original notebook values.
If a new experiment is intentional, choose a new `DRIVE_ROOT`; do not reuse the
old results folder.

### “Existing manifest has a different frozen configuration”

The output directory belongs to a different policy/checkpoint/configuration.
Restore the original command or use a new condition directory.

### `CalledProcessError` from Cell 10

The first run durably completed 21 random-policy episodes. It then reached
`random|level_5|2|922094758`, where `grade.passed` was a `numpy.bool_` rather than
a built-in Python `bool`. Canonical JSON hashing rejected that scalar before the
22nd record could be appended. The existing 21 records are intact.

Use the `research-v6-pilot-2026-07-17-r3` notebook. It:

1. converts the complete episode payload to JSON-native values and explicitly
   normalizes the grader pass flag;
2. includes seed `922094758` as a permanent serialization regression test;
3. removes unused optional packages that conflict with the merged text-only
   model's Transformers stack;
4. installs an exact compatible Transformers, Tokenizers, Hugging Face Hub,
   Accelerate, and Safetensors stack;
5. loads and unloads ARGUS once in Cell 7 as a preflight;
6. streams child stdout and stderr into Colab; and
7. saves persistent console output under `console_logs/` and structured failure
   details inside the condition `manifest.json`.

Do not delete `pilot/scripted/episodes.jsonl`. Rerun the updated notebook from
Cell 1 through Cell 10. The 21 verified episode keys are skipped, and evaluation
continues with the exact previously failing seed.

### `huggingface-hub>=0.34.0,<1.0` import error from Cell 5

The old editable install followed Panopticon's full production dependency graph:
Panopticon → OpenEnv → Gradio → Hugging Face Hub 1.x. Transformers 4.57.6
requires Hugging Face Hub below 1.0, so its import preflight correctly stopped.

Use the `research-v6-pilot-2026-07-17-r3` notebook. Cell 5 removes this unused
optional chain, pins Hub 0.36.2, Tokenizers 0.22.1, and Safetensors 0.8.0, then
installs Panopticon in editable mode with `--no-deps`. It verifies all resolved
versions before continuing. A fresh Colab runtime is recommended, but no Drive
files should be deleted.

### `Token indices sequence length ... (566 > 512)` warning

The stopped r2 run completed 75 valid random, heuristic, and security-first
baselines, followed by three provisional `model_raw` episodes. The older runtime
allowed prompts above the model's documented 512-token training limit, so those
model episodes must not be used in paper results.

The r3 notebook performs a one-time, audit-safe migration in Cell 10:

1. renames `pilot/scripted` to `pilot/scripted_pre_prompt512_r2`;
2. preserves every old record instead of deleting or rewriting it;
3. validates episode hashes and imports only the 75 context-independent baseline
   records into a new `pilot/scripted` result set;
4. ignores the provisional model rows;
5. freezes a 512-token prompt limit and 128-token generation limit in the manifest;
6. compacts structured observation sections until the complete prompt fits;
7. refuses to run if fitting would require silent token truncation; and
8. records prompt-token and compaction provenance on every model turn.

Rerun the r3 notebook from Cell 1 through Cell 10. A correct resume prints
`completed 75/125`, skips the 75 baselines, and begins again at
`model_raw|easy|1|516436961`.

### Drive is nearly full

Do not delete the current rolling `.pt` checkpoints or episode JSONL files.
Remove unrelated Drive files. The notebook itself does not accumulate historical
neural checkpoints.

### Colab stopped during `.pt.tmp`

Rerun through Cell 9. It removes the incomplete temporary file and retains the
last durable `.pt` checkpoint.

## 10. What remains manual

Only cloud-runtime actions remain manual:

1. opening the notebook;
2. selecting a GPU;
3. authorizing the Drive mount;
4. rerunning cells after Colab disconnects;
5. reviewing pilot outputs before final; and
6. deliberately unlocking the final cell after the protocol is frozen.

Training, checkpoint replacement, resumption, seed enforcement, evaluation,
analysis-table generation, and integrity manifests are automated.
