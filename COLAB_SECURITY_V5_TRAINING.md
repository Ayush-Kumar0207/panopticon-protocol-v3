# Security-First V5 Colab Training

This notebook sequence trains the Security-First V5 model on a T4 GPU while
writing checkpoints, curriculum state, datasets, metrics, and logs directly to
Google Drive.

Before starting, push the latest repository changes containing these files to
GitHub:

- `security_policy.py`
- `security_regression_test.py`
- `benchmark_acceptance.py`
- `train_trl_v2.py` with `curriculum-expert-v5-security-first`

The source-verification cell below deliberately stops if Colab cloned an older
version. This prevents accidentally repeating the previous unsafe training run.

## Resume Guarantees

- On a T4, optimizer checkpoints are written every 25 training steps.
- The latest two checkpoints for each curriculum level are retained.
- Completed levels are recorded in Drive at `curriculum_state.json`.
- Generated datasets, expert metrics, event logs, adapters, and the merged
  model are stored under the Drive run directory.
- After a disconnect, reconnect to a T4, rerun Cells 1-8, then rerun Cell 9
  with exactly the same model, episode count, seed, and `TRAIN_ROOT`.
- Do not resume using a CPU, A100, or a different `TRAIN_ROOT`. A changed
  runtime profile or configuration intentionally forces fresh compatible data.
- If the runtime is killed during the short expert-generation phase before its
  level dataset is written, that level's expert generation restarts. Optimizer
  training progress resumes from its latest Drive checkpoint.

## Cell 1 - Mount Google Drive

```python
from google.colab import drive

drive.mount("/content/drive", force_remount=False)
```

## Cell 2 - Define the Immutable Run Configuration

```python
import os
from pathlib import Path

REPO_URL = "https://github.com/Ayush-Kumar0207/panopticon-protocol-v3.git"
SOURCE_ROOT = Path("/content/panopticon-protocol-v3-security-v5")
TRAIN_ROOT = Path("/content/drive/MyDrive/panopticon-security-v5-ep50")

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EPISODES = 50
SEED = 42

TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["TRAIN_ROOT"] = str(TRAIN_ROOT)
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Source:", SOURCE_ROOT)
print("Persistent run root:", TRAIN_ROOT)
```

## Cell 3 - Clone or Refresh the Security-First Source

```python
import subprocess

if not (SOURCE_ROOT / ".git").exists():
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(SOURCE_ROOT)],
        check=True,
    )
else:
    subprocess.run(["git", "-C", str(SOURCE_ROOT), "pull", "--ff-only"], check=True)

required_files = [
    "security_policy.py",
    "security_regression_test.py",
    "benchmark_acceptance.py",
    "train_trl_v2.py",
]
missing = [name for name in required_files if not (SOURCE_ROOT / name).exists()]
if missing:
    raise RuntimeError(
        "The GitHub repository does not contain the Security-First V5 changes. "
        f"Missing: {missing}. Push the latest local changes before training."
    )

training_source = (SOURCE_ROOT / "train_trl_v2.py").read_text(encoding="utf-8")
if "curriculum-expert-v5-security-first" not in training_source:
    raise RuntimeError("Refusing to train: Colab cloned an older trajectory schema.")

commit = subprocess.check_output(
    ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
    text=True,
).strip()
print("Security-First source commit:", commit)
```

## Cell 4 - Install the Training Stack

Do not install the repository's pinned `torch==2.2.1`; keep Colab's
CUDA-enabled PyTorch build.

```python
%pip install -q --upgrade \
  "transformers==4.46.3" \
  "tokenizers==0.20.3" \
  "accelerate==0.34.2" \
  "peft==0.12.0" \
  "trl==0.12.2" \
  "datasets==3.1.0" \
  "huggingface-hub==0.26.5" \
  "safetensors==0.4.5" \
  "pydantic==2.6.1" \
  "matplotlib>=3.8"
```

## Cell 5 - Require a T4-Class Runtime

```python
import torch

if not torch.cuda.is_available():
    raise RuntimeError("GPU is not enabled. Select Runtime > Change runtime type > T4 GPU.")

gpu_name = torch.cuda.get_device_name(0)
gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

print("GPU:", gpu_name)
print(f"VRAM: {gpu_gb:.1f} GB")
print("PyTorch:", torch.__version__)

if gpu_gb >= 20:
    raise RuntimeError(
        "This persistent run is configured for the low-VRAM T4 profile. "
        "Reconnect using a T4 so checkpoint/runtime profiles remain compatible."
    )
```

## Cell 6 - Lock and Save the Run Configuration

```python
import json
import subprocess

commit = subprocess.check_output(
    ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
    text=True,
).strip()

run_config = {
    "schema": "security-first-v5-colab-run-v1",
    "source_commit": commit,
    "base_model": BASE_MODEL,
    "episodes": EPISODES,
    "seed": SEED,
    "train_root": str(TRAIN_ROOT),
    "gpu_profile": "low-vram-t4",
}
config_path = TRAIN_ROOT / "colab_run_config.json"

if config_path.exists():
    existing = json.loads(config_path.read_text(encoding="utf-8"))
    if existing != run_config:
        raise RuntimeError(
            "Run configuration changed. Use the original values or choose a new TRAIN_ROOT.\n"
            f"Existing: {existing}\nRequested: {run_config}"
        )
else:
    config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

print(config_path.read_text(encoding="utf-8"))
```

## Cell 7 - Validate the Security-First Environment

```python
import subprocess
import sys

subprocess.run([sys.executable, "smoke_test.py"], cwd=SOURCE_ROOT, check=True)
subprocess.run([sys.executable, "security_regression_test.py"], cwd=SOURCE_ROOT, check=True)
```

Expected advanced-tier results include:

- Level 4: security `100`, caught `4/4`, missed `0`
- Level 5: security `100`, caught `5/5`, missed `0`

## Cell 8 - Inspect Resume State and Existing Checkpoints

Run this cell before every training/resume attempt.

```python
import json
from pathlib import Path

state_path = TRAIN_ROOT / "curriculum_state.json"
events_path = TRAIN_ROOT / "training_events.jsonl"
levels = ["easy", "medium", "hard", "level_4", "level_5"]

if state_path.exists():
    print("Curriculum state:")
    print(json.dumps(json.loads(state_path.read_text(encoding="utf-8")), indent=2))
else:
    print("No curriculum state yet. Training will start from easy.")

print("\nLatest checkpoints:")
for level in levels:
    model_dir = TRAIN_ROOT / f"trl_model_{level}"
    checkpoints = sorted(
        model_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    print(f"{level:>8}: {checkpoints[-1] if checkpoints else 'none'}")

if events_path.exists():
    lines = events_path.read_text(encoding="utf-8", errors="replace").splitlines()
    print("\nLast structured events:")
    for line in lines[-8:]:
        print(line)
```

## Cell 9 - Start or Resume Training

This is the only training cell. Rerun the exact same cell after a disconnect.
It streams output to Colab and appends a full console log in Drive.

```python
import os
import subprocess
import sys
from datetime import datetime, timezone

cmd = [
    sys.executable,
    "-u",
    "train_trl_v2.py",
    "--model", BASE_MODEL,
    "--curriculum",
    "--episodes", str(EPISODES),
    "--seed", str(SEED),
    "--merge",
]

env = os.environ.copy()
env["TRAIN_ROOT"] = str(TRAIN_ROOT)
log_path = TRAIN_ROOT / "console_security_v5.log"

with log_path.open("a", encoding="utf-8", buffering=1) as log:
    header = (
        f"\n\n===== TRAIN/RESUME {datetime.now(timezone.utc).isoformat()} =====\n"
        f"Command: {' '.join(cmd)}\n"
    )
    print(header, end="")
    log.write(header)

    process = subprocess.Popen(
        cmd,
        cwd=SOURCE_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)

    return_code = process.wait()

if return_code != 0:
    raise RuntimeError(
        f"Training stopped with code {return_code}. "
        "Reconnect to a T4, rerun Cells 1-8, then rerun this cell to resume."
    )

print("Training and merge completed.")
print("Merged model:", TRAIN_ROOT / "merged_model")
```

## Cell 10 - Check Progress After a Disconnect or Completed Run

```python
import json

state_path = TRAIN_ROOT / "curriculum_state.json"
if state_path.exists():
    state = json.loads(state_path.read_text(encoding="utf-8"))
    print(json.dumps(state, indent=2))
    print("Completed levels:", state.get("completed_levels", []))

merged_model = TRAIN_ROOT / "merged_model"
print("Merged model exists:", merged_model.exists())
print("Structured event log:", TRAIN_ROOT / "training_events.jsonl")
print("Console log:", TRAIN_ROOT / "console_security_v5.log")
```

## Cell 11 - Verify the Merged Model Files

```python
merged_model = TRAIN_ROOT / "merged_model"

if not merged_model.exists():
    raise RuntimeError("Merged model is missing. Rerun the training/resume cell.")

files = sorted(path.name for path in merged_model.iterdir())
print("\n".join(files))

if "config.json" not in files:
    raise RuntimeError("Merged model is incomplete: config.json is missing.")
if not any(name.endswith(".safetensors") for name in files):
    raise RuntimeError("Merged model is incomplete: no safetensors weights found.")

print("Merged model verification passed.")
```

## Cell 12 - Run the Matched Base Evaluation

Evaluation outputs are also saved directly to Drive. Evaluation itself does not
currently resume from episode checkpoints, so run the base and candidate in
separate cells.

```python
import subprocess
import sys

base_results = TRAIN_ROOT / "evaluationResults_base_security_v2.json"
base_plots = TRAIN_ROOT / "plots_base_security_v2"

cmd = [
    sys.executable, "-u", "full_evaluation.py",
    "--model", BASE_MODEL,
    "--episodes", "20",
    "--seed", str(SEED),
    "--output", str(base_results),
    "--plot-dir", str(base_plots),
]
subprocess.run(cmd, cwd=SOURCE_ROOT, check=True)
print("Saved:", base_results)
```

## Cell 13 - Run the Matched Candidate Evaluation

```python
import subprocess
import sys

candidate_results = TRAIN_ROOT / "evaluationResults_fixed_security_v2.json"
candidate_plots = TRAIN_ROOT / "plots_fixed_security_v2"

cmd = [
    sys.executable, "-u", "full_evaluation.py",
    "--model", str(TRAIN_ROOT / "merged_model"),
    "--episodes", "20",
    "--seed", str(SEED),
    "--output", str(candidate_results),
    "--plot-dir", str(candidate_plots),
]
subprocess.run(cmd, cwd=SOURCE_ROOT, check=True)
print("Saved:", candidate_results)
```

## Cell 14 - Enforce the Release Gate

The model is not considered fully successful unless this cell prints
`Accepted: True`.

```python
import subprocess
import sys

acceptance_report = TRAIN_ROOT / "benchmark_acceptance_report.json"

cmd = [
    sys.executable, "-u", "benchmark_acceptance.py",
    "--base", str(TRAIN_ROOT / "evaluationResults_base_security_v2.json"),
    "--candidate", str(TRAIN_ROOT / "evaluationResults_fixed_security_v2.json"),
    "--report", str(acceptance_report),
]
subprocess.run(cmd, cwd=SOURCE_ROOT, check=True)
print("Acceptance report:", acceptance_report)
```

## Exact Resume Procedure

After a runtime disconnect or Colab GPU usage-limit interruption:

1. Reconnect using a T4 GPU.
2. Rerun Cells 1-8.
3. Confirm Cell 8 shows the expected completed levels/checkpoint.
4. Rerun Cell 9 without changing any configuration value.

The training script will skip completed curriculum levels and resume the
currently interrupted level from its latest Drive checkpoint.
