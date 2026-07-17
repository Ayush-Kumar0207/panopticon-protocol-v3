#!/usr/bin/env python3
"""Generate the complete resume-safe Panopticon V6 Google Colab notebook."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "Panopticon_V6_Research_Colab.ipynb"


def source_lines(source: str) -> list[str]:
    normalized = textwrap.dedent(source).strip("\n") + "\n"
    return normalized.splitlines(keepends=True)


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines(source)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines(source),
    }


CELLS = [
    markdown(
        """
        # Panopticon Protocol V6 — Complete Resume-Safe Research Run

        This notebook performs the remaining compute-heavy research work:

        1. mounts Google Drive;
        2. checks out the immutable research release;
        3. validates code and seed separation;
        4. verifies and fingerprints the existing ARGUS model;
        5. runs the scripted-HYDRA pilot;
        6. trains five independent neural-HYDRA models;
        7. evaluates every neural model on the same unseen pilot seeds;
        8. creates paired analysis tables and plots; and
        9. exposes the untouched final evaluation only behind an explicit safety switch.

        **Resume guarantee:** neural-HYDRA saves one rolling atomic checkpoint after every
        completed episode. The previous checkpoint is replaced only after the new file is
        safely written, so checkpoints do not accumulate. V6 evaluation appends one durable
        record per completed episode. If Colab disconnects, reconnect and rerun the notebook
        from the top; training and evaluation continue instead of starting over.

        Do not tune anything using final-split results. Run the guarded final cell only after
        reviewing the pilot and freezing all decisions.
        """
    ),
    markdown(
        """
        ## Before Cell 1

        In Colab choose **Runtime → Change runtime type → T4 GPU** (or another CUDA GPU).
        Then run every cell in order. Green output means continue. A red exception is a
        deliberate safety stop: read its message before changing anything.

        Persistent files are stored under
        `MyDrive/panopticon-v6-research-20260717`. Reusing that exact folder is what enables
        resume. Do not delete it between sessions.

        When this notebook is opened from GitHub, Colab may show a crossed-out save/cloud icon.
        That icon concerns the notebook document, not mounted-Drive experiment outputs. Click
        **Copy to Drive** if you want an editable notebook copy; output persistence is verified
        separately inside Cell 9.
        """
    ),
    code(
        """
        # Cell 1 — Require a GPU before doing expensive setup.
        import platform
        import sys
        import torch

        print("Python:", sys.version)
        print("Platform:", platform.platform())
        print("PyTorch:", torch.__version__)

        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU is not enabled. In Colab choose Runtime > Change runtime type > T4 GPU, "
                "then rerun this cell."
            )

        GPU_NAME = torch.cuda.get_device_name(0)
        GPU_GB = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print("GPU:", GPU_NAME)
        print(f"GPU memory: {GPU_GB:.1f} GB")
        """
    ),
    code(
        """
        # Cell 2 — Mount the Google Drive that contains the V5 model.
        from google.colab import drive

        drive.mount("/content/drive", force_remount=False)
        print("Drive mounted for experiment outputs.")
        print(
            "The Colab save icon is separate. Use Copy to Drive only if you want the notebook "
            "document itself stored in MyDrive."
        )
        """
    ),
    code(
        """
        # Cell 3 — Define the immutable experiment configuration.
        import os
        from pathlib import Path

        REPO_URL = "https://github.com/Ayush-Kumar0207/panopticon-protocol-v3.git"
        RESEARCH_TAG = "research-v6-pilot-2026-07-17-r2"
        ALLOWED_PREVIOUS_RESEARCH_TAGS = {
            "research-v6-pilot-2026-07-17",
            "research-v6-pilot-2026-07-17-r1",
        }
        SOURCE_ROOT = Path("/content/panopticon-protocol-v3-v6")

        DRIVE_ROOT = Path("/content/drive/MyDrive/panopticon-v6-research-20260717")
        ARGUS_MODEL = Path(
            "/content/drive/MyDrive/panopticon-security-v5-ep50/merged_model"
        )

        HYDRA_TRAINING_SEEDS = [
            2026071701,
            2026071702,
            2026071703,
            2026071704,
            2026071705,
        ]
        HYDRA_TARGET_EPISODES = 2000

        # One rolling atomic checkpoint is written after EVERY completed episode.
        # Storage stays constant because the prior checkpoint is replaced.
        HYDRA_CHECKPOINT_EVERY = 1

        POLICIES = "random,heuristic,security_first,model_raw,model_repair"
        MODEL_LABEL = "argus-security-v5-ep50"
        MAX_STEPS = 300

        DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
        (DRIVE_ROOT / "checkpoints").mkdir(exist_ok=True)
        (DRIVE_ROOT / "training_logs").mkdir(exist_ok=True)
        (DRIVE_ROOT / "pilot").mkdir(exist_ok=True)
        (DRIVE_ROOT / "final").mkdir(exist_ok=True)
        (DRIVE_ROOT / "analysis").mkdir(exist_ok=True)
        (DRIVE_ROOT / "console_logs").mkdir(exist_ok=True)

        os.environ["PYTHONUNBUFFERED"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        print("Pinned release:", RESEARCH_TAG)
        print("Persistent run folder:", DRIVE_ROOT)
        print("ARGUS model:", ARGUS_MODEL)
        print("Neural-HYDRA seeds:", HYDRA_TRAINING_SEEDS)
        """
    ),
    code(
        """
        # Cell 4 — Clone or refresh, then detach at the immutable research tag.
        import os
        import subprocess
        from collections import deque

        def run(command, *, cwd=None, log_path=None):
            command = list(map(str, command))
            print("+", " ".join(command), flush=True)
            log_handle = None
            if log_path is not None:
                log_path = Path(log_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = log_path.open("a", encoding="utf-8", buffering=1)
                log_handle.write("\\n+ " + " ".join(command) + "\\n")
            tail = deque(maxlen=80)
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    tail.append(line.rstrip())
                    if log_handle is not None:
                        log_handle.write(line)
                return_code = process.wait()
            finally:
                if log_handle is not None:
                    log_handle.flush()
                    os.fsync(log_handle.fileno())
                    log_handle.close()
            if return_code != 0:
                recent = "\\n".join(tail) or "(the child process emitted no text)"
                location = f" Persistent log: {log_path}." if log_path else ""
                raise RuntimeError(
                    f"Command failed with exit code {return_code}.{location}\\n"
                    f"Last child-process output:\\n{recent}"
                )
            return return_code

        if not (SOURCE_ROOT / ".git").exists():
            run(["git", "clone", "--filter=blob:none", REPO_URL, SOURCE_ROOT])
        else:
            dirty = subprocess.check_output(
                ["git", "-C", str(SOURCE_ROOT), "status", "--porcelain"],
                text=True,
            ).strip()
            if dirty:
                raise RuntimeError(
                    f"{SOURCE_ROOT} has local edits. Use a new SOURCE_ROOT instead of "
                    "overwriting an experiment checkout."
                )

        run(["git", "-C", SOURCE_ROOT, "fetch", "origin", "--tags", "--force"])
        run(["git", "-C", SOURCE_ROOT, "checkout", "--detach", RESEARCH_TAG])

        SOURCE_COMMIT = subprocess.check_output(
            ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        TAG_COMMIT = subprocess.check_output(
            ["git", "-C", str(SOURCE_ROOT), "rev-list", "-n", "1", RESEARCH_TAG],
            text=True,
        ).strip()
        if SOURCE_COMMIT != TAG_COMMIT:
            raise RuntimeError("Checked-out commit does not match the research tag.")

        print("Frozen source commit:", SOURCE_COMMIT)
        """
    ),
    code(
        """
        # Cell 5 — Install the exact inference/research stack.
        # Colab's CUDA-enabled torch is preserved.
        import subprocess
        import sys

        # The research workflow does not use these optional Colab packages. OpenEnv pulls
        # Gradio, whose Hub 1.x requirement conflicts with Transformers 4.57.6 (<1.0).
        # Removing the unused chain makes the text-only ARGUS environment deterministic.
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "uninstall",
                "-q",
                "-y",
                "peft",
                "trl",
                "torchvision",
                "timm",
                "diffusers",
                "gradio",
                "gradio-client",
                "openenv-core",
                "panopticon-protocol-v3",
            ]
        )
        packages = [
            "transformers==4.57.6",
            "tokenizers==0.22.1",
            "huggingface-hub==0.36.2",
            "accelerate==1.14.0",
            "safetensors==0.8.0",
            "pydantic>=2.6,<3",
            "fastapi==0.115.12",
            "httpx==0.28.1",
            "gymnasium==0.29.1",
            "numpy>=1.26",
            "pandas>=2.2",
            "scipy>=1.12",
            "matplotlib>=3.8",
            "pytest==8.3.5",
        ]
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                *packages,
            ]
        )
        # Dependencies are deliberately installed above. --no-deps prevents the editable
        # project metadata from reintroducing OpenEnv/Gradio and upgrading Hub to 1.x.
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--no-deps",
                "-e",
                str(SOURCE_ROOT),
            ]
        )
        run(
            [
                sys.executable,
                "-c",
                (
                    "import accelerate, importlib.metadata as md, torch, transformers; "
                    "from transformers import AutoModelForCausalLM, AutoTokenizer; "
                    "expected={'transformers':'4.57.6','tokenizers':'0.22.1',"
                    "'huggingface-hub':'0.36.2','accelerate':'1.14.0',"
                    "'safetensors':'0.8.0'}; "
                    "actual={name:md.version(name) for name in expected}; "
                    "assert actual == expected, (actual, expected); "
                    "print('torch', torch.__version__, 'transformers', "
                    "transformers.__version__, 'accelerate', accelerate.__version__, "
                    "'resolved', actual)"
                ),
            ]
        )
        print("Python-3.12-compatible text-inference dependencies installed.")
        """
    ),
    code(
        """
        # Cell 6 — Activate the checkout and validate the research artifact.
        import json
        import os
        import sys

        os.chdir(SOURCE_ROOT)
        if str(SOURCE_ROOT) not in sys.path:
            sys.path.insert(0, str(SOURCE_ROOT))

        required = [
            "train_hydra.py",
            "hydra_neural.py",
            "hydra_policy.py",
            "v6_evaluation.py",
            "research_paper/data/seed_plans/v6_seed_plan.json",
            "research_paper/data/training_seed_ledger.drive_verified.json",
            "research_paper/data/seed_plans/v6_training_separation_report.json",
        ]
        missing = [name for name in required if not (SOURCE_ROOT / name).is_file()]
        if missing:
            raise RuntimeError(f"Research release is incomplete: {missing}")

        separation = json.loads(
            (SOURCE_ROOT / required[-1]).read_text(encoding="utf-8")
        )
        if separation["conclusion"] != "directly-verified-disjoint":
            raise RuntimeError("Training/evaluation seed separation is not directly verified.")

        run([sys.executable, "research_paper/scripts/validate_package.py"], cwd=SOURCE_ROOT)
        run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "tests/test_panopticon_bench.py",
                "tests/test_research_validity.py",
                "tests/test_server_hardening.py",
                "tests/test_packaging_contract.py",
            ],
            cwd=SOURCE_ROOT,
        )
        print("Source and research validation passed.")
        """
    ),
    code(
        """
        # Cell 7 — Validate and fingerprint the existing merged ARGUS model.
        # The first run may take several minutes because large weights are hashed.
        import hashlib
        import json

        if not ARGUS_MODEL.is_dir():
            raise FileNotFoundError(
                f"Merged ARGUS model not found at {ARGUS_MODEL}. Confirm that the correct "
                "Google account is mounted and that panopticon-security-v5-ep50 exists."
            )

        required_model_files = ["config.json"]
        missing = [name for name in required_model_files if not (ARGUS_MODEL / name).is_file()]
        weight_files = sorted(ARGUS_MODEL.glob("*.safetensors")) + sorted(
            ARGUS_MODEL.glob("pytorch_model*.bin")
        )
        if missing or not weight_files:
            raise RuntimeError(
                f"Model directory is incomplete. Missing={missing}, weight_files={weight_files}"
            )

        def file_sha256(path, chunk_size=8 * 1024 * 1024):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(chunk_size), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        model_manifest_path = DRIVE_ROOT / "argus_model_manifest.json"
        inventory = [
            {
                "relative_path": path.relative_to(ARGUS_MODEL).as_posix(),
                "size_bytes": path.stat().st_size,
                "modified_time_ns": path.stat().st_mtime_ns,
            }
            for path in sorted(ARGUS_MODEL.rglob("*"))
            if path.is_file()
        ]

        if model_manifest_path.exists():
            model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
            if model_manifest.get("inventory") != inventory:
                raise RuntimeError(
                    "ARGUS model files changed after the run manifest was created. "
                    "Use a new DRIVE_ROOT rather than mixing checkpoints."
                )
            print("Reusing verified model manifest.")
        else:
            files = []
            for item in inventory:
                path = ARGUS_MODEL / item["relative_path"]
                print("Hashing:", item["relative_path"])
                files.append({**item, "sha256": file_sha256(path)})
            encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
            model_manifest = {
                "schema_version": "panopticon-model-manifest-v1",
                "model_path": str(ARGUS_MODEL),
                "inventory": inventory,
                "files": files,
                "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
            }
            temporary = model_manifest_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(model_manifest, indent=2) + "\\n")
            temporary.replace(model_manifest_path)

        MODEL_MANIFEST_SHA256 = model_manifest["manifest_sha256"]
        print("ARGUS manifest digest:", MODEL_MANIFEST_SHA256)

        # Load the exact model once before any pilot episode. If loading fails, this cell
        # shows the complete traceback before the long evaluator starts.
        from argus_llm import LocalArgusModel

        print("Preflighting merged ARGUS model load...")
        model_probe = LocalArgusModel(str(ARGUS_MODEL))
        try:
            print("ARGUS load preflight passed:", model_probe.model_info())
        finally:
            model_probe.close()
            del model_probe
        """
    ),
    code(
        """
        # Cell 8 — Freeze the run configuration. Existing mismatches stop the run.
        import datetime
        import json

        seed_plan = json.loads(
            (
                SOURCE_ROOT
                / "research_paper/data/seed_plans/v6_seed_plan.json"
            ).read_text(encoding="utf-8")
        )
        frozen_config = {
            "schema_version": "panopticon-colab-v6-run-v1",
            "research_tag": RESEARCH_TAG,
            "source_commit": SOURCE_COMMIT,
            "seed_plan_sha256": seed_plan["seed_plan_sha256"],
            "training_ledger_sha256": (
                "741276b1fcbab159db7fee95d5e418f91a12316f6ca4861479f78af595c4415d"
            ),
            "argus_model_manifest_sha256": MODEL_MANIFEST_SHA256,
            "hydra_training_seeds": HYDRA_TRAINING_SEEDS,
            "hydra_target_episodes": HYDRA_TARGET_EPISODES,
            "hydra_checkpoint_every": HYDRA_CHECKPOINT_EVERY,
            "policies": POLICIES.split(","),
            "max_steps": MAX_STEPS,
        }
        config_path = DRIVE_ROOT / "frozen_run_config.json"
        if config_path.exists():
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            scientific_keys = set(frozen_config) - {"research_tag", "source_commit"}
            scientific_differences = {
                key: {"existing": existing.get(key), "requested": frozen_config.get(key)}
                for key in sorted(scientific_keys)
                if existing.get(key) != frozen_config.get(key)
            }
            if scientific_differences:
                raise RuntimeError(
                    "Frozen scientific configuration mismatch: "
                    + json.dumps(scientific_differences, sort_keys=True)
                )
            source_changed = any(
                existing.get(key) != frozen_config.get(key)
                for key in ("research_tag", "source_commit")
            )
            if source_changed:
                if existing.get("research_tag") not in ALLOWED_PREVIOUS_RESEARCH_TAGS:
                    raise RuntimeError(
                        "Existing source is not in the approved diagnostic-patch lineage. "
                        "Use a new DRIVE_ROOT."
                    )
                patch_record = {
                    "schema_version": "panopticon-runtime-patch-v1",
                    "applied_at_utc": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                    "previous_research_tag": existing.get("research_tag"),
                    "previous_source_commit": existing.get("source_commit"),
                    "new_research_tag": RESEARCH_TAG,
                    "new_source_commit": SOURCE_COMMIT,
                    "reason": (
                        "NumPy scalar serialization, isolated Python 3.12 inference "
                        "dependencies, streamed diagnostics, and failure-manifest "
                        "persistence; no environment or metric change"
                    ),
                }
                history_path = DRIVE_ROOT / "runtime_patch_history.jsonl"
                prior_patch_commits = set()
                if history_path.exists():
                    prior_patch_commits = {
                        json.loads(line)["new_source_commit"]
                        for line in history_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    }
                if SOURCE_COMMIT not in prior_patch_commits:
                    with history_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(patch_record, sort_keys=True) + "\\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                print("Approved runtime-only patch; existing episode records remain valid.")
        else:
            temporary = config_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(frozen_config, indent=2) + "\\n")
            temporary.replace(config_path)

        import accelerate
        import safetensors
        import tokenizers
        import transformers

        runtime_record = {
            "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "gpu": GPU_NAME,
            "gpu_memory_gb": GPU_GB,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "tokenizers": tokenizers.__version__,
            "safetensors": safetensors.__version__,
            "research_tag": RESEARCH_TAG,
            "source_commit": SOURCE_COMMIT,
        }
        with (DRIVE_ROOT / "runtime_sessions.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(runtime_record, sort_keys=True) + "\\n")

        print(config_path.read_text(encoding="utf-8"))
        """
    ),
    code(
        """
        # Cell 9 — Inspect resumable progress and Drive storage.
        import shutil
        import torch

        # Prove that mounted-Drive writes persist independently of the Colab save icon.
        drive_probe = DRIVE_ROOT / ".drive_write_probe"
        drive_probe_tmp = DRIVE_ROOT / ".drive_write_probe.tmp"
        probe_text = f"drive-persistence-ok:{SOURCE_COMMIT}"
        with drive_probe_tmp.open("w", encoding="utf-8") as handle:
            handle.write(probe_text)
            handle.flush()
            os.fsync(handle.fileno())
        drive_probe_tmp.replace(drive_probe)
        if drive_probe.read_text(encoding="utf-8") != probe_text:
            raise RuntimeError("Mounted Drive write/read verification failed.")
        drive_probe.unlink()
        print("Mounted Drive write/read verification passed.")

        usage = shutil.disk_usage(DRIVE_ROOT)
        print(f"Drive free space: {usage.free / 1024**3:.2f} GB")
        if usage.free < 5 * 1024**3:
            raise RuntimeError(
                "Less than 5 GB is free. Free Drive space before model evaluation."
            )

        for seed in HYDRA_TRAINING_SEEDS:
            checkpoint = DRIVE_ROOT / "checkpoints" / f"hydra_seed_{seed}.pt"
            stale_temp = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
            if stale_temp.exists():
                # A temp file means Colab stopped during a save. The old .pt remains durable.
                stale_temp.unlink()
                print("Removed incomplete temporary checkpoint:", stale_temp.name)
            if checkpoint.exists():
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                episode = payload.get("metadata", {}).get("episode", 0)
                print(f"HYDRA seed {seed}: durable episode {episode}/{HYDRA_TARGET_EPISODES}")
            else:
                print(f"HYDRA seed {seed}: not started")

        for condition in ["scripted", *[f"neural_seed_{s}" for s in HYDRA_TRAINING_SEEDS]]:
            manifest = DRIVE_ROOT / "pilot" / condition / "manifest.json"
            if manifest.exists():
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                print(
                    f"Pilot {condition}: {payload.get('status')} "
                    f"{payload.get('completed_episodes', 0)} episodes"
                )
        """
    ),
    markdown(
        """
        ## Pilot phase

        The next cell evaluates the default scripted HYDRA. It saves every completed episode.
        If Colab stops, rerun Cells 1–10 and then rerun the same pilot cell. The `--resume`
        flag skips all durable records and continues with the first unfinished episode.
        """
    ),
    code(
        """
        # Cell 10 — Run or resume the scripted-HYDRA pilot.
        scripted_output = DRIVE_ROOT / "pilot" / "scripted"
        command = [
            sys.executable,
            "v6_evaluation.py",
            "--split",
            "pilot",
            "--policies",
            POLICIES,
            "--model",
            ARGUS_MODEL,
            "--model-label",
            MODEL_LABEL,
            "--output-dir",
            scripted_output,
            "--max-steps",
            MAX_STEPS,
            "--trace-level",
            "compact",
            "--resume",
        ]
        run(
            command,
            cwd=SOURCE_ROOT,
            log_path=DRIVE_ROOT / "console_logs" / "pilot_scripted.log",
        )
        print((scripted_output / "summary.json").read_text(encoding="utf-8")[:4000])
        """
    ),
    markdown(
        """
        ## Neural-HYDRA training

        Cell 11 trains five independent neural policies against a frozen population of random,
        heuristic, and security-first ARGUS defenders. Environment seeds come **only** from the
        development split. Pilot and final seeds are never used for training.

        Checkpoints are tiny compared with ARGUS. Each seed owns exactly one `.pt` file. A new
        checkpoint is written to `.pt.tmp` and atomically replaces `.pt`; therefore old
        checkpoints never accumulate. The training log is append-only and aligned with the last
        durable checkpoint after every resume.

        If the runtime disconnects, rerun from Cell 1 and execute Cell 11 again. Completed seeds
        immediately report “nothing to do”; an incomplete seed resumes from its exact PyTorch RNG
        state and episode number.
        """
    ),
    code(
        """
        # Cell 11 — Train or exactly resume all five neural-HYDRA initializations.
        for seed in HYDRA_TRAINING_SEEDS:
            checkpoint = DRIVE_ROOT / "checkpoints" / f"hydra_seed_{seed}.pt"
            training_log = DRIVE_ROOT / "training_logs" / f"hydra_seed_{seed}.jsonl"
            command = [
                sys.executable,
                "train_hydra.py",
                "--episodes",
                HYDRA_TARGET_EPISODES,
                "--seed",
                seed,
                "--seed-plan",
                SOURCE_ROOT / "research_paper/data/seed_plans/v6_seed_plan.json",
                "--checkpoint-every",
                HYDRA_CHECKPOINT_EVERY,
                "--output",
                checkpoint,
                "--training-log",
                training_log,
            ]
            if checkpoint.exists():
                command.extend(["--resume", checkpoint])
            print("\\n", "=" * 78)
            print(f"Training/resuming neural HYDRA seed {seed}")
            run(
                command,
                cwd=SOURCE_ROOT,
                log_path=DRIVE_ROOT / "console_logs" / f"train_hydra_{seed}.log",
            )

        print("All requested neural-HYDRA checkpoints reached the target.")
        """
    ),
    code(
        """
        # Cell 12 — Inspect neural-HYDRA training curves and checkpoint metadata.
        import json
        import matplotlib.pyplot as plt
        import pandas as pd
        import torch

        training_frames = []
        for seed in HYDRA_TRAINING_SEEDS:
            checkpoint = DRIVE_ROOT / "checkpoints" / f"hydra_seed_{seed}.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            metadata = payload["metadata"]
            print(
                f"seed={seed} episode={metadata['episode']} "
                f"recent_mean={metadata['recent_mean_score']:.4f} "
                f"parameters={payload['parameter_count']}"
            )
            log_path = DRIVE_ROOT / "training_logs" / f"hydra_seed_{seed}.jsonl"
            records = [json.loads(line) for line in log_path.read_text().splitlines() if line]
            frame = pd.DataFrame(records)
            frame["training_seed"] = seed
            frame["rolling_score_100"] = frame["hydra_score"].rolling(100, min_periods=1).mean()
            training_frames.append(frame)

        training_data = pd.concat(training_frames, ignore_index=True)
        training_data.to_csv(
            DRIVE_ROOT / "analysis" / "hydra_training_all_seeds.csv",
            index=False,
        )

        plt.figure(figsize=(11, 6))
        for seed, frame in training_data.groupby("training_seed"):
            plt.plot(frame["episode"], frame["rolling_score_100"], label=str(seed))
        plt.xlabel("Training episode")
        plt.ylabel("HYDRA adversary score (100-episode rolling mean)")
        plt.title("Neural-HYDRA training trajectories")
        plt.legend(title="Initialization seed", fontsize=8)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plot_path = DRIVE_ROOT / "analysis" / "hydra_training_curves.png"
        plt.savefig(plot_path, dpi=180)
        plt.show()
        print("Saved:", plot_path)
        """
    ),
    code(
        """
        # Cell 13 — Evaluate or resume every neural HYDRA on the unseen pilot split.
        for seed in HYDRA_TRAINING_SEEDS:
            checkpoint = DRIVE_ROOT / "checkpoints" / f"hydra_seed_{seed}.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(f"Missing trained checkpoint: {checkpoint}")
            output = DRIVE_ROOT / "pilot" / f"neural_seed_{seed}"
            command = [
                sys.executable,
                "v6_evaluation.py",
                "--split",
                "pilot",
                "--policies",
                POLICIES,
                "--model",
                ARGUS_MODEL,
                "--model-label",
                MODEL_LABEL,
                "--hydra-checkpoint",
                checkpoint,
                "--output-dir",
                output,
                "--max-steps",
                MAX_STEPS,
                "--trace-level",
                "compact",
                "--resume",
            ]
            print("\\n", "=" * 78)
            print(f"Evaluating neural HYDRA seed {seed}")
            run(
                command,
                cwd=SOURCE_ROOT,
                log_path=DRIVE_ROOT / "console_logs" / f"pilot_neural_{seed}.log",
            )

        print("All neural-HYDRA pilot evaluations are complete.")
        """
    ),
    code(
        """
        # Cell 14 — Build episode-level pilot tables and paired comparisons.
        import json
        import math
        import pandas as pd
        from scipy import stats

        conditions = [("scripted", None)] + [
            (f"neural_seed_{seed}", seed) for seed in HYDRA_TRAINING_SEEDS
        ]
        rows = []
        for condition, hydra_seed in conditions:
            path = DRIVE_ROOT / "pilot" / condition / "episodes.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"Missing pilot records: {path}")
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                record = json.loads(line)
                episode = record["episode"]
                final_state = episode["final_state"]
                provenance = episode["provenance_summary"]
                rows.append(
                    {
                        "condition": condition,
                        "hydra_kind": "scripted" if hydra_seed is None else "neural",
                        "hydra_training_seed": hydra_seed,
                        "policy": record["policy_stratum"],
                        "level": record["level"],
                        "environment_seed": record["seed"],
                        "grade": episode["grade"]["score"],
                        "passed": bool(episode["grade"]["passed"]),
                        "reward": episode["total_reward"],
                        "security": final_state["security_score"],
                        "revenue": final_state["enterprise_revenue"],
                        "sleepers_caught": final_state["sleepers_caught"],
                        "sleepers_missed": final_state["sleepers_missed"],
                        "false_accusations": final_state["false_accusations"],
                        "interventions": provenance["interventions"],
                        "turns": provenance["turns"],
                    }
                )

        pilot = pd.DataFrame(rows)
        pilot_path = DRIVE_ROOT / "analysis" / "pilot_episode_metrics.csv"
        pilot.to_csv(pilot_path, index=False)

        def summarize_group(frame):
            n = len(frame)
            grade_mean = frame["grade"].mean()
            grade_sem = frame["grade"].sem() if n > 1 else 0.0
            critical = stats.t.ppf(0.975, n - 1) if n > 1 else 0.0
            return pd.Series(
                {
                    "episodes": n,
                    "grade_mean": grade_mean,
                    "grade_ci95_low": grade_mean - critical * grade_sem,
                    "grade_ci95_high": grade_mean + critical * grade_sem,
                    "security_mean": frame["security"].mean(),
                    "pass_rate": frame["passed"].mean(),
                    "sleepers_missed_mean": frame["sleepers_missed"].mean(),
                    "intervention_rate": (
                        frame["interventions"].sum() / max(frame["turns"].sum(), 1)
                    ),
                }
            )

        group_summary = (
            pilot.groupby(["condition", "policy", "level"], dropna=False)
            .apply(summarize_group, include_groups=False)
            .reset_index()
        )
        group_path = DRIVE_ROOT / "analysis" / "pilot_group_summary.csv"
        group_summary.to_csv(group_path, index=False)

        scripted = pilot[pilot["condition"] == "scripted"][
            ["policy", "level", "environment_seed", "grade", "security"]
        ].rename(columns={"grade": "grade_scripted", "security": "security_scripted"})
        neural = pilot[pilot["hydra_kind"] == "neural"]
        paired = neural.merge(
            scripted,
            on=["policy", "level", "environment_seed"],
            how="inner",
            validate="many_to_one",
        )
        paired["grade_delta_neural_minus_scripted"] = (
            paired["grade"] - paired["grade_scripted"]
        )
        paired["security_delta_neural_minus_scripted"] = (
            paired["security"] - paired["security_scripted"]
        )
        paired_path = DRIVE_ROOT / "analysis" / "pilot_paired_differences.csv"
        paired.to_csv(paired_path, index=False)

        display(group_summary)
        print("Saved:", pilot_path)
        print("Saved:", group_path)
        print("Saved:", paired_path)
        print(
            "Pilot results are descriptive and must not be promoted to final efficacy claims."
        )
        """
    ),
    code(
        """
        # Cell 15 — Enforce pilot completion before exposing the final run.
        expected_per_condition = len(POLICIES.split(",")) * 5 * 5
        incomplete = []
        for condition, _ in conditions:
            summary_path = DRIVE_ROOT / "pilot" / condition / "summary.json"
            if not summary_path.exists():
                incomplete.append((condition, "missing summary"))
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not summary.get("complete") or summary.get("completed_episodes") != expected_per_condition:
                incomplete.append(
                    (
                        condition,
                        f"{summary.get('completed_episodes')}/{expected_per_condition}",
                    )
                )

        if incomplete:
            raise RuntimeError(f"Pilot is incomplete: {incomplete}")

        PILOT_COMPLETE = True
        print("Pilot completion gate passed for scripted + five neural conditions.")
        print("Review the CSV files under:", DRIVE_ROOT / "analysis")
        print("Freeze any analysis/model-selection decisions before running final.")
        """
    ),
    markdown(
        """
        ## Final split — intentionally locked

        The final split contains 200 seeds per level. Across five ARGUS policies, scripted HYDRA,
        and five neural-HYDRA initializations, the full matrix is **30,000 episodes** and may take
        multiple Colab sessions. Every completed episode is resume-safe.

        Before unlocking:

        1. confirm Cell 15 passed;
        2. inspect the pilot tables for software/data errors;
        3. do not change model checkpoints, thresholds, policies, or metrics afterward;
        4. make sure Drive has adequate free space; and
        5. set the two values in Cell 16 exactly as instructed.

        Final outcomes must never be used to select a checkpoint or tune the system.
        """
    ),
    code(
        """
        # Cell 16 — Guarded final evaluation. Leave locked until the protocol is frozen.
        RUN_FINAL = False
        FINAL_AUTHORIZATION_PHRASE = ""

        required_phrase = "I_UNDERSTAND_FINAL_SPLIT_IS_SINGLE_USE"
        if not RUN_FINAL:
            print("Final split remains locked. This is the correct default.")
        else:
            if not globals().get("PILOT_COMPLETE", False):
                raise RuntimeError("Pilot completion gate has not passed in this session.")
            if FINAL_AUTHORIZATION_PHRASE != required_phrase:
                raise RuntimeError(
                    f"Set FINAL_AUTHORIZATION_PHRASE exactly to {required_phrase!r} "
                    "only after freezing the protocol."
                )

            final_conditions = [("scripted", None)] + [
                (f"neural_seed_{seed}", seed) for seed in HYDRA_TRAINING_SEEDS
            ]
            for condition, hydra_seed in final_conditions:
                output = DRIVE_ROOT / "final" / condition
                command = [
                    sys.executable,
                    "v6_evaluation.py",
                    "--split",
                    "final",
                    "--policies",
                    POLICIES,
                    "--model",
                    ARGUS_MODEL,
                    "--model-label",
                    MODEL_LABEL,
                    "--output-dir",
                    output,
                    "--max-steps",
                    MAX_STEPS,
                    "--trace-level",
                    "compact",
                    "--resume",
                ]
                if hydra_seed is not None:
                    checkpoint = (
                        DRIVE_ROOT / "checkpoints" / f"hydra_seed_{hydra_seed}.pt"
                    )
                    command.extend(["--hydra-checkpoint", checkpoint])
                print("\\n", "=" * 78)
                print("Final condition:", condition)
                run(
                    command,
                    cwd=SOURCE_ROOT,
                    log_path=DRIVE_ROOT / "console_logs" / f"final_{condition}.log",
                )

            print("All final conditions completed.")
        """
    ),
    code(
        """
        # Cell 17 — Create a compact integrity inventory without duplicating large traces.
        import hashlib
        import json

        important_patterns = [
            "frozen_run_config.json",
            "argus_model_manifest.json",
            "runtime_patch_history.jsonl",
            "console_logs/*.log",
            "checkpoints/*.pt",
            "training_logs/*.jsonl",
            "pilot/*/manifest.json",
            "pilot/*/summary.json",
            "final/*/manifest.json",
            "final/*/summary.json",
            "analysis/*.csv",
            "analysis/*.png",
        ]
        important_files = []
        for pattern in important_patterns:
            important_files.extend(DRIVE_ROOT.glob(pattern))
        important_files = sorted({path for path in important_files if path.is_file()})

        def sha256_file(path):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        inventory = {
            "schema_version": "panopticon-v6-result-inventory-v1",
            "source_commit": SOURCE_COMMIT,
            "seed_plan_sha256": seed_plan["seed_plan_sha256"],
            "files": [
                {
                    "path": path.relative_to(DRIVE_ROOT).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in important_files
            ],
        }
        inventory_path = DRIVE_ROOT / "result_inventory.json"
        temporary = inventory_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(inventory, indent=2) + "\\n")
        temporary.replace(inventory_path)
        print("Wrote:", inventory_path)
        print("Files inventoried:", len(inventory["files"]))
        """
    ),
    markdown(
        """
        ## Exactly how to resume after Colab disconnects

        1. Reopen this same notebook.
        2. Select a GPU runtime again.
        3. Run Cells 1 through 9 in order.
        4. Rerun whichever long cell was interrupted:
           - Cell 10 for scripted pilot;
           - Cell 11 for neural-HYDRA training;
           - Cell 13 for neural pilot evaluation; or
           - Cell 16 for final evaluation after it was legitimately unlocked.
        5. Do **not** rename `DRIVE_ROOT`, checkpoints, output folders, or the research tag.

        Training resumes from the rolling `.pt` checkpoint and restored RNG state. Evaluation
        reads `episodes.jsonl`, verifies every record digest/config hash, skips completed episode
        keys, and continues. At most the single episode executing during a sudden disconnect must
        run again; no completed checkpoint or episode is lost.

        ## What not to delete

        Keep `frozen_run_config.json`, `argus_model_manifest.json`, `checkpoints/`,
        `training_logs/`, `pilot/`, and `final/`. Old neural checkpoints do not accumulate—the
        single file per seed is atomically replaced. Temporary `.pt.tmp` files from interrupted
        writes are safely removed by Cell 9 after confirming the durable `.pt` checkpoint.
        """
    ),
]


NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    OUTPUT.write_text(json.dumps(NOTEBOOK, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT} with {len(CELLS)} cells")


if __name__ == "__main__":
    main()
