# Historical Colab/T4 workflow — non-canonical

The former cell-by-cell T4 instructions have been retired. They combined two
unsafe assumptions:

- T4/Turing was treated as suitable for the canonical BF16 experiment even
  though the frozen profile requires native BF16 on Ampere-or-newer hardware.
- contributors were told to preserve Colab's preinstalled PyTorch, so the actual
  framework/CUDA build could drift between sessions.

The historical FP16 path also has confirmed loss-underflow evidence. Therefore
this file is **not an executable training runbook**, and no T4, FP16, FP32, or
unlocked-Colab result may use the `security-first-v5` canonical identity.

Use [`TRAINING_CONTRIBUTION_GUIDE.md`](TRAINING_CONTRIBUTION_GUIDE.md) instead.
Its version-controlled installer selects `torch==2.2.1+cu121` from PyTorch's
official CUDA 12.1 index, and preflight requires Python 3.11, CUDA, native BF16,
compute capability 8.0 or newer, sufficient VRAM, clean source, exact packages,
and a real assistant-only LoRA optimizer step before expert generation.

Google Colab is usable only if the allocated runtime independently satisfies
that exact profile. A free-tier T4 allocation will fail closed with an explicit
message before training. Runtime availability and product-tier labels are not
scientific compatibility evidence.

Candidate low-cost alternatives—FP32 on T4 or stabilized/scaled FP16—must be
specified and numerically validated under new preregistered experiment IDs.
They must not be introduced as an automatic fallback because doing so would
change the optimization path and repeat a known numerical-risk category.
