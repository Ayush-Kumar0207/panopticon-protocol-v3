# Contributor response templates

These are maintainer aids, not automatic replies. Re-read the person's message,
verify the current repository state, personalize the opening, and send only what
is relevant. Never promise payment, employment, publication, or authorship.

## Interested contributor

> Thank you for your interest in Panopticon. The currently authorized work is the
> frozen development/model-selection campaign; final refit and held-out evaluation
> remain sealed. Please read the [onboarding guide](CONTRIBUTOR_ONBOARDING.md) and
> tell me which track interests you. Before any substantial compute, please add
> the non-sensitive preflight details requested in [issue #1](https://github.com/Ayush-Kumar0207/panopticon-protocol-v3/issues/1)
> so we can avoid duplicate work. Please do not post credentials, private paths,
> phone numbers, email addresses, or account identifiers.

## “What task should I take?”

> A good first step is Track A: install the exact environment and return the
> preflight report and real BF16 LoRA probe. If your machine qualifies and you
> have stable persistent storage, we can then acknowledge Track B, the complete
> 8→3→2→1 development campaign. If you prefer not to train, Track C is an
> independent artifact review. Which of those best matches your time and skills?

## Limited compute

> Thanks for being clear about the limit. Panopticon training requires an NVIDIA
> Ampere-or-newer GPU with native BF16 and at least 14 GiB VRAM, so we should not
> force the frozen run onto incompatible hardware or change precision. You can
> still help through code/test review, documentation, methodology, or independent
> verification of non-executable evidence. If you share only your GPU/VRAM, CPU,
> RAM, OS, and approximate availability, I can suggest a bounded track.

## Substantial compatible GPU compute

> That environment may be suitable. Before committing expensive compute, please
> post the GPU/VRAM, OS, Python version, stability/interruption plan, persistent
> storage, artifact host, and intended commit in issue #1. After the claim is
> acknowledged, start with the exact installer and preflight in the onboarding
> guide. Stop on any fail-closed error; do not modify precision, dependencies,
> candidates, seeds, budgets, or gates. A passing preflight qualifies the machine;
> it does not authorize final refit or held-out evaluation.

## Failed or interrupted run

> Thank you for reporting it—please preserve the directory exactly as it is.
> Do not delete failed records, edit locks/state, or start a favorable replacement.
> Share the exact command, commit, last successful stage, error text, relevant
> environment report, interruption timeline, and a hash manifest, with secrets and
> private paths removed. If the runner identifies the state as compatible, resume
> by rerunning the identical campaign command; otherwise stop for review. A
> well-preserved negative or incomplete run is still valuable evidence.

## Completed results

> Thank you. Before describing this as successful, please provide the full
> unedited campaign directory through a durable artifact link, its archive hash,
> a recursive SHA-256 manifest, the exact commit and command, and a disclosure of
> interruptions or deviations. Open a small PR linking those materials and issue
> #1; do not commit weights or raw large artifacts. We will check identity,
> completeness, all registered seeds, gates, raw-to-summary agreement, decisions,
> and absence of held-out access. “Campaign complete” and “development winner
> proposed” do not yet mean “scientifically accepted policy.”

## Paper or authorship question

> There is research-paper work in progress, but publication depends on obtaining
> defensible reproducible evidence. Accepted execution or verification work
> receives public technical attribution. Authorship is considered only for a
> substantial scholarly contribution—such as methodology, analysis,
> interpretation, or writing—under the applicable venue's authorship standards;
> compute alone does not guarantee authorship.

## Maintainer acknowledgement of a run claim

> Acknowledged for **[track and bounded scope]** against commit **[commit]**.
> Please use a new persistent directory and only the exact command in the
> onboarding guide. Preserve all failures and interruptions, keep held-out splits
> sealed, and stop if any identity or preflight check fails. This acknowledgement
> covers only the stated development work and expires if the commit, protocol, or
> hardware/runtime changes; report a change before continuing.
