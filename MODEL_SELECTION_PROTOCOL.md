# Panopticon model-selection protocol

V5 is a reproducible fixed-method baseline, not evidence that its hyperparameters
are optimal. No expensive selection or training was run in this infrastructure
PR. The preregistered design is machine-readable in
`training_specs/model_selection_v1.json` and is checked by
`python tools/validate_model_selection.py`.

`tools/run_model_selection.py` is the only campaign entry point. It refuses real
execution while the committed protocol remains
`preregistered-design-compute-not-authorized`. Its orchestration can be exercised
without a model or GPU, and the output is permanently labelled non-evidence:

```bash
python tools/run_model_selection.py \
  --campaign-dir /tmp/panopticon-selection-fixture \
  --synthetic-fixture
```

After a future reviewed commit changes the protocol to
`preregistered-development-compute-authorized`, rerun the same command without
`--synthetic-fixture`. Contributors do not enter candidates, seeds, budgets,
paths within the campaign, evaluator settings, or survivor choices.

## Development-only selection

The protocol searches exactly eight candidates: two learning rates, two coupled
LoRA rank/alpha choices, and two epoch counts. Dropout, model/revision, expert,
trajectory schema, completion-only loss, optimizer, schedule, precision, sequence
length, batching, and all other training-critical choices stay fixed. Successive
halving evaluates 8→3→2→1 candidates with increasing registered training and
development budgets and optimization seeds.

A candidate is ineligible before ranking unless the security regression suite
passes, the complete development matrix exists, no model turn was truncated,
every level has 100% pass rate, and there are zero sleepers missed and zero false
accusations. Ineligible candidates cannot win through a high aggregate score.
Every registered optimization-seed run must independently pass every gate. The
minimum-level metric is the worst level across every registered seed, while the
macro metric is averaged across all registered seeds. Missing or failed seeds make
the candidate ineligible; an individually favorable seed can never be selected.
All candidates within a round use the identical development episode plan.
Eligible candidates are ranked by the exact, unrounded minimum level grade, macro
grade lower bound from a fixed 5,000-sample stratified paired bootstrap, macro
grade, lower registered episode×epoch budget, then candidate ID. Every candidate/seed result,
artifact identity, exclusion, and survivor decision must be retained.

Only the disjoint training and development seed namespaces may be used. Canonical
and confirmation are sealed and cannot be used for early stopping, selection,
reranking, troubleshooting, or retries. The code rejects those splits while the
active spec is provisional.

## Freezing the winner

The orchestrator emits `security_first_v6_selected.json` with
`status: proposed-selected-review-required`; it cannot authorize held-out access.
The final selected configuration uses the separately preregistered optimization
seed 7200 (level seeds 7201–7205), rather than choosing a favorable search seed.
The proposal must record the full selection evidence, pass review, be moved into
`training_specs/`, be committed with clean source, and only then be changed to
`status: frozen-selected-canonical`. Only then can the standard runner
train the final frozen experiment or open canonical/confirmation evaluation.
Held-out failure is a negative result, never a reason to select another candidate.
