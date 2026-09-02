# Panopticon model-selection protocol

V5 is a reproducible fixed-method baseline, not evidence that its hyperparameters
are optimal. No expensive selection or training was run in this infrastructure
PR. The preregistered design is machine-readable in
`training_specs/model_selection_v1.json` and is checked by
`python tools/validate_model_selection.py`.

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
Eligible candidates are ranked by the exact, unrounded minimum level grade, macro
grade, lower training-token count, then candidate ID. Every candidate/seed result,
artifact identity, exclusion, and survivor decision must be retained.

Only the disjoint training and development seed namespaces may be used. Canonical
and confirmation are sealed and cannot be used for early stopping, selection,
reranking, troubleshooting, or retries. The code rejects those splits while the
active spec is provisional.

## Freezing the winner

After all registered development results exist, create (do not silently mutate)
`training_specs/security_first_v6_selected.json`. It must record the winning
candidate and full selection evidence, pass review, be committed with clean source,
and use `status: frozen-selected-canonical`. Only then can the standard runner
train the final frozen experiment or open canonical/confirmation evaluation.
Held-out failure is a negative result, never a reason to select another candidate.
