# Research Submission Checklist

## Scientific claim integrity

- [ ] Title and abstract match completed evidence, not planned experiments.
- [ ] Current and older evaluation snapshots are not mixed.
- [ ] Every number is generated or linked to an immutable source.
- [ ] Sample size and uncertainty accompany empirical claims.
- [ ] Negative/null results and all registered ablations are reported.
- [ ] Raw, syntax-cleaned, repaired, supervisor, and hybrid systems are distinct.
- [ ] Supervisor results are not attributed to raw V5.
- [ ] “Passed sampled gates” is not replaced by “proved safe.”
- [ ] No “first” or priority claim survives without a current literature check.

## Method and reproducibility

- [ ] Source commit and dirty status frozen.
- [ ] Model/tokenizer/adapter/config hashes frozen.
- [x] Reconstructed V5 seeds match all 250 ordered seeds in the five original Drive-side expert-metrics files.
- [x] Frozen V6 seeds have zero overlap with the directly verified V5 training ledger.
- [ ] Reward, grader, trajectory, prompt, and event schemas are versioned.
- [ ] Per-episode logs and analysis code archived.
- [ ] Hardware/software/compute details included.
- [ ] Environment and benchmark tests pass from a clean setup.
- [ ] Another person completed an independent reproduction slice.

## Evaluation quality

- [ ] Final paired sample size justified before outcome inspection.
- [ ] Exact intervals reported for advanced pass/failure rates.
- [ ] Timeouts/crashes/malformed outputs remain in denominators.
- [ ] Utility preservation and false-positive behavior reported.
- [ ] CMG and intervention dependence reported with definitions.
- [ ] At least the core ablations are complete.
- [ ] OOD/shifted scenarios are reported separately.

## Ethics, safety, and release

- [ ] Ethics/dual-use statement matches released artifacts.
- [ ] Synthetic versus real data is explicit.
- [ ] No secrets, personal data, credentials, or unauthorized traces are present.
- [ ] Base model, code, data, and figure licenses checked.
- [ ] External-target testing had written authorization.
- [ ] Taxonomy mappings are not described as certifications.

## Authorship and administration

- [ ] Final author order, spelling, affiliations, emails, and ORCIDs confirmed.
- [ ] CRediT contributions approved by all authors.
- [ ] Funding, compute credits, acknowledgements, and conflicts resolved.
- [ ] Venue/track and page limits confirmed from current official instructions.
- [ ] Anonymous-review rules applied to manuscript, code, metadata, and links.
- [ ] Prior/preprint/workshop disclosure completed.
- [ ] Cover letter and suggested reviewers prepared if required.

## Files and presentation

- [ ] PDF builds without errors/warnings that affect content.
- [ ] Fonts embedded and figures legible in grayscale/color-blind conditions.
- [ ] Tables fit margins and define abbreviations.
- [ ] References compile; DOIs/URLs and citation metadata verified.
- [ ] Supplement and artifact links work from a clean browser/session.
- [ ] Final PDF/source/artifact hashes recorded.
