# TotalSegmentator v3 adrenal CT: source-label quality review rubric

**Before looking at predictions**, review source-only montages from
`scripts/run_totalseg_v3_adrenal_qc.sbatch` and the corresponding CT/mask NIfTIs
using suitable 3D imaging software if the 2D montage is uncertain.

The plan draws 60 images from the official train partition and 15 from the official
test partition using only metadata, age, institution, scanner model, acquisition
thickness, study type and gland-mask volumes. Fixed seed 20261110. Selected IDs
and metadata live in local files:
- `runs/totalsegmentator_v3_qc_plan/training_label_review_sample.csv`
- `runs/totalsegmentator_v3_qc_plan/external_test_label_review_sample.csv`
- `runs/totalsegmentator_v3_qc_plan/training_bilateral_inventory.csv` (all 1053 candidate train)
- `runs/totalsegmentator_v3_qc_plan/external_test_bilateral_inventory.csv` (all 76 official test)
- `runs/totalsegmentator_v3_qc_plan/plan_summary.json` (metadata/audit SHA-256 and sampling method).

**Editing rule:** keep the original selection IDs and base columns unchanged.
Make a timestamped copy of the sample CSV before filling reviewer fields.
Record reviewer, review date (ISO), clinical/anatomical judgement and reason;
preserve prior copies when adjudication changes.

## Human-review checklist

1. **Presence/coverage**: both adrenal glands visible within CT FoV and not truncated at image edge; if uncertain use orthogonal slices and source NIfTI.
2. **Laterality**: source left and right masks anatomically consistent with verified DICOM/NIfTI orientation and midline/adjacent anatomy; do not infer solely from the sign of a scanner x coordinate.
3. **Completeness**: gland masks follow true adrenal structures, not only a small fragment; no visible obvious unlabelled gland subregion.
4. **Boundary precision**: no major missed gland/oversegmentation into kidney, vessels, liver, fat or adjacent organs; difficult tumor/pathology recorded.
5. **Geometry**: mask and CT aligned, plausible mm spacing, non-overlapping left/right binary masks, no obvious deformation artifacts.
6. **Acquisition domain**: original slice thickness from `meta.csv` vs distributed resampled 1.5mm grid; record pediatric/adult status and pathology only as necessary in protected local metadata.
7. **Uncertain cases**: flag for qualified reviewer adjudication; do not assume empty annotations are true negatives.

Suggested local CSV fields:
- `review_status`: `ACCEPT`, `QUERY`, `REJECT` (original value `UNREVIEWED`).
- `laterality_check`, `boundary_check`, `complete_gland_check`: `PASS`, `FAIL`, `UNCERTAIN`.
- `review_reason`: brief specific evidence (e.g., `gland truncated at superior CT edge`, `laterality unclear from affine`).
- `reviewer`, `review_date`: reviewer ID and YYYY-MM-DD.

**Important:** sample-based inspection does NOT validate all remaining 993 unreviewed training scans or all 61 unreviewed external test cases. Quantify sampled acceptance rate and sampling limitations. All 76 external cases should remain in the frozen benchmark by default. Any pre-inference exclusion must have a documented label-only reason, not low model Dice, and will change the frozen expected cohort count, requiring a new protocol and code version rather than silently dropping scans.

Do not view model predictions while adjudicating a source-label quality issue. If a later evaluation reveals poor Dice, investigate it as a reported failure; do not retrospectively exclude it.

## Operational review procedure and external-study sign-off

**Do not alter the frozen sample lists.** Create review copies in the existing
Kelvin2 project workspace, then edit only their status, check, reviewer, date
and reason columns while inspecting the corresponding PNG/CT NIfTI:

```bash
cd /mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation
mkdir -p runs/totalsegmentator_v3_qc_plan/reviews
cp -n runs/totalsegmentator_v3_qc_plan/external_test_label_review_sample.csv \
  runs/totalsegmentator_v3_qc_plan/reviews/external_test_review_completed.csv
cp -n runs/totalsegmentator_v3_qc_plan/training_label_review_sample.csv \
  runs/totalsegmentator_v3_qc_plan/reviews/training_review_completed.csv
find runs/totalsegmentator_v3_qc_plan/montages_external_test \
  -maxdepth 1 -name '*_reference_qc.png' | wc -l
find runs/totalsegmentator_v3_qc_plan/montages_train \
  -maxdepth 1 -name '*_reference_qc.png' | wc -l
```

Review each source-only montage; where necessary open the associated
CT/mask NIfTIs in a medical-image viewer for multi-plane evaluation.
The training sample is needed to assess whether to launch enlarged-data
training. The external-test sample is needed to establish *exploratory*
external-test eligibility. An appropriately experienced human reviewer
must populate the completed review CSVs; automation cannot infer genuine
clinical label quality.

Before exploratory external test evaluation, create a human-signed local
`runs/totalsegmentator_v3_qc_plan/reviews/reviewer_attestation.txt`
stating the source-only review, reviewer ID/date, source provenance judgement,
and explicit limitations: 61 unreviewed external cases and the fact that
image similarity does not prove patient independence. Use no patient
identifiers in this note.

`scripts/validate_totalseg_v3_qc_review.py` independently checks that:
- all **15 sampled external-test** cases have `review_status=ACCEPT`,
  all three anatomical checks `PASS`, an identified reviewer and ISO date;
- none of the original metadata fields or selected cases were modified;
- all 15 are members of the **frozen official 76-case external test** cohort;
- the review, manifest and human attestation have unchanged SHA-256 hashes.

It writes a non-overwriting local
`runs/totalsegmentator_v3_qc_plan/reviews/external_review_gate.json`.
If any case is `QUERY`, `REJECT`, or not yet reviewed, the test is blocked
pending **manual adjudication**—do not falsely mark a defective label as
`ACCEPT`. Preserve rejected cases and reasons; decide and preregister a
new study protocol rather than retrospectively omitting data.

```bash
python -u scripts/validate_totalseg_v3_qc_review.py
# Then, and only after a valid human signoff:
sbatch scripts/evaluate_totalseg_external_adrenal.sbatch
```

The GPU script reruns the gate verification and blocks if the signed
files change. It no longer accepts an environment flag as a substitute
for review. **This does NOT clinically approve all 76 labels or prove
cohort independence**. The ensuing evaluation is still labelled
exploratory, and no tuning on external-test performance is permitted.

Before any external CNN inference:
1. Save reviewer judgement and any second review/adjudication in a dated local signed record.
2. Verify the integrity, source provenance, official split and complete frozen 76-case test manifest.
3. Declare any unavoidable label-quality uncertainty in the manuscript and decide whether results should be labelled exploratory.
4. Authorise `scripts/evaluate_totalseg_external_adrenal.sbatch` using the signed, checksum-validated local review gate after completing the above. No network upload of CT/PACS screenshots/identity-bearing clinical metadata.

All external GT exists for scoring **after** the predictions are frozen; no label-derived ROI crop or direct GT guidance is permitted for model inference.
