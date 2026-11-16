#!/bin/bash -l
#SBATCH --job-name=adrenal_evidence_archive
#SBATCH --partition=k2-medpri
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=adrenal_evidence_archive.%j.out
#SBATCH --error=adrenal_evidence_archive.%j.err

# Optional: create a SELF-CONTAINED archive of trained model files, prediction
# masks and source metric traces for institutional backup. No source patient CT
# or downloaded dataset is included. DO NOT publish this tar/contents on public
# GitHub or transfer to an unapproved service. Copy the tar+SHA256 to approved
# independent institutional storage after success; same-HPC copies are NOT
# disaster recovery backups.
set -Eeuo pipefail
ROOT="${ADRENAL_PROJECT_ROOT:-/mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/adrenal_segmentation}"
cd "$ROOT"
module load apps/python3/3.12.4/gcc-14.1.0
module load compilers/gcc/14.1.0
module load libs/gcc/14.1.0
source .venv/bin/activate

mkdir -p runs/research_evidence_archives
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SNAPSHOT="runs/research_evidence_snapshots/$STAMP"
python -u scripts/snapshot_adrenal_research_evidence.py \
  --out "$SNAPSHOT" --hash-checkpoint --hash-nifti

ARCHIVE="runs/research_evidence_archives/adrenaspike3d_evidence_$STAMP.tar"
LIST="runs/research_evidence_archives/adrenaspike3d_evidence_$STAMP.files.txt"

# Only frozen outputs, checkpoints, logs and research notes; never patient CT,
# source TotalSegmentator dataset or scratch predictions from energy trials.
cat > "$LIST" <<'EOF'
docs/research_evidence_register_2026-11-10.md
docs/adrenaspike3d_prospective_experiment_protocol_2026-11-10.md
scripts/prepare_nnunet_adrenal.py
scripts/evaluate_nnunet_adrenal.py
scripts/postprocess_nnunet_adrenal.py
scripts/audit_nnunet_adrenal_failures.py
scripts/measure_gpu_energy.py
scripts/run_nnunet_energy_repro.sbatch
scripts/run_nnunet_energy_suite.py
scripts/audit_totalsegmentator_v3.py
scripts/screen_amos_totalseg_ct_overlap.py
EOF
printf '%s\n' "$SNAPSHOT" >> "$LIST"
cat >> "$LIST" <<'EOF'
nnunet_results/Dataset701_AMOSAdrenalCT/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0
nnunet_raw/Dataset701_AMOSAdrenalCT/split_manifest.json
nnunet_raw/Dataset701_AMOSAdrenalCT/dataset.json
runs/nnunet3d_holdout_predictions_fold0
runs/nnunet3d_holdout_predictions_cc_largest
runs/nnunet3d_holdout_predictions_cc_relative15
runs/nnunet3d_holdout_metrics_fold0.csv
runs/nnunet3d_holdout_metrics_cc_largest.csv
runs/nnunet3d_holdout_metrics_cc_relative15.csv
runs/energy_nnunet_fullvolume_repro
runs/totalsegmentator_v3_audit
runs/ct_overlap_screen
EOF

# Optional completed external and later expanded-training evidence.
# These sources may not exist in an early archive. Include original score
# outputs and prediction masks without embedding any original patient CTs.
for path in \
  docs/START_HERE_adrenaspike3d_research.md \
  scripts/repair_totalseg_external_ct_geometry.py \
  scripts/report_external_tsv3_unreviewed.py \
  scripts/freeze_expanded_amos_tsv3_manifest.py \
  scripts/prepare_expanded_amos_tsv3_dataset702.py \
  scripts/prepare_expanded_amos_tsv3_dataset702.sbatch \
  scripts/preprocess_expanded_amos_tsv3_dataset702.sbatch \
  scripts/install_dataset702_frozen_split.py \
  scripts/train_expanded_amos_tsv3_dataset702.sbatch \
  runs/nnunet3d_external_tsv3_metrics_fold0_ortho_native_unreviewed.csv \
  runs/nnunet3d_external_tsv3_metrics_fold0_ortho_native_unreviewed.summary.json \
  runs/nnunet3d_external_tsv3_failure_audit_unreviewed.json \
  runs/nnunet3d_external_tsv3_unreviewed_provenance.json \
  runs/external_geometry_restore_summary.json \
  runs/nnunet3d_external_tsv3_predictions_fold0_unreviewed \
  runs/nnunet3d_external_tsv3_ortho_predictions_fold0 \
  runs/nnunet3d_external_tsv3_native_predictions_fold0 \
  nnunet_raw/ExternalTotalSegV3AdrenalTest_v1/external_test_manifest.json \
  nnunet_raw/ExternalTotalSegV3AdrenalTest_v1/labelsTs \
  nnunet_raw/ExternalTotalSegV3AdrenalOrtho_v1/geometry_manifest.json \
  runs/expanded_dataset702/plan.json \
  nnunet_raw/Dataset702_AMOSTotalSegAdrenalCT/geometry_manifest.json \
  nnunet_preprocessed/Dataset702_AMOSTotalSegAdrenalCT/splits_final.json \
  nnunet_results/Dataset702_AMOSTotalSegAdrenalCT; do
    if [[ -e "$path" ]]; then
      printf '%s\n' "$path" >> "$LIST"
    fi
done

while IFS= read -r path; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: required evidence source absent: $path" >&2
    exit 2
  fi
done < "$LIST"

tar -cf "$ARCHIVE" --files-from="$LIST"
sha256sum "$ARCHIVE" > "$ARCHIVE.sha256"
du -h "$ARCHIVE"
cat "$ARCHIVE.sha256"
echo "SUCCESS: local archival snapshot created; includes original CNN baseline and available external/expanded research evidence."
echo "NEXT: copy $ARCHIVE and $ARCHIVE.sha256 to approved independent institutional storage."
