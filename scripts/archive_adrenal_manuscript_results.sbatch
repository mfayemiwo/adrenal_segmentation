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
echo "SUCCESS: local research evidence archive created, including original weights and 300 NIfTI predictions."
echo "NEXT: copy $ARCHIVE and $ARCHIVE.sha256 to approved independent institutional storage."
