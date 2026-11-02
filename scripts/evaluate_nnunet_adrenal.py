#!/usr/bin/env python3
"""Evaluate true full abdominal nnU-Net prediction NIfTIs on the heldout CT cohort."""
import argparse,csv,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import nibabel as nib
import numpy as np
from src.evaluation.metrics import hausdorff_distance_95, normalized_surface_dice


def dice(a,b):
    n=a.sum()+b.sum()
    return float(2*np.logical_and(a,b).sum()/n) if n else 1.


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred-dir",type=Path,default=Path("runs/nnunet3d_holdout_predictions"))
    p.add_argument("--dataset-dir",type=Path,default=Path("nnunet_raw/Dataset701_AMOSAdrenalCT"))
    p.add_argument("--output",type=Path,default=Path("runs/nnunet3d_holdout_metrics.csv"))
    a=p.parse_args()
    files=sorted((a.dataset_dir/"labelsTs").glob("*.nii.gz"))
    if not files:raise FileNotFoundError("No heldout ground truth labels")
    rows=[]
    for gt_path in files:
        cid=gt_path.name[:-7]
        pred_path=a.pred_dir/gt_path.name
        if not pred_path.is_file():raise FileNotFoundError(pred_path)
        truth_n=nib.load(str(gt_path));pred_n=nib.load(str(pred_path))
        if truth_n.shape!=pred_n.shape:raise ValueError(f"Shape mismatch for {cid}")
        t=np.asarray(truth_n.dataobj,dtype=np.uint8).transpose(2,0,1)
        pred=np.asarray(pred_n.dataobj,dtype=np.uint8).transpose(2,0,1)
        sx,sy,sz=truth_n.header.get_zooms()[:3]
        row={"case_id":cid}
        for g,name in ((1,"left"),(2,"right")):
            gt=t==g;pr=pred==g
            row[f"dice_{name}"]=dice(pr,gt)
            if gt.any() and pr.any():
                coords=np.argwhere(pr|gt)
                lo=np.maximum(coords.min(0)-2,0);hi=np.minimum(coords.max(0)+3,gt.shape)
                sl=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))
                row[f"hd95_{name}_mm"]=float(hausdorff_distance_95(
                    pr[sl],gt[sl],spacing=(sz,sx,sy)))
                row[f"nsd2mm_{name}"]=float(normalized_surface_dice(
                    pr[sl],gt[sl],spacing=(sz,sx,sy),tolerance_mm=2.))
            else:
                row[f"hd95_{name}_mm"]=float("nan")
                row[f"nsd2mm_{name}"]=float("nan")
        rows.append(row)
        print(f"{cid}: {(row['dice_left']+row['dice_right'])/2:.4f}",flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(f"mean Dice: {np.mean([(r['dice_left']+r['dice_right'])/2 for r in rows]):.5f} over {len(rows)} heldout patients")

    # Optional energy experiment injected by an explicit marker file.
    # This permits attaching a non-critical energy study to an ALREADY QUEUED
    # SLURM job (the batch script itself is snapshotted at sbatch submission).
    # Baseline nnU-Net metrics are already written before this hook executes.
    marker=Path("runs/enable_energy_after_nnunet")
    if marker.is_file():
        import subprocess
        print("Energy marker found: collecting matched Phase 2 SNN/analog GPU measurements "
              "inside this SLURM allocation.", flush=True)
        logdir=Path("runs/energy_phase2")
        logdir.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,"-u","scripts/run_phase2_energy_suite.py",
             "--output-dir",str(logdir), "--repeats","3",
             "--min-remaining-minutes","180"]
        with (logdir/"hook.log").open("a",encoding="utf-8") as logfile:
            try:
                result=subprocess.run(cmd,stdout=logfile,stderr=subprocess.STDOUT,
                                      check=False)
                print(f"Energy follow-up finished with status {result.returncode}; "
                      f"see {logdir}/hook.log",flush=True)
            except Exception as exc:
                logfile.write(f"Energy suite launch failed: {type(exc).__name__}: {exc}\n")
                print("Optional energy suite failed to start; baseline metrics "
                      "were saved successfully.",flush=True)
    else:
        print("Optional Phase 2 energy benchmark not enabled; "
              "create runs/enable_energy_after_nnunet before this stage to opt in.",
              flush=True)


if __name__=="__main__":main()
