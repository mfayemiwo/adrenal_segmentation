#!/usr/bin/env python3
"""Evaluate gland-specific ROI memory on true full-z cached CT volumes."""
import argparse,csv,sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from src.models.adrenaspike3d import AdrenaSpike3D
from src.data.adrenaspike_roi import crop,insert,locate
from src.postprocessing.connected_components import keep_largest_k_components,remove_small_components
from src.evaluation.metrics import hausdorff_distance_95, normalized_surface_dice

def dice(a,b):
    n=a.sum()+b.sum()
    return float(2*np.logical_and(a,b).sum()/n) if n else 1.

def postprocess(mask):
    mask,_=remove_small_components(mask,30)
    if mask.any():mask=keep_largest_k_components(mask,2)
    return mask

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--volume-dir",type=Path,default=Path("runs/adrenaspike3d_cache"))
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--depth",type=int,default=24)
    p.add_argument("--size",type=int,default=96)
    p.add_argument("--max-cases",type=int,default=0)
    p.add_argument("--zero-events",action="store_true")
    p.add_argument("--no-gate",action="store_true")
    a=p.parse_args()
    if not torch.cuda.is_available():raise SystemExit("GPU required")
    ckpt=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    model=AdrenaSpike3D(ckpt["variant"],
                       event_threshold=float(ckpt["event_threshold"])).cuda().eval()
    model.load_state_dict(ckpt["model_state_dict"])
    threshold=float(ckpt["operating_threshold"])
    shape=(a.depth,a.size,a.size)
    files=sorted((a.volume_dir/"val").glob("*.npz"))
    if a.max_cases:files=files[:a.max_cases]
    if not files:raise FileNotFoundError("No validation full-z cache")
    a.output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    @torch.no_grad()
    def refine(ct,p):
        x=torch.from_numpy(ct.astype(np.float32)[None,None]).cuda()
        y=torch.from_numpy(p.astype(np.float32)[None,None]).cuda()
        out=model(x,y,operating_threshold=threshold,
                  zero_events=a.zero_events,use_gate=not a.no_gate)
        rate=float(model.last_spike_rate.cpu())
        density=float(model.last_event_density.cpu())
        offset=math.log(threshold/(1-threshold))
        return (out+offset).sigmoid()[0,0].float().cpu().numpy(),rate,density
    for i,file in enumerate(files,1):
        with np.load(file) as d:
            ct=d["ct"].astype(np.float32)
            prob=d["prob"].astype(np.float32)
            labels=d["mask"]
        output=prob.copy()
        row={"case_id":file.stem}
        for gland,name in enumerate(("left","right")):
            gt=labels==(gland+1)
            centre=locate(prob[gland])
            patch=crop(prob[gland],centre,shape)
            ctpatch=crop(ct,centre,shape)
            refined,rate,density=refine(ctpatch,patch)
            insert(output[gland],refined,centre,blend=True)
            patch_gt=crop(gt.astype(np.uint8),centre,shape)
            row[f"roi_coverage_{name}"]=float(patch_gt.sum()/max(gt.sum(),1))
            row[f"spike_rate_{name}"]=rate
            row[f"event_density_{name}"]=density
            for label,pr in (("baseline",prob[gland]),("refined",output[gland])):
                pred=postprocess(pr>=threshold)
                row[f"{label}_dice_{name}"]=dice(pred,gt)
                if pred.any() and gt.any():
                    from src.evaluation.metrics import hausdorff_distance_95,normalized_surface_dice
                    # Crop surface-distance computation to union to bound memory.
                    coords=np.argwhere(pred|gt)
                    low=np.maximum(coords.min(axis=0)-2,0)
                    high=np.minimum(coords.max(axis=0)+3,pred.shape)
                    sl=tuple(slice(int(lo),int(hi)) for lo,hi in zip(low,high))
                    row[f"{label}_hd95_{name}_mm"]=float(
                        hausdorff_distance_95(pred[sl],gt[sl],spacing=(2.5,1.,1.)))
                    row[f"{label}_nsd2mm_{name}"]=float(
                        normalized_surface_dice(pred[sl],gt[sl],spacing=(2.5,1.,1.),tolerance_mm=2.))
                else:
                    row[f"{label}_hd95_{name}_mm"]=float("nan")
                    row[f"{label}_nsd2mm_{name}"]=float("nan")
                present=pred.any(axis=(1,2));gpres=gt.any(axis=(1,2))
                row[f"{label}_z_presence_dice_{name}"]=dice(present,gpres)
        rows.append(row)
        print(f"{i}/{len(files)} {file.stem}: baseline={np.mean([row[f'baseline_dice_{n}'] for n in ('left','right')]):.4f}, "
              f"refined={np.mean([row[f'refined_dice_{n}'] for n in ('left','right')]):.4f}",flush=True)
    with (a.output_dir/"per_case.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for metric in ("dice","hd95","nsd2mm","z_presence_dice"):
        for label in ("baseline","refined"):
            keys=[f"{label}_{metric}_{g}"+("_mm" if metric=="hd95" else "")
                  for g in ("left","right")]
            vals=[np.nanmean([row[k] for k in keys]) for row in rows]
            summary.append(f"{label} {metric}: {np.nanmean(vals):.6f}")
    summary.append(f"ROI coverage mean: {np.mean([row[f'roi_coverage_{g}'] for row in rows for g in ('left','right')]):.4f}")
    (a.output_dir/"evaluation_report.txt").write_text("\n".join(summary)+"\n")
    print("\n".join(summary),flush=True)

if __name__=="__main__":main()
