#!/usr/bin/env python3
"""Make compact gland-specific ROI patches from full-volume baseline predictions.

ROI localisation never uses ground truth; report mask coverage separately.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import csv
from pathlib import Path
import numpy as np
from src.data.adrenaspike_roi import crop, locate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--volume-dir",type=Path,default=Path("runs/adrenaspike3d_cache"))
    p.add_argument("--tiles-dir",type=Path,default=Path("runs/adrenaspike3d_tiles"))
    p.add_argument("--depth",type=int,default=24)
    p.add_argument("--size",type=int,default=96)
    p.add_argument("--max-cases",type=int,default=0)
    a=p.parse_args()
    shape=(a.depth,a.size,a.size)
    a.tiles_dir.mkdir(parents=True,exist_ok=True)
    for split in ("train","val"):
        files=sorted((a.volume_dir/split).glob("*.npz"))
        if a.max_cases: files=files[:a.max_cases]
        if not files: raise FileNotFoundError(f"No full-volume predictions for {split}")
        dst=a.tiles_dir/split
        dst.mkdir(parents=True,exist_ok=True)
        rows=[]
        for i,path in enumerate(files,1):
            with np.load(path) as d:
                ct=d["ct"]; prob=d["prob"]; truth=d["mask"]
            for gland in range(2):
                p3=prob[gland].astype(np.float32)
                center=locate(p3)
                mask=(truth==(gland+1)).astype(np.uint8)
                crop_mask=crop(mask,center,shape)
                coverage=float(crop_mask.sum()/max(mask.sum(),1))
                out=dst/f"{path.stem}_g{gland}.npz"
                np.savez_compressed(out,
                    ct=crop(ct,center,shape).astype(np.float16),
                    prob=crop(p3,center,shape).astype(np.float16),
                    mask=crop_mask, center=np.asarray(center,dtype=np.int32))
                rows.append(dict(case_id=path.stem,gland=gland,
                                 coverage=coverage,gt_voxels=int(mask.sum()),
                                 roi_gt_voxels=int(crop_mask.sum())))
            print(f"{split} prepared {i}/{len(files)}",flush=True)
        with (a.tiles_dir/f"{split}_coverage.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=rows[0].keys())
            writer.writeheader();writer.writerows(rows)
        print(f"{split}: {len(rows)} patches; mean coverage {np.mean([r['coverage'] for r in rows]):.3f}; "
              f"missed ROI {(np.asarray([r['coverage'] for r in rows])<0.9).sum()}",flush=True)


if __name__=="__main__":
    main()
