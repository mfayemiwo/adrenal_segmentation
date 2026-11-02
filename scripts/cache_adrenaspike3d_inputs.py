#!/usr/bin/env python3
"""Cache full-z CT, ground truth and 2.5D baseline probabilities without label-cropping.

The output is used for gland-localised AdrenaSpike-3D ROI refinement.
IMPORTANT: ROI selection uses probabilities only, never the mask.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import torch
import segmentation_models_pytorch as smp


def spatial_crop_pad(a, size, pad):
    for axis in (1, 2):
        d = a.shape[axis]
        if d > size:
            k = (d-size)//2
            a = np.take(a, np.arange(k, k+size), axis=axis)
        elif d < size:
            widths = [(0, 0)] * 3
            widths[axis] = ((size-d)//2, (size-d+1)//2)
            a = np.pad(a, widths, constant_values=pad)
    return a


def load_case(im, lb, config):
    v = nib.load(str(im))
    t = nib.load(str(lb))
    if v.shape != t.shape:
        raise ValueError(f"Image/label mismatch: {im}")
    image = np.asarray(v.dataobj, dtype=np.float32).transpose(2, 0, 1)
    lab = np.asarray(t.dataobj, dtype=np.int16).transpose(2, 0, 1)
    zz, xx, yy = v.header.get_zooms()[:3][::-1]
    target_z = float(config.get("target_spacing_z", 2.5))
    target_xy = float(config.get("target_spacing_xy", 1.0))
    factors = (zz / target_z, xx / target_xy, yy / target_xy)
    label = np.zeros_like(lab, dtype=np.uint8)
    label[lab == int(config.get("left_label", 12))] = 1
    label[lab == int(config.get("right_label", 11))] = 2
    if not np.any(label):
        raise ValueError(f"No adrenal annotation in {lb}")
    image = zoom(image, factors, order=1, mode="nearest", prefilter=False)
    label = zoom(label, factors, order=0, mode="nearest", prefilter=False)
    lo, hi = config.get("hu_window", [-135.0, 215.0])
    image = (np.clip(image, lo, hi)-lo)/max(hi-lo, 1e-6)
    if config.get("clahe", False):
        from skimage import exposure
        image = np.stack([exposure.equalize_adapthist(sl, clip_limit=0.01)
                          for sl in image]).astype(np.float32)
    mean, sd = float(image.mean()), float(image.std())
    image = (image-mean)/sd if sd > 1e-6 else image-mean
    size = int(config.get("image_size", 384))
    image = spatial_crop_pad(image, size, float(image.min()))
    label = spatial_crop_pad(label, size, 0)
    return image.astype(np.float32), label


@torch.no_grad()
def infer(model, image, k, batch, device):
    n = image.shape[0]
    half = k//2
    out = []
    for start in range(0, n, batch):
        inds = [np.clip(np.arange(i-half,i+half+1),0,n-1)
                for i in range(start,min(start+batch,n))]
        x = torch.from_numpy(np.stack([image[idx] for idx in inds])).to(device)
        out.append(model(x).sigmoid().float().cpu().numpy())
    return np.concatenate(out, axis=0).transpose(1,0,2,3)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("../data/amos22"))
    p.add_argument("--checkpoint", type=Path, default=Path("runs/sb_positives/best_model.pt"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/adrenaspike3d_cache"))
    p.add_argument("--split", choices=["train", "val", "both"], default="both")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-cases", type=int, default=0)
    a = p.parse_args()
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise SystemExit("GPU not detected. Submit this script through SLURM.")
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    if cfg.get("combine_glands", False):
        raise SystemExit("Need separate left and right output channels.")
    k = int(cfg.get("slice_window", 5))
    model = smp.Unet(
        encoder_name=str(cfg.get("encoder", "resnet34")),
        encoder_weights=None, in_channels=k, classes=2,
        decoder_attention_type=None if cfg.get("decoder_attention") == "none" else "scse"
    ).cuda().eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    selected = ["train", "val"] if a.split == "both" else [a.split]
    for split in selected:
        xdir, ydir = ("imagesTr", "labelsTr") if split == "train" else ("imagesVa", "labelsVa")
        dest = a.out_dir / split
        dest.mkdir(parents=True, exist_ok=True)
        image_paths = sorted((a.data_root / xdir).glob("*.nii.gz"))
        image_paths = [im for im in image_paths
                       if int("".join(c for c in im.stem if c.isdigit())) < int(cfg.get("mri_id_threshold", 500))]
        if a.max_cases:
            image_paths = image_paths[:a.max_cases]
        if not image_paths:
            raise ValueError(f"No CT cases under {a.data_root / xdir}")
        for n, im in enumerate(image_paths, 1):
            cid = im.name[:-7]
            path = dest / f"{cid}.npz"
            if path.exists():
                print(f"SKIP {split} {cid} (already cached)", flush=True)
                continue
            image, mask = load_case(im, a.data_root / ydir / im.name, cfg)
            probs = infer(model, image, k, a.batch_size, "cuda")
            tmp = dest / f"{cid}.tmp.npz"
            np.savez_compressed(tmp, ct=image.astype(np.float16),
                                prob=probs.astype(np.float16),
                                mask=mask.astype(np.uint8))
            tmp.replace(path)
            print(f"{split} {n}/{len(image_paths)} {cid}: z={len(image)}", flush=True)
    (a.out_dir / "manifest.json").write_text(json.dumps({
        "baseline_checkpoint": str(a.checkpoint), "baseline_config": cfg,
        "note": "Full resampled CT z-axis; no label-based z cropping; predictions generated from frozen Stage B."
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
