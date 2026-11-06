#!/usr/bin/env python3
"""Read-only AMOS/TotalSegmentator CT overlap SCREEN (not proof of independence).

Compare AMOS CT training+heldout images with all TotalSegmentator CT cases
having nonempty bilateral adrenal masks. Two signals:
 - Exact canonical voxel checksum, with rounded CT-HU integer intensities.
 - Coarse body-normalised thumbnail correlation, used ONLY to flag potential
   resampled/reformatted duplicates for human review.

A low correlation does NOT rule out shared patients (different examination,
contrast, acquisition, crop, FOV or reconstruction). Never use this screen alone
to certify external independence. Preserve the original AMOS holdout and the
TotalSegmentator test split. Does not edit any CT images or labels.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates


def thumbnail_from_array(data: np.ndarray, size: int = 16) -> np.ndarray:
    """Orientation-normalised crude body thumbnail, not a diagnostic image.

    The array must already have canonical axis orientation if coming from NIfTI.
    -500 HU threshold approximates body coverage, not an anatomy label.
    """
    if data.ndim != 3 or min(data.shape) < 4:
        raise ValueError(f"Unexpected CT shape: {data.shape}")
    body = np.asarray(data > -500)
    bounds = []
    for axis in range(3):
        occupied = np.any(body, axis=tuple(x for x in range(3) if x != axis))
        locations = np.flatnonzero(occupied)
        if locations.size < 4:
            bounds.append((0, data.shape[axis] - 1))
        else:
            bounds.append((int(locations[0]), int(locations[-1])))
    coords = np.meshgrid(
        *[np.linspace(lo, hi, num=size, dtype=np.float32)
          for lo, hi in bounds],
        indexing="ij",
    )
    clipped = np.asarray(np.clip(data, -500, 1000), dtype=np.float32)
    sampled = map_coordinates(clipped, coords, order=1, mode="nearest")
    sampled = np.clip(sampled, -300, 500).astype(np.float32)
    sampled -= float(sampled.mean())
    norm = float(np.linalg.norm(sampled))
    if norm < 1e-5:
        raise ValueError("Constant-volume thumbnail cannot be compared")
    return (sampled / norm).ravel().astype(np.float32)


def fingerprint_ct(path: Path, size: int = 16) -> tuple[str, np.ndarray]:
    image = nib.as_closest_canonical(nib.load(str(path)))
    values = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite CT intensities: {path}")
    # Store decoded canonical image bytes rounded to 1HU; exact matches only.
    # Different resampling can easily prevent matches.
    clipped = np.rint(np.clip(values, -32768, 32767)).astype(np.int16)
    h = hashlib.sha256()
    h.update(np.asarray(clipped.shape, np.int64).tobytes())
    h.update(clipped.tobytes(order="C"))
    feature = thumbnail_from_array(values, size)
    return h.hexdigest(), feature


def eligible_total_cases(audit_csv: Path, dataset: Path):
    with audit_csv.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    chosen = []
    for r in rows:
        if str(r["both_present"]).lower() != "true":
            continue
        if r.get("error"):
            continue
        if str(r["left_aligned"]).lower() != "true" or str(r["right_aligned"]).lower() != "true":
            continue
        if str(r["left_binary"]).lower() != "true" or str(r["right_binary"]).lower() != "true":
            continue
        split = str(r.get("split", "")).strip().lower()
        if split not in ("train", "test"):
            continue
        case_id = r["case_id"]
        ct_path = dataset / case_id / "ct.nii.gz"
        if not ct_path.is_file():
            raise FileNotFoundError(ct_path)
        chosen.append((case_id, ct_path, split))
    return sorted(chosen)


def amos_cts(data_root: Path):
    cases = []
    ids = set()
    for split in ("imagesTr", "imagesVa"):
        for path in sorted((data_root / split).glob("amos_*.nii.gz")):
            cid = path.name[:-7]
            if int(cid.split("_")[-1]) >= 500:
                continue
            if cid in ids:
                raise ValueError(f"AMOS case duplicated across splits: {cid}")
            ids.add(cid)
            cases.append((cid, path, split))
    if not cases:
        raise FileNotFoundError("No AMOS CT cases under imagesTr/imagesVa")
    return cases


def cached_fingerprint(path: Path, cache: Path, size: int) -> tuple[str, np.ndarray]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    # Namespaced cache; source size+mtime provides a basic stale-cache guard.
    stat = path.stat()
    source_key = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{size}"
    source_hash = hashlib.sha256(source_key.encode("utf-8")).hexdigest()
    if cache.exists():
        try:
            with np.load(cache, allow_pickle=False) as prev:
                if str(prev["source_hash"]) == source_hash:
                    return str(prev["voxel_sha256"]), prev["thumbnail"].astype(np.float32)
        except (OSError, ValueError, KeyError):
            pass
    digest, feature = fingerprint_ct(path, size)
    tmp = cache.with_name(cache.name + ".partial")
    with tmp.open("wb") as fp:
        np.savez_compressed(fp, source_hash=source_hash, voxel_sha256=digest,
                            thumbnail=feature)
    tmp.replace(cache)
    return digest, feature


def top_candidates(ref: np.ndarray, pool: np.ndarray, cutoff: float = 0.985,
                   top_k: int = 3) -> list[tuple[int, int, float]]:
    """Return all pairs >= cutoff, optionally adding top-k per reference for review.

    This is a similarity screen, not a validated matching/registration method.
    """
    if not (0 <= cutoff <= 1):
        raise ValueError("cutoff must be in [0,1]")
    if ref.ndim != 2 or pool.ndim != 2 or ref.shape[1] != pool.shape[1]:
        raise ValueError("Mismatched feature matrices")
    scores = ref @ pool.T
    selected = set()
    for row in range(len(ref)):
        close = np.flatnonzero(scores[row] >= cutoff)
        for col in close:
            selected.add((row, int(col)))
        if top_k:
            for col in np.argsort(scores[row])[-min(top_k, len(pool)):]:
                selected.add((row, int(col)))
    return sorted(
        ((i, j, float(scores[i, j])) for i, j in selected),
        key=lambda x: -x[2],
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--amos-root", type=Path, default=Path("../data/amos22"))
    p.add_argument("--totalseg-root", type=Path, default=Path(
        "/mnt/userapps/isrcdl_apps/mfayemiwo/DaisyProject/data/"
        "totalsegmentator_v3/extracted/Totalsegmentator_dataset_v300"))
    p.add_argument("--audit", type=Path, default=Path(
        "runs/totalsegmentator_v3_audit/case_audit.csv"))
    p.add_argument("--output-dir", type=Path, default=Path(
        "runs/ct_overlap_screen"))
    p.add_argument("--size", type=int, default=16)
    p.add_argument("--correlation-flag", type=float, default=0.985)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--limit", type=int, default=0,
                   help="Debug only: process first N cases per source")
    args = p.parse_args()
    if args.size < 8 or args.size > 32 or args.top_k < 0 or args.limit < 0:
        p.error("size must be 8..32, top-k/limit >=0")

    amos = amos_cts(args.amos_root)
    total = eligible_total_cases(args.audit, args.totalseg_root)
    if args.limit:
        amos, total = amos[:args.limit], total[:args.limit]
    if not total:
        raise ValueError("No bilateral TotalSegmentator CTs selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fingerprints = {}
    for source, cases in (("amos", amos), ("totalseg", total)):
        digest_list, feature_list = [], []
        for i, (cid, path, split) in enumerate(cases, 1):
            digest, feature = cached_fingerprint(
                path, args.output_dir / "fingerprints" /
                source / (cid + ".npz"), args.size
            )
            digest_list.append(digest)
            feature_list.append(feature)
            if i == 1 or i % 50 == 0 or i == len(cases):
                print(f"{source}: fingerprinted {i}/{len(cases)}", flush=True)
        fingerprints[source] = (digest_list, np.stack(feature_list))

    amos_hashes, amos_features = fingerprints["amos"]
    total_hashes, total_features = fingerprints["totalseg"]
    matches = []
    for i, j, score in top_candidates(
        amos_features, total_features, args.correlation_flag, args.top_k
    ):
        exact = amos_hashes[i] == total_hashes[j]
        if not exact and score < args.correlation_flag and args.top_k == 0:
            continue
        matches.append({
            "amos_case": amos[i][0], "amos_split": amos[i][2],
            "totalseg_case": total[j][0], "totalseg_split": total[j][2],
            "exact_canonical_voxel_match": exact,
            "coarse_thumbnail_correlation": round(score, 6),
            "high_similarity_flag": score >= args.correlation_flag,
            "action": "MANUAL_REVIEW_REQUIRED",
        })
    output = args.output_dir / "candidate_overlap_pairs.csv"
    with output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=[
            "amos_case", "amos_split", "totalseg_case", "totalseg_split",
            "exact_canonical_voxel_match", "coarse_thumbnail_correlation",
            "high_similarity_flag", "action",
        ])
        writer.writeheader()
        writer.writerows(matches)
    summary = {
        "amos_cases_screened": len(amos),
        "totalseg_bilateral_cases_screened": len(total),
        "exact_canonical_voxel_matches": sum(
            x["exact_canonical_voxel_match"] for x in matches),
        "high_similarity_pairs_flagged": sum(
            x["high_similarity_flag"] for x in matches),
        "candidate_pairs_for_manual_review": len(matches),
        "cutoff_is_unvalidated": args.correlation_flag,
        "complete_screen": args.limit == 0,
        "cautions": [
            "No matches is NOT proof of patient independence.",
            "Low-resolution similarity can give false positives and false negatives.",
            "Different CT acquisition, FOV, orientation, crop and resampling can evade detection.",
            "Manual review of flagged images and source provenance is required.",
            "NEVER add confirmed AMOS-heldout matches to training data.",
            "This tool does not alter datasets, train, or apply labels.",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print("Candidates:", output, flush=True)


if __name__ == "__main__":
    main()
