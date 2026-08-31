#!/usr/bin/env python3
"""Where the spiking gate's neurons actually fire - and whether training can reach them.

Surrogate-gradient training has a specific failure mode: a neuron whose membrane
potential never approaches threshold receives a surrogate derivative of
approximately zero, so it never learns and never will. Dead neurons are
invisible in the loss curve and cost capacity silently.

This measures three things a single evaluation pass can settle.

1. HOW MANY NEURONS ARE DEAD. Per spiking layer, the fraction of channels that
   never fire across the whole validation set, and the fraction that fire on
   essentially every input (saturated - equally uninformative).

2. WHETHER NORMALISATION IS REALLY A REACHABILITY FIX. Replacing BatchNorm with
   GroupNorm was worth +0.051 PR-AUC to the spiking gate and +0.0001 to the
   recurrent baseline. If the GroupNorm checkpoint also has markedly fewer dead
   channels, that gain is best explained as surrogate gradients reaching neurons
   they previously could not, rather than as regularisation. Pass both
   checkpoints to compare them directly.

3. WHETHER COST SCALES WITH CONTENT. A gate exists to reject empty slices, and a
   spiking network only does work when neurons fire - so in principle it should
   be cheapest on exactly the slices it discards. That is the one efficiency
   claim available here that does not require neuromorphic hardware to state,
   and it is measurable: compare activity on gland-bearing against gland-free
   windows.

Usage
-----
    python scripts/analyse_spike_activity.py \
        --checkpoints runs/seed_snn_s1/best_model.pt runs/snn_group/best_model.pt \
        --data-root ../data/amos22 --cache-dir ../cache --max-cases 20
"""
from __future__ import annotations

import os as _os
import tempfile as _tempfile

for _var, _sub in (("MIOPEN_USER_DB_PATH", "miopen-db"), ("MIOPEN_CUSTOM_CACHE_DIR", "miopen-cache")):
    _os.environ.setdefault(_var, _os.path.join(_tempfile.gettempdir(), _sub))
    _os.makedirs(_os.environ[_var], exist_ok=True)

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_adrenal_segmenter import (            # noqa: E402
    GeometryConfig, discover_cases, load_or_build_cache, setup_logging,
)
from train_slice_gate import (                   # noqa: E402
    _make_dataset_class, prepare_gate_cases,
)

DEAD_RATE = 1e-6          # never fired at all
NEAR_DEAD_RATE = 1e-3     # fires on fewer than one input in a thousand
SATURATED_RATE = 0.99     # fires on essentially everything


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True,
                   help="one or more spiking-gate checkpoints to compare")
    p.add_argument("--labels", nargs="*", default=None,
                   help="display names, one per checkpoint (default: parent directory)")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=Path("runs/_volume_cache"))
    p.add_argument("--max-cases", type=int, default=20,
                   help="validation cases to measure over; 20 is plenty for this")
    p.add_argument("--max-windows", type=int, default=600,
                   help="cap per class (positive / negative) so the pass stays quick")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--output-dir", type=Path, default=None)
    return p


class ActivityRecorder:
    """Accumulates per-channel spike statistics from forward hooks on the LIF layers.

    The hook fires once per time step, so counts aggregate over batch, time and
    space; the per-channel firing rate is spikes divided by the number of
    (sample, time, position) opportunities that channel had to fire.
    """

    def __init__(self, model, time_steps: int):
        import snntorch as snn
        self.layers = [(n or "readout", m) for n, m in model.named_modules()
                       if isinstance(m, snn.Leaky)]
        self.T = max(1, int(time_steps))
        self.spikes: dict[str, np.ndarray] = {}
        self.opportunities: dict[str, float] = {}
        # Activity resolved BY TIME STEP. The hook fires once per step within a
        # forward pass, so the call index modulo T recovers which step it was -
        # averaging them together would hide exactly the accumulation this
        # section exists to look for.
        self.step_spikes: dict[str, np.ndarray] = {}
        self.step_calls: dict[str, np.ndarray] = {}
        self._calls: dict[str, int] = {}
        self._handles = []

    def _hook(self, name):
        def fn(_module, _inp, out):
            spk = out[0] if isinstance(out, (tuple, list)) else out
            s = spk.detach()
            dims = tuple(d for d in range(s.ndim) if d != 1)      # keep the channel axis
            per_channel = s.sum(dim=dims).float().cpu().numpy()
            n_per_channel = float(s.numel() / s.shape[1])
            if name not in self.spikes:
                self.spikes[name] = np.zeros_like(per_channel)
                self.opportunities[name] = 0.0
                self.step_spikes[name] = np.zeros(self.T)
                self.step_calls[name] = np.zeros(self.T)
                self._calls[name] = 0
            self.spikes[name] += per_channel
            self.opportunities[name] += n_per_channel
            step = self._calls[name] % self.T
            self.step_spikes[name][step] += float(s.mean())
            self.step_calls[name][step] += 1
            self._calls[name] += 1
        return fn

    def __enter__(self):
        for name, module in self.layers:
            self._handles.append(module.register_forward_hook(self._hook(name)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def summary(self) -> list[dict]:
        rows = []
        for name, _ in self.layers:
            if name not in self.spikes:
                continue
            rate = self.spikes[name] / max(self.opportunities[name], 1.0)
            rows.append({
                "layer": name,
                "channels": int(rate.size),
                "mean_rate": float(rate.mean()),
                "dead": int((rate <= DEAD_RATE).sum()),
                "near_dead": int((rate < NEAR_DEAD_RATE).sum()),
                "saturated": int((rate >= SATURATED_RATE).sum()),
                "max_rate": float(rate.max()),
            })
        return rows

    def per_step_rates(self) -> dict[str, list[float]]:
        out = {}
        for name in self.step_spikes:
            calls = np.maximum(self.step_calls[name], 1.0)
            out[name] = (self.step_spikes[name] / calls).tolist()
        return out

    def overall_rate(self) -> float:
        tot_s = sum(float(v.sum()) for v in self.spikes.values())
        tot_n = sum(self.opportunities[k] * self.spikes[k].size for k in self.spikes)
        return tot_s / max(tot_n, 1.0)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = args.output_dir or args.checkpoints[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "spike_activity.log")

    import torch
    from torch.utils.data import DataLoader

    from evaluate_slice_gate import build_gate

    labels = args.labels or [c.parent.name for c in args.checkpoints]
    if len(labels) != len(args.checkpoints):
        print("--labels must have one entry per checkpoint", file=sys.stderr)
        return 2

    device = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available())
                          else "cpu")
    lines: list[str] = []
    def emit(t: str = "") -> None:
        lines.append(t); print(t, flush=True)

    emit("=" * 80)
    emit(" SPIKING GATE - WHERE THE NEURONS FIRE")
    emit("=" * 80)
    emit(f" device {device} | {len(args.checkpoints)} checkpoint(s) | "
         f"{args.max_cases} validation cases")
    emit("")

    results = {}
    for ckpt_path, label in zip(args.checkpoints, labels):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        if str(cfg.get("model", "snn")) != "snn":
            emit(f" skipping {label}: not a spiking checkpoint")
            continue

        model, _ = build_gate(cfg, device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()

        geom = GeometryConfig(
            spacing_z=float(cfg.get("target_spacing_z", 2.5)),
            spacing_xy=float(cfg.get("target_spacing_xy", 1.0)),
            image_size=int(cfg.get("cache_image_size", 384)),
            z_margin=int(cfg.get("z_margin", 32)),
            right_label=int(cfg.get("right_label", 11)),
            left_label=int(cfg.get("left_label", 12)),
        )
        disc = SimpleNamespace(
            data_root=args.data_root, seed=int(cfg.get("seed", 42)),
            val_fraction=float(cfg.get("val_fraction", 0.2)),
            max_train_cases=0, max_val_cases=args.max_cases,
            modality=str(cfg.get("modality", "ct")),
            mri_id_threshold=int(cfg.get("mri_id_threshold", 500)),
        )
        _, val_records = discover_cases(disc, logger)
        cases = prepare_gate_cases(
            load_or_build_cache(val_records, geom, args.cache_dir, False, logger),
            int(cfg.get("image_size", 192)), int(cfg.get("cache_image_size", 384)), logger)

        # Two separate passes so activity can be attributed to slice content.
        positives, negatives = [], []
        for ci, case in enumerate(cases):
            any_gland = case["labels"].any(axis=1)
            for z in range(case["labels"].shape[0]):
                (positives if any_gland[z] else negatives).append((ci, z))
        rng = np.random.default_rng(0)
        for pool in (positives, negatives):
            rng.shuffle(pool)
        positives, negatives = positives[: args.max_windows], negatives[: args.max_windows]

        Dataset = _make_dataset_class()
        per_class = {}
        for cls, index in (("gland-bearing", positives), ("gland-free", negatives)):
            if not index:
                continue
            loader = DataLoader(Dataset(cases, index, int(cfg.get("slice_window", 5)),
                                        bool(cfg.get("static_repeat", False))),
                                batch_size=args.batch_size, shuffle=False)
            with ActivityRecorder(model, int(cfg.get("slice_window", 5))) as rec, torch.no_grad():
                for x, _y in loader:
                    model(x.to(device))
            per_class[cls] = {"summary": rec.summary(), "overall": rec.overall_rate(),
                              "per_step": rec.per_step_rates(), "n": len(index)}
        results[label] = per_class
        logger.info("measured %s", label)

    if not results:
        emit(" no spiking checkpoints to analyse")
        (out_dir / "spike_activity.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 2

    emit("-" * 80)
    emit(" 1. DEAD AND SATURATED CHANNELS  (measured on gland-bearing windows)")
    emit("-" * 80)
    emit(" A channel that never fires cannot be reached by a surrogate gradient; a")
    emit(" channel that always fires carries no information. Both are lost capacity.")
    emit("")
    emit(f"  {'checkpoint':<20}{'layer':<16}{'chans':>7}{'mean rate':>11}"
         f"{'dead':>7}{'near-dead':>11}{'saturated':>11}")
    emit("  " + "-" * 76)
    for label, per_class in results.items():
        block = per_class.get("gland-bearing") or next(iter(per_class.values()))
        for r in block["summary"]:
            emit(f"  {label:<20}{r['layer']:<16}{r['channels']:>7}{r['mean_rate']:>11.4f}"
                 f"{r['dead']:>7}{r['near_dead']:>11}{r['saturated']:>11}")
        tot_c = sum(r["channels"] for r in block["summary"])
        tot_d = sum(r["dead"] for r in block["summary"])
        tot_nd = sum(r["near_dead"] for r in block["summary"])
        emit(f"  {'':<20}{'TOTAL':<16}{tot_c:>7}{'':>11}{tot_d:>7}{tot_nd:>11}")
        emit(f"  {'':<20}{'':16}{'':>7}{'':>11}{tot_d/max(tot_c,1):>6.1%}{tot_nd/max(tot_c,1):>11.1%}")
        emit("")

    if len(results) > 1:
        emit("-" * 80)
        emit(" 2. COMPARISON")
        emit("-" * 80)
        emit(" If the configuration that scored higher also has fewer dead channels, its")
        emit(" gain is better explained as surrogate gradients reaching neurons that were")
        emit(" previously unreachable than as a regularisation effect.")
        emit("")
        emit(f"  {'checkpoint':<24}{'dead channels':>16}{'near-dead':>13}{'mean firing rate':>19}")
        emit("  " + "-" * 72)
        for label, per_class in results.items():
            block = per_class.get("gland-bearing") or next(iter(per_class.values()))
            tot_c = sum(r["channels"] for r in block["summary"])
            tot_d = sum(r["dead"] for r in block["summary"])
            tot_nd = sum(r["near_dead"] for r in block["summary"])
            emit(f"  {label:<24}{tot_d:>6} / {tot_c:<7}{tot_nd/max(tot_c,1):>13.1%}"
                 f"{block['overall']:>19.4f}")
        emit("")

    emit("-" * 80)
    emit(" 3. DOES COST SCALE WITH CONTENT?")
    emit("-" * 80)
    emit(" A gate exists to reject empty slices. A spiking network only works when it")
    emit(" fires, so it should be cheapest on exactly the slices it discards - the one")
    emit(" efficiency claim here that needs no neuromorphic hardware to state.")
    emit("")
    emit(f"  {'checkpoint':<24}{'gland-bearing':>16}{'gland-free':>14}{'ratio':>9}")
    emit("  " + "-" * 64)
    for label, per_class in results.items():
        pos = per_class.get("gland-bearing")
        neg = per_class.get("gland-free")
        if not (pos and neg):
            emit(f"  {label:<24}   (need both classes present)")
            continue
        ratio = neg["overall"] / max(pos["overall"], 1e-12)
        emit(f"  {label:<24}{pos['overall']:>16.4f}{neg['overall']:>14.4f}{ratio:>9.2f}")
    emit("")
    emit("  A ratio below 1 means the gate genuinely spends less on the slices it")
    emit("  rejects. Near 1 means activity is content-independent, and the")
    emit("  event-driven efficiency argument does not hold for this model.")
    emit("")

    emit("-" * 80)
    emit(" 4. ACTIVITY ACROSS THE k TIME STEPS  (gland-bearing windows)")
    emit("-" * 80)
    emit(" Rising activity suggests membrane potential is accumulating across slices;")
    emit(" flat activity suggests each step is being processed largely independently.")
    emit("")
    for label, per_class in results.items():
        block = per_class.get("gland-bearing")
        if not block:
            continue
        steps = len(next(iter(block["per_step"].values()), []))
        emit(f"  {label}")
        emit(f"    {'layer':<18}" + "".join(f"{'t=' + str(i + 1):>9}" for i in range(steps))
             + f"{'last/first':>12}")
        for layer, rates in block["per_step"].items():
            # A readout that is silent at t=1 is expected, not an error: nothing
            # has accumulated yet. Report a dash rather than a division by zero.
            trend = f"{rates[-1] / rates[0]:.2f}" if rates and rates[0] > 1e-9 else "-"
            emit(f"    {layer:<18}" + "".join(f"{r:>9.4f}" for r in rates) + f"{trend:>12}")
        emit("")
    emit("  A ratio above 1 is the signature of membrane potential accumulating along the")
    emit("  slice sequence. Close to 1 at every layer would mean each step is being")
    emit("  processed largely independently, and the recurrence is doing little.")
    emit("")

    csv_path = out_dir / "spike_activity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["checkpoint", "class", "layer", "channels",
                                           "mean_rate", "dead", "near_dead", "saturated",
                                           "max_rate"])
        w.writeheader()
        for label, per_class in results.items():
            for cls, block in per_class.items():
                for r in block["summary"]:
                    w.writerow({"checkpoint": label, "class": cls, **r})

    report = out_dir / "spike_activity.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nPer-layer CSV: {csv_path}\nReport:        {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
