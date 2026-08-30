#!/usr/bin/env python3
"""Control experiment: can the gate architecture use slice ORDER at all?

Why this exists
---------------
The first Stage A ablation found arm E (k consecutive slices as k SNN time
steps) beating arm G (the centre slice repeated k times) by 0.0035 PR-AUC -
that is, not at all. Two explanations fit that observation equally well, and
they call for opposite responses:

  (a) the ARCHITECTURE cannot represent slice order, so the sequence never had
      a chance. A fixable defect.
  (b) the architecture can represent order perfectly well, but craniocaudal
      ORDER carries little usable signal for deciding whether an adrenal gland
      is present. A genuine finding about the task, not a bug.

Real data cannot separate them: on AMOS22 both look like "no difference".
This script separates them on synthetic data where the answer is known.

The probe
---------
A Gaussian blob translates through a stack of k slices. Half the windows move
down, half move up - and the ascending class is the descending class REVERSED,
so both classes contain exactly the same k slices. The set is uninformative;
only the order distinguishes them. A model that cannot represent order is
pinned at 50% no matter how long it trains, and one that can should approach
100%.

Result (30 Aug 2026, k=5, 250 steps): every configuration tested, including
arm E exactly as it was run, reaches 99-100%. Explanation (a) is ruled out.
The null result on AMOS22 is about the data, not the model - which is a
stronger and more publishable statement than a bug would have been.

Usage
-----
    python scripts/probe_order_sensitivity.py                # default sweep
    python scripts/probe_order_sensitivity.py --steps 400 --slice-window 9
"""
from __future__ import annotations

import os as _os
import tempfile as _tempfile

for _var, _sub in (("MIOPEN_USER_DB_PATH", "miopen-db"), ("MIOPEN_CUSTOM_CACHE_DIR", "miopen-cache")):
    _os.environ.setdefault(_var, _os.path.join(_tempfile.gettempdir(), _sub))
    _os.makedirs(_os.environ[_var], exist_ok=True)

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slice-window", type=int, default=5, help="k: slices == time steps")
    p.add_argument("--image-size", type=int, default=32, help="must be divisible by 4")
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--eval-batches", type=int, default=12)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--output", type=Path, default=None, help="also write the table here")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    import torch
    import torch.nn as nn

    from src.models.cnn_lstm_gate import CNNLSTMSliceGate
    from src.models.snn_gate import SpikingSliceGate

    device = torch.device("cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available())
                          else "cpu")
    T, H = args.slice_window, args.image_size
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(H), indexing="ij")

    def make_batch(n, gen):
        """Both classes hold the same k slices; only the order differs."""
        xb, yb = [], []
        for _ in range(n):
            cx = torch.randint(H // 4, 3 * H // 4, (1,), generator=gen).item()
            cy0 = torch.randint(H // 8, H // 3, (1,), generator=gen).item()
            step = torch.randint(2, max(3, H // 8), (1,), generator=gen).item()
            sl = []
            for t in range(T):
                cy = cy0 + t * step
                g = torch.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * (H / 9.0) ** 2)))
                sl.append(g + 0.10 * torch.randn(H, H, generator=gen))
            w = torch.stack(sl)
            descending = torch.rand(1, generator=gen).item() < 0.5
            xb.append(w if descending else torch.flip(w, dims=[0]))
            yb.append(1.0 if descending else 0.0)
        return torch.stack(xb).unsqueeze(2), torch.tensor(yb)

    def probe(make, label):
        torch.manual_seed(args.seed)
        model = make().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        lossf = nn.BCEWithLogitsLoss()
        gen = torch.Generator().manual_seed(args.seed + 1)

        model.train()
        for _ in range(args.steps):
            x, y = make_batch(args.batch_size, gen)
            x, y = x.to(device), y.to(device)
            loss = lossf(model(x)[0]["left_present"], y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        gv = torch.Generator().manual_seed(9999)
        correct = total = 0
        with torch.no_grad():
            for _ in range(args.eval_batches):
                x, y = make_batch(args.batch_size, gv)
                x, y = x.to(device), y.to(device)
                pred = (torch.sigmoid(model(x)[0]["left_present"]) >= 0.5).float()
                correct += int((pred == y).sum())
                total += int(y.numel())
        return correct / max(total, 1)

    configs = [
        ("SNN  rate + batchnorm   (arm E as run)",
         lambda: SpikingSliceGate(slice_window=T, readout="rate", norm="batch")),
        ("SNN  rate + groupnorm",
         lambda: SpikingSliceGate(slice_window=T, readout="rate", norm="group")),
        ("SNN  integrator + batchnorm",
         lambda: SpikingSliceGate(slice_window=T, readout="integrator", norm="batch")),
        ("SNN  integrator + groupnorm",
         lambda: SpikingSliceGate(slice_window=T, readout="integrator", norm="group")),
        ("SNN  integrator + groupnorm + learnable LIF",
         lambda: SpikingSliceGate(slice_window=T, readout="integrator", norm="group",
                                  learn_beta=True, learn_threshold=True)),
        ("CNN-LSTM   (arm D reference)",
         lambda: CNNLSTMSliceGate(slice_window=T, lstm_hidden=32)),
    ]

    lines = [
        "Can the architecture use slice ORDER at all?",
        "Both classes contain identical slices; only the order differs, so 50% means",
        "the model cannot represent order and cannot be rescued by more training.",
        f"k = {T}, {H}x{H}, {args.steps} steps, device {device}.",
        "",
        f"  {'configuration':<50}{'accuracy':>9}",
        "  " + "-" * 60,
    ]
    for line in lines:
        print(line, flush=True)

    for label, make in configs:
        acc = probe(make, label)
        row = f"  {label:<50}{acc:>9.1%}"
        lines.append(row)
        print(row, flush=True)

    verdict = [
        "",
        "Reading it: if every row is near 100%, the architecture can represent slice",
        "order, and a null result for arm E vs arm G on real data is a statement about",
        "the data - the ordering of adjacent slices carries little extra information",
        "for gland presence - rather than an architectural defect to be fixed.",
    ]
    for line in verdict:
        lines.append(line)
        print(line)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nWritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
