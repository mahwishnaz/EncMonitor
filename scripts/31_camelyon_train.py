#!/usr/bin/env python3
"""Train the Camelyon17 model once, then freeze it.

ResNet-18 (ImageNet weights) fine-tuned on a fixed random subset of the official WILDS
training split (the three training hospitals). Fixed settings, no checkpoint selection:
the model after the last epoch is the one that is monitored, as for MNIST and CIFAR-10.
Augmentation: random horizontal/vertical flips only (no colour augmentation, so that the
model is not made blind to the stain and brightness shifts that the monitor must see).

Output (outputs/camelyon/model/):
  resnet18_camelyon.pt   state_dict + settings + training indices
  train_report.json      accuracy per hospital on held-out patches (id_val for the
                         training hospitals, OOD val/test for the other two)

Usage (bundle root):
  python scripts/31_camelyon_train.py --root data/wilds
  (add --no-pretrained if the ImageNet weights cannot be downloaded)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import camelyon_common as cc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the frozen Camelyon17 model")
    ap.add_argument("--root", default="data/wilds")
    ap.add_argument("--out-dir", default="outputs/camelyon/model")
    ap.add_argument("--n-train", type=int, default=100_000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--n-eval", type=int, default=5_000, help="held-out patches per hospital for the report")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    t0 = time.time()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    ds = cc.load_wilds(args.root)
    md = cc.metadata(ds)
    train_pool = np.flatnonzero(md["split"] == md["split_dict"]["train"])
    train_idx = np.sort(rng.choice(train_pool, size=min(args.n_train, train_pool.size), replace=False))
    train_hosp = sorted(int(h) for h in np.unique(md["hospital"][train_pool]))
    print(f"training hospitals {train_hosp}: {train_pool.size:,} patches, using {train_idx.size:,}")

    x_train = torch.from_numpy(cc.load_images(ds, train_idx, args.workers))       # uint8 [N,96,96,3]
    y_train = torch.from_numpy(md["y"][train_idx])
    print(f"loaded {len(x_train):,} training patches in {time.time() - t0:.0f} s, tumor fraction {y_train.float().mean():.3f}")

    try:
        model = cc.build_model(pretrained=not args.no_pretrained)
        pretrained = not args.no_pretrained
    except Exception as exc:  # weights could not be downloaded
        print(f"ImageNet weights not available ({exc}); training from scratch")
        model, pretrained = cc.build_model(pretrained=False), False
    model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    steps = args.epochs * int(np.ceil(len(x_train) / args.batch_size))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    gen = torch.Generator().manual_seed(args.seed)
    for epoch in range(args.epochs):
        perm = torch.randperm(len(x_train), generator=gen)
        loss_sum, correct, seen = 0.0, 0, 0
        for start in range(0, len(perm), args.batch_size):
            b = perm[start:start + args.batch_size]
            x = cc.to_float(x_train[b], device)
            y = y_train[b].to(device)
            if torch.rand(1, generator=gen).item() < 0.5:
                x = x.flip(3)
            if torch.rand(1, generator=gen).item() < 0.5:
                x = x.flip(2)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += float(loss) * len(b)
            correct += int((logits.argmax(1) == y).sum())
            seen += len(b)
        print(f"epoch {epoch + 1}/{args.epochs}: loss {loss_sum / seen:.4f}  train acc {correct / seen:.4f}  ({time.time() - t0:.0f} s)")

    model.eval()
    report = {"train_hospitals": train_hosp, "per_hospital": {}}
    held = cc.heldout_by_hospital(md)
    print(f"\n{'hospital':>8} {'role':>9} {'patches':>8} {'tumor %':>8} {'accuracy':>9}")
    for h, idx in held.items():
        sel = np.sort(np.random.default_rng([args.seed, h]).choice(idx, size=min(args.n_eval, idx.size), replace=False))
        emb = cc.embed(model, cc.load_images(ds, sel, args.workers), device)
        with torch.no_grad():
            pred = model.classifier(torch.as_tensor(emb, device=device)).argmax(1).cpu().numpy()
        acc = float(np.mean(pred == md["y"][sel]))
        role = "train" if h in train_hosp else "unseen"
        report["per_hospital"][str(h)] = {"role": role, "heldout_patches": int(idx.size), "evaluated": int(sel.size),
                                          "tumor_fraction": float(md["y"][sel].mean()), "accuracy": acc}
        print(f"{h:>8} {role:>9} {sel.size:>8,} {100 * md['y'][sel].mean():>7.1f}% {acc:>9.4f}")

    torch.save({"state_dict": model.state_dict(), "pretrained_init": pretrained, "args": vars(args),
                "train_indices": train_idx, "train_hospitals": train_hosp}, out / "resnet18_camelyon.pt")
    report.update({"settings": vars(args), "pretrained_init": pretrained, "n_train": int(train_idx.size),
                   "seconds": round(time.time() - t0, 1), "torch": torch.__version__, "device": device})
    (out / "train_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved {out / 'resnet18_camelyon.pt'}  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
