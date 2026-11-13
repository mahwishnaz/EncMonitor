#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from encmonitor.artifacts import write_json
from encmonitor.data import load_dataset
from encmonitor.models import build_model
from encmonitor.repro import seed_everything


def evaluate(model, loader, device):
    import torch

    model.eval()
    correct = total = 0
    loss_sum = 0.0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += float(criterion(logits, y))
            correct += int((logits.argmax(1) == y).sum())
            total += int(y.numel())
    return {"loss": loss_sum / total, "accuracy": correct / total}


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the local plaintext embedding model")
    ap.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    ap.add_argument("--model", choices=["lenet", "resnet18_cifar"], default="lenet")
    ap.add_argument("--root", default="data")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/models/model.pt")
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    seed_everything(args.seed)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    train = load_dataset(args.dataset, "train", args.root, args.download)
    test = load_dataset(args.dataset, "test", args.root, args.download)
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=g, num_workers=2)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=2)
    model = build_model(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        seen = 0
        loss_sum = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            seen += int(y.numel())
            loss_sum += float(loss.detach()) * y.numel()
        row = {"epoch": epoch, "train_loss": loss_sum / seen, **evaluate(model, test_loader, device)}
        history.append(row)
        print(json.dumps(row))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_name": args.model,
        "dataset": args.dataset,
        "state_dict": model.state_dict(),
        "seed": args.seed,
        "history": history,
    }, out)
    write_json(out.with_suffix(".json"), {"checkpoint": str(out), "history": history})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
