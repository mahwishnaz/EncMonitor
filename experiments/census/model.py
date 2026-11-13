#!/usr/bin/env python3
"""Census experiment, step 1 (GPU): ACSIncome tables, frozen MLP, 64-dim penultimate representations.

Writes cache/acs_<year>.npz (per row: state code, household id, raw features, label, representation, model logit)
and cache/model.pt. Stops if CUDA is unavailable.

Usage (bundle root):
  for i in 1 2 3 4 5 6 7 8; do CUDA_VISIBLE_DEVICES=0 python experiments/census/model.py && break; sleep 15; done
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
from folktables import ACSDataSource  # noqa: E402
from folktables.acs import adult_filter  # noqa: E402

STATES = ["CA", "TX", "FL", "NY", "PA"]
FIPS = {"CA": 6, "TX": 48, "FL": 12, "NY": 36, "PA": 42}
YEARS = [2014, 2015, 2016, 2017, 2018]
FEATURES = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P"]
NUMERIC = ["AGEP", "WKHP", "SCHL"]
CATEG = ["COW", "MAR", "RELP", "SEX", "RAC1P", "OCCP_GRP", "POBP_GRP"]
CACHE = HERE / "cache"


def load_year(year: int) -> pd.DataFrame:
    raw = ACSDataSource(survey_year=str(year), horizon="1-Year", survey="person", root_dir=str(HERE / "data")).get_data(states=STATES, download=False)
    df = adult_filter(raw)[["ST", "SERIALNO"] + FEATURES + ["PINCP"]].copy()
    df["label"] = (df.PINCP > 50000).astype(np.int64)
    df[FEATURES] = df[FEATURES].fillna(-1)
    df["OCCP_GRP"] = np.floor(df.OCCP.astype(float) / 100).astype(int)
    df["POBP_GRP"] = np.floor(df.POBP.astype(float) / 100).astype(int)
    df["year"] = year
    return df.reset_index(drop=True)


class Encoder:
    def fit(self, df):
        self.mu = df[NUMERIC].astype(float).mean().to_numpy()
        self.sd = df[NUMERIC].astype(float).std().to_numpy() + 1e-9
        self.cats = {c: sorted(df[c].astype(int).unique().tolist()) for c in CATEG}
        return self

    def transform(self, df) -> np.ndarray:
        parts = [(df[NUMERIC].astype(float).to_numpy() - self.mu) / self.sd]
        for c in CATEG:
            v = df[c].astype(int).to_numpy()
            parts.append((v[:, None] == np.asarray(self.cats[c])[None, :]).astype(np.float32))
        return np.concatenate(parts, axis=1).astype(np.float32)

    def state(self):
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist(), "cats": self.cats}


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.fc = nn.Linear(64, 2)

    def forward_features(self, x):
        return self.body(x)

    def forward(self, x):
        return self.fc(self.body(x))


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available -- stop (device rule: model training and extraction on the GPU).")
    dev = torch.device("cuda")
    CACHE.mkdir(exist_ok=True)
    frames = {y: load_year(y) for y in YEARS}
    d14 = frames[2014]
    hh = np.unique(d14.SERIALNO.astype(str).to_numpy())
    rng = np.random.default_rng(2026)
    train_hh = set(rng.choice(hh, size=hh.size // 2, replace=False).tolist())
    d14["train"] = d14.SERIALNO.astype(str).isin(train_hh)
    enc = Encoder().fit(d14[d14.train])
    Xtr = torch.as_tensor(enc.transform(d14[d14.train]))
    ytr = torch.as_tensor(d14[d14.train].label.to_numpy())
    torch.manual_seed(2026)
    model = MLP(Xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(2026)
    for ep in range(10):
        perm = torch.randperm(Xtr.shape[0], generator=g)
        tot = 0.0
        for i in range(0, perm.numel(), 512):
            idx = perm[i:i + 512]
            xb, yb = Xtr[idx].to(dev), ytr[idx].to(dev)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss) * idx.numel()
        print(f"epoch {ep + 1}: loss {tot / perm.numel():.4f}", flush=True)
    model.eval()
    torch.save({"state_dict": model.state_dict(), "encoder": enc.state(), "input_dim": int(Xtr.shape[1])}, CACHE / "model.pt")
    meta = {"train_rows": int(d14.train.sum()), "train_households": len(train_hh), "acc": {}}
    for y, df in frames.items():
        X = torch.as_tensor(enc.transform(df))
        reps, logits = [], []
        with torch.no_grad():
            for i in range(0, X.shape[0], 65536):
                xb = X[i:i + 65536].to(dev)
                h = model.forward_features(xb)
                reps.append(h.cpu().numpy()); logits.append(model.fc(h).cpu().numpy())
        reps, logits = np.concatenate(reps), np.concatenate(logits)
        train = df.train.to_numpy() if "train" in df else np.zeros(len(df), bool)
        np.savez_compressed(CACHE / f"acs_{y}.npz", st=df.ST.astype(int).to_numpy(), serial=df.SERIALNO.astype(str).to_numpy(),
                            label=df.label.to_numpy(), agep=df.AGEP.astype(float).to_numpy(), occp=df.OCCP.astype(float).to_numpy(),
                            rep=reps.astype(np.float32), logit=logits.astype(np.float32), train=train)
        pred = logits.argmax(1)
        held = ~train
        meta["acc"][str(y)] = {s: float((pred[held & (df.ST.to_numpy() == FIPS[s])] == df.label.to_numpy()[held & (df.ST.to_numpy() == FIPS[s])]).mean()) for s in STATES}
        meta.setdefault("rows", {})[str(y)] = {s: int((df.ST.to_numpy() == FIPS[s]).sum()) for s in STATES}
        meta.setdefault("positive_rate", {})[str(y)] = {s: float(df.label.to_numpy()[df.ST.to_numpy() == FIPS[s]].mean()) for s in STATES}
        print(y, reps.shape, flush=True)
    (CACHE / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
