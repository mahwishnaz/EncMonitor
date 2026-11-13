#!/usr/bin/env python3
"""Exact-kernel compute step (GPU for the exact kernel, CPU for RFF).

For every dataset, seed and window: exact RKHS Gram matrix of the site residuals (V-statistic),
unbiased per-site MMD^2_U, and RFF Gram matrices for m = 32, 64, 128, 256. Cached per seed in windows/.

Usage (bundle root):
  for i in 1 2 3 4 5 6 7 8; do CUDA_VISIBLE_DEVICES=0 python experiments/exact_site_mmd/run.py && break; sleep 15; done
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
from rx_windows import SeedContext, conditions, label_prior_idx  # noqa: E402
from encmonitor.rff import RFFMap  # noqa: E402

DATASETS = ["mnist", "cifar10", "camelyon17"]
MS = [32, 64, 128, 256]
OUT = HERE / "windows"


class RawContext(SeedContext):
    """SeedContext that returns the standardized pools (needed by the exact kernel)."""

    def raw_pool(self, c: dict, s: int) -> np.ndarray:
        f = float(c["directions"][s])
        if f == 0:
            return self.clean_raw[s]
        kind, sev = c["kind"], c["severity"]
        if kind == "image":
            return self.pieces[f"{c['family']}|{sev:g}|{f:+g}|site_{s:03d}"].astype(np.float64)
        if kind == "label_prior":
            q = sev * abs(f)
            rng = np.random.default_rng([self.seed, 3, s, int(round(q * 1000)), int(f > 0)])
            return self.clean_raw[s][label_prior_idx(self.labels[s], self.target, q, np.sign(f), rng)]
        if kind == "vector":
            return self.clean_raw[s] + f * sev * self.v * self.prep.scale
        if kind == "exchange":
            h = int(self.cfg["exchange_sources"][c["geometry"]][s])
            q = sev * abs(f)
            rng = np.random.default_rng([self.seed, 5, s, int(round(q * 1000))])
            N = self.clean_raw[s].shape[0]
            k = int(round(q * N))
            keep = rng.choice(N, N - k, replace=False)
            M = self.clean_raw[h].shape[0]
            take = rng.choice(M, k, replace=k > M)
            return np.concatenate([self.clean_raw[s][keep], self.clean_raw[h][take]])
        raise ValueError(kind)


class ExactGram:
    def __init__(self, ref_std: list[np.ndarray], sigma: float, dev):
        self.dev, self.gamma, self.S = dev, 1.0 / (2.0 * sigma * sigma), len(ref_std)
        self.R = torch.as_tensor(np.concatenate(ref_std), dtype=torch.float64, device=dev)
        sizes = [r.shape[0] for r in ref_std]
        lab = np.repeat(np.arange(self.S), sizes)
        self.IR = torch.zeros(self.R.shape[0], self.S, dtype=torch.float64, device=dev)
        self.IR[torch.arange(self.R.shape[0]), torch.as_tensor(lab, device=dev)] = 1.0
        self.nR = torch.as_tensor(sizes, dtype=torch.float64, device=dev)
        self.rr = (self.R * self.R).sum(1)
        # reference-reference block means, in chunks
        sums = torch.zeros(self.S, self.S, dtype=torch.float64, device=dev)
        for a in range(0, self.R.shape[0], 4096):
            K = self.k(self.R[a:a + 4096], self.rr[a:a + 4096], self.R, self.rr)
            sums += self.IR[a:a + 4096].T @ K @ self.IR
        self.KRR = sums / (self.nR[:, None] * self.nR[None, :])
        self.KRR_off = (torch.diagonal(sums) - self.nR) / (self.nR * (self.nR - 1))

    def k(self, a, aa, b, bb):
        return torch.exp(-self.gamma * (aa[:, None] + bb[None, :] - 2.0 * a @ b.T).clamp_(min=0.0))

    def __call__(self, cur: np.ndarray, n: int):
        """cur: (S*n, d) window samples in site order. Returns (G_V (S,S), mmd2_U (S,))."""
        C = torch.as_tensor(cur, dtype=torch.float64, device=self.dev)
        cc = (C * C).sum(1)
        IC = torch.zeros(C.shape[0], self.S, dtype=torch.float64, device=self.dev)
        IC[torch.arange(C.shape[0]), torch.arange(self.S, device=self.dev).repeat_interleave(n)] = 1.0
        Scc = IC.T @ self.k(C, cc, C, cc) @ IC
        KCC = Scc / (n * n)
        KCR = (IC.T @ self.k(C, cc, self.R, self.rr) @ self.IR) / (n * self.nR[None, :])
        G = KCC - KCR - KCR.T + self.KRR
        KCC_off = (torch.diagonal(Scc) - n) / (n * (n - 1))
        U = KCC_off - 2.0 * torch.diagonal(KCR) + self.KRR_off
        return G.cpu().numpy(), U.cpu().numpy()


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available -- stop (device rule: exact kernel on the GPU).")
    dev = torch.device("cuda")
    OUT.mkdir(exist_ok=True)
    for ds in DATASETS:
        for seed in SeedContext(ds, 2026).cfg["partition_seeds"]:
            path = OUT / f"{ds}_seed{seed}.npz"
            if path.exists():
                print(f"[{ds} {seed}] cached, skipping", flush=True)
                continue
            t0 = time.time()
            ctx = RawContext(ds, int(seed))
            cfg, S, n = ctx.cfg, ctx.S, ctx.n
            d = ROOT_DIR(ds, seed)
            bc = np.load(d / "baseline_clean.npz")
            ref_std = [ctx.prep.transform(bc[f"site_{s:03d}"].astype(np.float64)) for s in range(S)]
            maps = {m: (ctx.rff if m == 64 else RFFMap.fit(ctx.rff.input_dim, m, ctx.rff.bandwidth, 1729)) for m in MS}
            bases = {m: np.stack([maps[m].mean_sketch(r) for r in ref_std]) for m in MS}
            base_check = float(np.abs(bases[64] - ctx.base).max())
            ex = ExactGram(ref_std, ctx.rff.bandwidth, dev)
            jobs = [({"condition": "clean_calibration", "directions": [0] * S}, int(cfg["calibration_windows"]), seed),
                    ({"condition": "clean_heldout", "directions": [0] * S}, int(cfg["heldout_windows"]), seed + 1000)]
            jobs += [(c, int(cfg["windows_per_condition"]), seed + 2000) for c in conditions(ds)]
            names, GV, GU, GR = [], [], [], {m: [] for m in MS}
            for c, W, wseed in jobs:
                std = [ctx.prep.transform(ctx.raw_pool(c, s)) for s in range(S)]
                feats = {m: [maps[m].transform(z) for z in std] for m in MS}
                rng = np.random.default_rng(wseed)
                for w in range(W):
                    idx = []
                    for s in range(S):
                        N = std[s].shape[0]
                        idx.append(rng.choice(N, n, replace=N < n))
                    g, u = ex(np.concatenate([std[s][idx[s]] for s in range(S)]), n)
                    GV.append(g); GU.append(u); names.append(c["condition"])
                    for m in MS:
                        res = np.stack([feats[m][s][idx[s]].mean(axis=0) for s in range(S)]) - bases[m]
                        GR[m].append(res @ res.T)
            np.savez_compressed(path, condition=np.array(names), exact_v=np.stack(GV), exact_u_diag=np.stack(GU),
                                **{f"rff{m}": np.stack(GR[m]) for m in MS},
                                ref_sizes=np.array([r.shape[0] for r in ref_std]), bandwidth=ctx.rff.bandwidth,
                                base_check=base_check)
            print(f"[{ds} {seed}] {len(names)} windows, base check {base_check:.2e}, {time.time() - t0:.0f}s", flush=True)
    return 0


def ROOT_DIR(ds, seed):
    from rx_windows import CACHES, ROOT
    return ROOT / CACHES[ds] / f"seed_{seed}"


if __name__ == "__main__":
    raise SystemExit(main())
