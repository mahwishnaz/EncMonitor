"""Window generator for experiments on cached representations (CPU only).

Copies the pool construction and window sampling of scripts/15_run_screening.py (MNIST, CIFAR-10) and
scripts/33_camelyon_windows.py (Camelyon17): same caches, same random streams, same seeds. The exact-kernel
MMD (GPU) and the accuracy fields are not computed. Instead of the statistics alone, each window returns
the S x S Gram matrix of the site residuals (and optionally the pooled residual), from which D_mix, D_site,
H, A_s and any clipped variant follow exactly.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from encmonitor.embeddings import EmbeddingStandardizer  # noqa: E402
from encmonitor.rff import RFFMap  # noqa: E402

CONFIGS = {"mnist": "configs/main.yaml", "cifar10": "configs/main.yaml",
           "camelyon17": "configs/camelyon.yaml"}
CACHES = {"mnist": "outputs/main/mnist", "cifar10": "outputs/main/cifar10",
          "camelyon17": "outputs/camelyon/main/camelyon17"}


def _load15():
    spec = importlib.util.spec_from_file_location("screening15", ROOT / "scripts" / "15_run_screening.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M15 = _load15()


def config(ds: str) -> dict:
    return yaml.safe_load(open(ROOT / CONFIGS[ds]))


def conditions(ds: str) -> list[dict]:
    return M15.conditions(config(ds))


def label_prior_idx(labels, target, q, sign, rng):
    """Same random calls as label_prior_pool() in script 15 / label_prior_idx() in script 33."""
    classes = np.unique(labels)
    p = np.array([np.mean(labels == c) for c in classes])
    onehot = (classes == target).astype(float)
    if sign > 0:
        p_new = (1 - q) * p + q * onehot
    else:
        rest = p * (1 - onehot)
        p_new = (1 - q) * p + q * rest / rest.sum()
    counts = rng.multinomial(labels.size, p_new)
    idx = np.concatenate([rng.choice(np.flatnonzero(labels == c), k, replace=True)
                          for c, k in zip(classes, counts) if k > 0])
    rng.shuffle(idx)
    return idx


class SeedContext:
    def __init__(self, ds: str, seed: int):
        self.ds, self.seed, self.cfg = ds, int(seed), config(ds)
        d = ROOT / CACHES[ds] / f"seed_{seed}"
        self.S = int(self.cfg["sites"])
        self.n = int(self.cfg["window_size"])
        self.alpha = np.ones(self.S) / self.S
        self.prep = EmbeddingStandardizer.load(d / "preprocessor.npz")
        self.rff = RFFMap.load(str(d / "rff_map.npz"))
        bz = np.load(d / "baselines.npz")
        self.base = np.stack([np.asarray(bz[f"site_{s:03d}"], dtype=np.float64) for s in range(self.S)])
        mc = np.load(d / "monitor_clean.npz")
        self.clean_raw = [mc[f"site_{s:03d}"].astype(np.float64) for s in range(self.S)]
        self.labels = [mc[f"labels_{s:03d}"] for s in range(self.S)]
        self.pieces = np.load(d / "pieces.npz")
        if ds == "camelyon17":
            self.target = int(self.cfg["label_prior_target"])
        else:   # script 15: class with the largest minimum count over sites 0..2
            classes = np.unique(np.concatenate(self.labels))
            self.target = int(classes[np.argmax([min(np.sum(self.labels[s] == c) for s in range(3)) for c in classes])])
        v = np.random.default_rng([self.seed, 7]).standard_normal(self.clean_raw[0].shape[1])
        self.v = v / np.linalg.norm(v)
        self.clean_feats = [self.feats(x) for x in self.clean_raw]

    def feats(self, raw: np.ndarray) -> np.ndarray:
        return self.rff.transform(self.prep.transform(raw))

    def site_feats(self, kind: str, family: str, severity: float, f: float, s: int, geometry: str = "") -> np.ndarray:
        """Shifted pool of site s with direction factor f (f = 0: clean pool)."""
        if f == 0:
            return self.clean_feats[s]
        if kind == "image":
            return self.feats(self.pieces[f"{family}|{severity:g}|{f:+g}|site_{s:03d}"].astype(np.float64))
        if kind == "label_prior":
            q = severity * abs(f)
            rng = np.random.default_rng([self.seed, 3, s, int(round(q * 1000)), int(f > 0)])
            return self.feats(self.clean_raw[s][label_prior_idx(self.labels[s], self.target, q, np.sign(f), rng)])
        if kind == "vector":
            return self.feats(self.clean_raw[s] + f * severity * self.v * self.prep.scale)
        if kind == "exchange":
            h = int(self.cfg["exchange_sources"][geometry][s])
            q = severity * abs(f)
            rng = np.random.default_rng([self.seed, 5, s, int(round(q * 1000))])
            N = self.clean_raw[s].shape[0]
            k = int(round(q * N))
            keep = rng.choice(N, N - k, replace=False)
            M = self.clean_raw[h].shape[0]
            take = rng.choice(M, k, replace=k > M)
            return self.feats(np.concatenate([self.clean_raw[s][keep], self.clean_raw[h][take]]))
        raise ValueError(kind)

    def condition_feats(self, c: dict) -> list[np.ndarray]:
        return [self.site_feats(c["kind"], c["family"], c["severity"], float(c["directions"][s]), s, c.get("geometry", ""))
                for s in range(self.S)]

    def run(self, feats: list[np.ndarray], windows: int, wseed: int, keep_pooled: bool = False) -> dict:
        """Window sampling identical to scripts 15/33: rng = default_rng(wseed); per window, per site,
        idx = rng.choice(N, n, replace=N < n)."""
        rng = np.random.default_rng(wseed)
        gram = np.empty((windows, self.S, self.S))
        pooled = np.empty((windows, self.base.shape[1])) if keep_pooled else None
        for w in range(windows):
            means = np.empty_like(self.base)
            for s in range(self.S):
                N = feats[s].shape[0]
                means[s] = feats[s][rng.choice(N, self.n, replace=N < self.n)].mean(axis=0)
            res = means - self.base
            gram[w] = res @ res.T
            if keep_pooled:
                pooled[w] = self.alpha @ res
        return {"gram": gram, "pooled": pooled}


def stats_from_gram(gram: np.ndarray, alpha: np.ndarray) -> dict:
    """D_mix, D_site, H, site energies and A_s from Gram matrices (W, S, S)."""
    energy = np.einsum("wss->ws", gram)
    d_mix = np.einsum("s,wst,t->w", alpha, gram, alpha)
    d_site = energy @ alpha
    return {"d_mix": d_mix, "d_site": d_site, "heterogeneity": d_site - d_mix, "site_energy": energy,
            "contributions": energy * alpha}


def clip_gram(gram: np.ndarray, R: float) -> np.ndarray:
    """Gram matrix after each site clips its residual to norm R."""
    norms = np.sqrt(np.einsum("wss->ws", gram))
    c = np.minimum(1.0, R / np.maximum(norms, 1e-300))
    return gram * c[:, :, None] * c[:, None, :]
