from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SitePartition:
    site_indices: tuple[np.ndarray, ...]
    class_proportions: np.ndarray  # [S,K]


def dirichlet_partition(
    labels: Sequence[int], n_sites: int, concentration: float, seed: int, min_per_site: int = 1
) -> SitePartition:
    """Persistent class-conditional Dirichlet partition with deterministic repair."""
    y = np.asarray(labels, dtype=np.int64)
    if y.ndim != 1 or y.size < n_sites:
        raise ValueError("labels must be a vector with at least n_sites entries")
    if n_sites < 2 or concentration <= 0:
        raise ValueError("n_sites >= 2 and concentration > 0 required")
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    buckets: list[list[int]] = [[] for _ in range(n_sites)]

    for cls in classes:
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        p = rng.dirichlet(np.full(n_sites, concentration, dtype=np.float64))
        counts = rng.multinomial(idx.size, p)
        cursor = 0
        for s, count in enumerate(counts):
            buckets[s].extend(idx[cursor : cursor + count].tolist())
            cursor += count

    # Repair empty/tiny sites by moving samples from the largest site.
    for s in range(n_sites):
        while len(buckets[s]) < min_per_site:
            donor = int(np.argmax([len(b) for b in buckets]))
            if donor == s or len(buckets[donor]) <= min_per_site:
                raise RuntimeError("Could not satisfy min_per_site")
            buckets[s].append(buckets[donor].pop())

    site_arrays = []
    props = np.zeros((n_sites, classes.size), dtype=np.float64)
    for s, bucket in enumerate(buckets):
        arr = np.asarray(bucket, dtype=np.int64)
        rng.shuffle(arr)
        site_arrays.append(arr)
        ys = y[arr]
        for j, cls in enumerate(classes):
            props[s, j] = np.mean(ys == cls)
    return SitePartition(tuple(site_arrays), props)


def sample_class_to_site_probabilities(
    n_classes: int, n_sites: int, concentration: float, seed: int
) -> np.ndarray:
    """Return P(site | class) with shape [K,S], reusable across data splits."""
    if n_classes < 2 or n_sites < 2 or concentration <= 0:
        raise ValueError("invalid partition dimensions or concentration")
    rng = np.random.default_rng(seed)
    return np.stack(
        [rng.dirichlet(np.full(n_sites, concentration, dtype=np.float64)) for _ in range(n_classes)],
        axis=0,
    )


def allocate_with_class_to_site(
    labels: Sequence[int], class_to_site: np.ndarray, seed: int, min_per_site: int = 1
) -> SitePartition:
    y = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(class_to_site, dtype=np.float64)
    classes = np.unique(y)
    if probs.ndim != 2 or probs.shape[0] <= int(classes.max()):
        raise ValueError("class_to_site does not cover all label values")
    n_sites = probs.shape[1]
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(n_sites)]
    for cls in classes:
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        p = probs[int(cls)] / probs[int(cls)].sum()
        counts = rng.multinomial(idx.size, p)
        cur = 0
        for site, count in enumerate(counts):
            buckets[site].extend(idx[cur : cur + count].tolist())
            cur += count
    for site in range(n_sites):
        while len(buckets[site]) < min_per_site:
            donor = int(np.argmax([len(x) for x in buckets]))
            if len(buckets[donor]) <= min_per_site:
                raise RuntimeError("unable to repair small site")
            buckets[site].append(buckets[donor].pop())
    props = np.zeros((n_sites, probs.shape[0]), dtype=np.float64)
    arrays: list[np.ndarray] = []
    for site, bucket in enumerate(buckets):
        arr = np.asarray(bucket, dtype=np.int64)
        rng.shuffle(arr)
        arrays.append(arr)
        for cls in classes:
            props[site, int(cls)] = np.mean(y[arr] == cls)
    return SitePartition(tuple(arrays), props)
