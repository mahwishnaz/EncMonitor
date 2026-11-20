#!/usr/bin/env python3
"""Figure of the RFF-dimension trade-off. The evaluation-key sizes of panel (d) are taken from
private_localany/key_sizes.json (each profile measured in a fresh process).

Sources: (a) exact_site_mmd/agreement.csv; (b) outputs/ablation/analysis/q095/ablation_summary.csv;
(c) outputs/ckks/cost_grid/cost_grid_summary.csv; (d) ciphertext bytes from outputs/ckks/cost_grid/ckks_profiles.csv,
key bytes from key_sizes.json. Writes fig_rff_tradeoff_r3.pdf into this folder.
Usage (from the repository root): python experiments/private_localany/fig_rff_r3.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import json  # noqa: E402

import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # four line colours (any distinct colours will do)
MS = [32, 64, 128, 256]
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e4e4", "grid.linewidth": 0.5, "lines.linewidth": 1.6, "lines.markersize": 4.5})

fig, ax = plt.subplots(1, 4, figsize=(11.5, 3.2))
# (a) like-for-like agreement
a = pd.read_csv(HERE.parent / "exact_site_mmd" / "agreement.csv")
for i, (ds, lab) in enumerate([("mnist", "MNIST"), ("cifar10", "CIFAR-10"), ("camelyon17", "Camelyon17")]):
    for sub, ls, mk in (("all", "-", "o"), ("clean_heldout", "--", "s")):
        g = a[(a.dataset == ds) & (a.subset == sub)].sort_values("m")
        ax[0].plot(g.m, g.pearson_site_energy_V, ls=ls, marker=mk, color=C[i],
                   label=f"{lab}, {'all windows' if sub == 'all' else 'clean windows'}")
ax[0].set_title("(a) site energy vs exact MMD$^2$", loc="left")
ax[0].set_ylabel("Pearson r")
# (b) hidden-drift rate on the natural cancellation conditions (base setting, S = 5)
b = pd.read_csv(ROOT / "outputs/ablation/analysis/q095/ablation_summary.csv")
base = b[(b.sites == 5) & (b.layer == "penultimate") & (b.window_size == 250) & (b.bandwidth_multiplier.astype(float) == 1.0)
         & (b.baseline == "site") & (b.weights == "uniform")]
conds = [("mnist", "contrast0.35__mixed", "MNIST: contrast 0.35, mixed"), ("mnist", "label_prior0.2__opposing", "MNIST: label prior 0.2, opp."),
         ("cifar10", "contrast0.35__opposing", "CIFAR-10: contrast 0.35, opp."), ("cifar10", "label_prior0.2__opposing", "CIFAR-10: label prior 0.2, opp.")]
for i, (ds, cn, lab) in enumerate(conds):
    g = base[(base.dataset == ds) & (base.condition == cn)].sort_values("rff_dim")
    ax[1].plot(g.rff_dim, g.tpr_hidden, marker="osD^"[i], ls=["-", "--", "-.", ":"][i], color=C[i], label=lab)
ax[1].set_title("(b) hidden-drift rate", loc="left"); ax[1].set_ylim(0, 1.05)
# (c) server time
c = pd.read_csv(ROOT / "outputs/ckks/cost_grid/cost_grid_summary.csv")
for i, S in enumerate((5, 10)):
    g = c[c.sites == S].sort_values("rff_dim")
    ax[2].errorbar(g.rff_dim, g.server_p50, yerr=[0 * g.server_p50, g.server_p95 - g.server_p50], marker="os"[i],
                   ls=["-", "--"][i], color=C[i], capsize=2, label=f"S = {S} (median, 95th pct.)")
ax[2].set_title("(c) server time per window (s)", loc="left"); ax[2].set_ylim(0, None)
# (d) communication
p = pd.read_csv(ROOT / "outputs/ckks/cost_grid/ckks_profiles.csv").sort_values("rff_dim")
ax[3].plot(p.rff_dim, 5 * p.bytes_residual_ciphertext / 1e6, marker="o", color=C[0], label="upload per window, S = 5")
ax[3].plot(p.rff_dim, 10 * p.bytes_residual_ciphertext / 1e6, marker="s", ls="--", color=C[1], label="upload per window, S = 10")
ks = pd.DataFrame([r for r in json.load(open(HERE / "key_sizes.json")) if r["key_switch"] == "hybrid"]).sort_values("m")
ax[3].plot(ks.m, ks.eval_keys_bytes / 1e6, marker="^", ls=":", color=C[2], label="evaluation keys (one-time)")
ax[3].set_title("(d) communication (MB)", loc="left")
for k in range(4):
    ax[k].set_xscale("log", base=2); ax[k].set_xticks(MS); ax[k].set_xticklabels(MS); ax[k].set_xlabel("RFF dimension m", labelpad=1)
    ax[k].legend(fontsize=6, frameon=False, loc="upper center", borderaxespad=2.2, bbox_to_anchor=(0.5, -0.24), ncol=1)
fig.tight_layout()
fig.savefig(HERE / "fig_rff_tradeoff_r3.pdf", bbox_inches="tight")
fig.savefig(HERE / "fig_rff_tradeoff_r3.png", dpi=110, bbox_inches="tight")
print("written")
