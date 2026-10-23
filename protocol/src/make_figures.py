"""N2 - paper figures.

    python make_figures.py --out ../../ICDE/paper

fig_tradeoff.pdf : CAT-protocol AP (a) vs mean hard-negative AP ((b)+(c))/2, Epinions and CiaoDVD,
                   observed pairs, mean of 3 seeds, all methods recomputed from per-pair score files.
fig_scale.pdf    : streaming EviTrust state on synthetic graphs (users vs update time, query time, memory).
Colour: three categorical slots of the dataviz reference palette (validated, all-pairs) - identity is
carried by direct labels, colour only groups the families.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_e8 as A

R = A.R
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
})


def mean_ap(D, m, cat, reg):
    return st.mean(A.metric(D[m][s], cat, s, reg, "observed")["AP"] for s in (42, 43, 44))


def v_scores(ds):
    d = R / "e6" if ds == "epinions" else R / "e7" / ds
    return {s: dict(np.load(d / f"scores_evitrust_v_s{s}.npz")) for s in (42, 43, 44)}


def tradeoff(out):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55))
    neural = [("cat", "CAT"), ("guardian", "Guardian"), ("trustguard", "TrustGuard"), ("tgn", "TGN"),
              ("cat_h", "CAT-H"), ("tgn_h", "TGN-H")]
    OFF = {"epinions": {"CAT": (4, 3), "Guardian": (4, -9), "TrustGuard": (-22, -13), "TGN": (4, 3), "CAT-H": (4, -9),
                        "TGN-H": (4, 3), "Heuristic-LR": (-50, -3), "U (1:0:0)": (-14, -12), "3:1:1": (4, 4),
                        "1:1:1": (-24, 4), "V": (-3, -12)},
           "ciaodvd": {"CAT": (-20, 6), "Guardian": (4, -3), "TrustGuard": (-22, -11), "TGN": (4, 3), "CAT-H": (4, -9),
                       "TGN-H": (4, 3), "Heuristic-LR": (-58, -3), "U (1:0:0)": (5, -3), "3:1:1": (4, 4),
                       "1:1:1": (-24, 4), "V": (-3, -12)}}
    for ax, (ds, dsn) in zip(axes, [("epinions", "Epinions"), ("ciaodvd", "CiaoDVD")]):
        offsets = OFF[ds]
        D = A.load(ds); cat = D["cat"]; D["evitrust_v"] = v_scores(ds)
        for f in ("full_m311mean", "full"):
            p = (R / "e6" if ds == "epinions" else R / "e7" / ds)
            D[f] = {s: dict(np.load(p / f"scores_{f}_s{s}.npz")) for s in (42, 43, 44)}
        xy = lambda m: (mean_ap(D, m, cat, "a_cat_own"),
                        (mean_ap(D, m, cat, "b_degree_matched") + mean_ap(D, m, cat, "c_twohop_time")) / 2)
        ax.grid(True, color=GRID, linewidth=0.5, zorder=0)
        # EviTrust: the rho curve (1:0:0 -> 3:1:1 -> 1:1:1) and the validation-selected V
        pts = [xy("uniform_only"), xy("full_m311mean"), xy("full")]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=BLUE, linewidth=1.5, zorder=2)
        for (x, y), lab in zip(pts, ["U (1:0:0)", "3:1:1", "1:1:1"]):
            ax.scatter([x], [y], s=26, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
            ax.annotate(lab, (x, y), xytext=offsets[lab], textcoords="offset points", fontsize=7, color=INK)
        vx, vy = xy("evitrust_v")
        ax.scatter([vx], [vy], s=110, marker="*", color=BLUE, edgecolor="white", linewidth=0.8, zorder=4)
        ax.annotate("V", (vx, vy), xytext=offsets["V"], textcoords="offset points", fontsize=7, color=INK, fontweight="bold")
        for m, lab in neural:
            if m not in D:
                continue
            x, y = xy(m); hollow = lab.endswith("-H")
            ax.scatter([x], [y], s=24, marker="s" if not hollow else "D", facecolor="white" if hollow else ORANGE,
                       edgecolor=ORANGE, linewidth=1.2, zorder=3)
            ax.annotate(lab, (x, y), xytext=offsets[lab], textcoords="offset points", fontsize=7, color=INK)
        x, y = xy("heur_lr")
        ax.scatter([x], [y], s=24, marker="^", color=AQUA, edgecolor="white", linewidth=1.0, zorder=3)
        ax.annotate("Heuristic-LR", (x, y), xytext=offsets["Heuristic-LR"], textcoords="offset points", fontsize=7, color=INK)
        ax.set_title(dsn, fontsize=8.5, color=INK)
        x0, x1 = ax.get_xlim(); ax.set_xlim(x0, x1 + (x1 - x0) * 0.08)
        ax.set_xlabel("AP, CAT protocol (a)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("AP, hard negatives ((b)+(c))/2")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=BLUE, marker="o", markersize=5, label="EviTrust (by negative ratio $\\rho$; $\\star$ = V)"),
         Line2D([], [], color=ORANGE, marker="s", linestyle="", markersize=5, label="neural baselines"),
         Line2D([], [], color=ORANGE, marker="D", markerfacecolor="white", linestyle="", markersize=5, label="neural, mixed negatives (control)"),
         Line2D([], [], color=AQUA, marker="^", linestyle="", markersize=5, label="Heuristic-LR")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "fig_tradeoff.pdf", bbox_inches="tight"); fig.savefig(out / "fig_tradeoff.png", dpi=200, bbox_inches="tight")


def scale(out):
    rows = json.loads((R / "e10" / "scale.json").read_text())
    N = np.array([r["users"] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.0))
    a = axes[0]
    a.plot(N, [r["update_trust"]["mean_us"] for r in rows], color=BLUE, marker="o", markersize=4, linewidth=1.5)
    a.set_ylabel("$\\mu$s per trust record"); a.set_title("update (mean)", fontsize=8.5, color=INK); a.set_ylim(0, 10)
    a = axes[1]
    a.plot(N, [r["query_edge_pair"]["mean_us"] for r in rows], color=BLUE, marker="o", markersize=4, linewidth=1.5, label="mean")
    a.plot(N, [r["query_edge_pair"]["p99_us"] for r in rows], color=ORANGE, marker="s", markersize=4, linewidth=1.5, label="p99")
    a.set_yscale("log"); a.set_ylabel("$\\mu$s per pair"); a.set_title("pair features (edge pairs)", fontsize=8.5, color=INK)
    a.annotate("mean", (N[-1], rows[-1]["query_edge_pair"]["mean_us"]), xytext=(-24, -11), textcoords="offset points", fontsize=7, color=INK)
    a.annotate("p99", (N[0], rows[0]["query_edge_pair"]["p99_us"]), xytext=(-2, 6), textcoords="offset points", fontsize=7, color=INK)
    a = axes[2]
    a.plot(N, [r["state_rss_mb"] / 1024 for r in rows], color=BLUE, marker="o", markersize=4, linewidth=1.5)
    a.plot(N, [r["dense_state_gb_required"] for r in rows], color=ORANGE, marker="s", markersize=4, linewidth=1.5, linestyle="--")
    a.axhline(12, color=INK2, linewidth=0.8, linestyle=":")
    a.annotate("12 GB GPU", (N[0], 12), xytext=(0, 3), textcoords="offset points", fontsize=7, color=INK2)
    a.annotate("streaming state", (N[2], rows[2]["state_rss_mb"] / 1024), xytext=(-8, -13), textcoords="offset points", fontsize=7, color=INK)
    a.annotate("dense $n^2$", (N[-2], rows[-2]["dense_state_gb_required"]), xytext=(-40, 4), textcoords="offset points", fontsize=7, color=INK)
    a.set_yscale("log"); a.set_ylabel("GB"); a.set_title("memory", fontsize=8.5, color=INK)
    for a in axes:
        a.set_xscale("log"); a.set_xlabel("users"); a.grid(True, color=GRID, linewidth=0.5)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "fig_scale.pdf", bbox_inches="tight"); fig.savefig(out / "fig_scale.png", dpi=200, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "ICDE" / "paper"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    scale(out); tradeoff(out)
    print("wrote", out / "fig_tradeoff.pdf", out / "fig_scale.pdf")


if __name__ == "__main__":
    main()
