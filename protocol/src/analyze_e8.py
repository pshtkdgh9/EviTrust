"""E8 (M2) summary: all methods on all datasets, recomputed from per-pair score files.

    python analyze_e8.py --out ../results_v2/e8/E8_summary.md

Score files (identical test pairs per dataset and seed):
    Epinions : results_v2/e6/scores_cat_s*.npz, scores_uniform_only_s*.npz, scores_full_m311af_s*.npz
    others   : results_v2/e7/<ds>/scores_{cat,uniform_only,full_m311af}_s*.npz
    baselines: results_v2/e8/<ds>/scores_{heur_lr,guardian,trustguard,tgn}_s*.npz
Negative observed masks are taken from the CAT file of the same seed (same negatives).
"""
from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

import numpy as np

import e3_negsampling as N3

R = Path(__file__).resolve().parents[1] / "results_v2"
METHODS = [("cat", "CAT"), ("heur_lr", "Heuristic-LR"), ("guardian", "Guardian"), ("trustguard", "TrustGuard"),
           ("tgn", "TGN"), ("cat_h", "CAT-H (mixed neg.)"), ("tgn_h", "TGN-H (mixed neg.)"),
           ("uniform_only", "EviTrust-U"), ("full_m311af", "EviTrust-H")]
DS = [("epinions", "Epinions"), ("ciaodvd", "CiaoDVD"), ("wikirfa", "wiki-RfA")]
REG = [("a_cat_own", "(a)"), ("b_degree_matched", "(b)"), ("c_twohop_time", "(c)"), ("d_oppose", "(d)")]


def path(ds, m, s):
    if m in ("cat_h", "tgn_h"):
        return R / "e9" / ds / f"scores_{m}_s{s}.npz"
    if m in ("heur_lr", "guardian", "trustguard", "tgn"):
        return R / "e8" / ds / f"scores_{m}_s{s}.npz"
    return (R / "e6" if ds == "epinions" else R / "e7" / ds) / f"scores_{m}_s{s}.npz"


def load(ds):
    out = {}
    for m, _ in METHODS:
        d = {s: dict(np.load(path(ds, m, s))) for s in (42, 43, 44) if path(ds, m, s).exists()}
        if d:
            out[m] = d
    return out


def masks(cat, s, reg, subset):
    C = cat[s]
    pm = C["observed"] if subset == "observed" else np.ones(len(C["pos"]), bool)
    nm = C[f"{reg}_obs"] if subset == "observed" else np.ones(len(C[reg]), bool)
    return pm, nm


def metric(E, cat, s, reg, subset):
    pm, nm = masks(cat, s, reg, subset)
    if pm.sum() == 0 or nm.sum() == 0:
        return None
    m, _ = N3.metrics(E["pos"][pm], E[reg][nm], np.random.default_rng(2025 + 7))
    return m


def paired(A, Bm, cat, s, reg, subset, B=1000):
    pm, nm = masks(cat, s, reg, subset)
    a_p, a_n, b_p, b_n = A["pos"][pm], A[reg][nm], Bm["pos"][pm], Bm[reg][nm]
    rng = np.random.default_rng(s); n, m = len(a_p), len(a_n); d = []
    for _ in range(B):
        i = rng.integers(0, n, n); j = rng.integers(0, m, m)
        d.append(N3.fast_ap(a_p[i], a_n[j]) - N3.fast_ap(b_p[i], b_n[j]))
    return N3.fast_ap(a_p, a_n) - N3.fast_ap(b_p, b_n), np.percentile(d, 2.5), np.percentile(d, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(R / "e8" / "E8_summary.md"))
    a = ap.parse_args()
    L = ["# E8 (M2) — 기준선 확충: 7개 방법 × 3개 데이터셋 (같은 분할·같은 테스트 쌍, seed 42·43·44)\n",
         "모든 수치는 쌍별 점수 파일에서 같은 코드로 다시 계산. (a) CAT 프로토콜, (b) 차수 일치, (c) 2-hop+시간, (d) wiki-RfA 실제 반대표.\n"]
    for ds, dsn in DS:
        D = load(ds)
        if "cat" not in D:
            continue
        cat = D["cat"]; seeds = sorted(cat)
        regs = [(r, k) for r, k in REG if r in cat[seeds[0]]]
        for subset in (["observed", "all"] if ds == "wikirfa" else ["observed"]):
            L.append(f"## {dsn} — {subset} (AP 평균 ± 표준편차; (a)는 AUC·MRR 병기)\n")
            hdr = "| 방법 | seed | " + " | ".join(f"{k} AP" for _, k in regs) + " | (a) AUC | (a) MRR |"
            L += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
            best = {}
            rows = []
            for m, mn in METHODS:
                if m not in D:
                    continue
                cells = []
                for r, _ in regs:
                    v = [metric(D[m][s], cat, s, r, subset)["AP"] for s in sorted(D[m]) if s in cat]
                    mu = st.mean(v); best[r] = max(best.get(r, -1), mu)
                    cells.append((mu, st.stdev(v) if len(v) > 1 else 0.0))
                aa = [metric(D[m][s], cat, s, "a_cat_own", subset) for s in sorted(D[m]) if s in cat]
                rows.append((mn, len(D[m]), cells, st.mean(x["AUC"] for x in aa), st.mean(x["MRR"] for x in aa)))
            for mn, ns, cells, auc, mrr in rows:
                cs = [(f"**{mu:.4f}**" if abs(mu - best[r]) < 1e-12 else f"{mu:.4f}") + f" ± {sd:.4f}"
                      for (mu, sd), (r, _) in zip(cells, regs)]
                L.append(f"| {mn} | {ns} | " + " | ".join(cs) + f" | {auc:.4f} | {mrr:.4f} |")
            pm, nm = masks(cat, seeds[0], "a_cat_own", subset)
            L.append(f"| 우연 수준 | | " + " | ".join(
                f"{masks(cat, seeds[0], r, subset)[0].sum() / (masks(cat, seeds[0], r, subset)[0].sum() + masks(cat, seeds[0], r, subset)[1].sum()):.3f}"
                for r, _ in regs) + " | 0.5 | |\n")
        # paired: EviTrust-U and -H vs every baseline
        subset = "observed"
        L.append(f"## {dsn} — 쌍대 bootstrap (observed): EviTrust − 기준선, AP 차이 평균 [95% CI가 0 초과인 seed 수/3]\n")
        L.append("| 비교 | " + " | ".join(k for _, k in regs) + " |")
        L.append("|---|" + "---|" * len(regs))
        for ev, evn in (("uniform_only", "EviTrust-U"), ("full_m311af", "EviTrust-H")):
            if ev not in D:
                continue
            for m, mn in METHODS:
                if m in ("uniform_only", "full_m311af") or m not in D:
                    continue
                cells = []
                for r, _ in regs:
                    res = [paired(D[ev][s], D[m][s], cat, s, r, subset) for s in seeds if s in D[ev] and s in D[m]]
                    if not res:
                        cells.append("—"); continue
                    up = sum(lo > 0 for _, lo, _ in res); dn = sum(hi < 0 for _, _, hi in res)
                    cells.append(f"{st.mean(x for x, _, _ in res):+.4f} [{up}/{len(res)}" + (f", 열세 {dn}" if dn else "") + "]")
                L.append(f"| {evn} − {mn} | " + " | ".join(cells) + " |")
        L.append("")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
