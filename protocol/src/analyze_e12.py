"""E12 (review v2 N1) - generalization of EviTrust-V to hard negatives it was not trained on.

(1) leave-one-generator-out: V-c (rho 3:1:0, selected on val (a)+(b)) is evaluated on (c);
    V-b (rho 3:0:1, selected on val (a)+(c)) is evaluated on (b).
(2) unseen generators for every model: (e) co-review, (f) active degree-matched.
EviTrust-V is re-selected here with the pre-registered rule (prereg §E11) from the E12 runs.

    python analyze_e12.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import numpy as np

import analyze_e8 as A
import e3_negsampling as N3

R = A.R / "e12"
SEEDS = (42, 43, 44)
DS = [("epinions", "Epinions"), ("ciaodvd", "CiaoDVD"), ("wikirfa", "wiki-RfA")]
BASE = [("cat", "CAT"), ("heur_lr", "Heuristic-LR"), ("guardian", "Guardian"), ("trustguard", "TrustGuard"), ("tgn", "TGN")]
REGS = [("a_cat_own", "(a)"), ("b_degree_matched", "(b)"), ("c_twohop_time", "(c)"), ("d_oppose", "(d)"),
        ("e_coreview", "(e) co-review*"), ("f_active_dm", "(f) active-dm*")]


def load(d, name):
    return {s: dict(np.load(d / f"scores_{name}_s{s}.npz")) for s in SEEDS if (d / f"scores_{name}_s{s}.npz").exists()}


def main():
    out = {}
    L = ["# E12 (N1) — 학습하지 않은 난이도 음성에서의 일반화\n",
         "(*) = 어떤 모델도 학습·선택에 쓰지 않은 음성. observed 부분집합 AP, seed 3개 평균.\n"]
    for ds, dsn in DS:
        d = R / ds
        M = {m: load(d, m) for m, _ in BASE}
        M["U"] = load(d, "uniform_only")
        # pre-registered V rule on the E12 runs
        V, chosen = {}, {}
        for s in SEEDS:
            crit = {}
            for tag in ("uniform_only", "full_m311mean", "full"):
                va = json.loads((d / f"evitrust_{tag}_s{s}.json").read_text())["selected"]["val_ap"]
                crit[tag] = (va["a"] + va["b"] + va["c"]) / 3
            chosen[s] = max(crit, key=crit.get); V[s] = dict(np.load(d / f"scores_{chosen[s]}_s{s}.npz"))
        M["V"] = V
        M["V-c"] = load(d, "full_logo_noc"); M["V-b"] = load(d, "full_logo_nob")
        cat = M["cat"]
        regs = [(r, k) for r, k in REGS if r in cat[42]]
        L.append(f"## {dsn}  (V가 고른 ρ: {', '.join(chosen.values())})\n")
        L.append("| 방법 | " + " | ".join(k for _, k in regs) + " |")
        L.append("|---|" + "---|" * len(regs))
        tab = {}
        for name, lab in [(m, n) for m, n in BASE] + [("U", "EviTrust-U"), ("V", "EviTrust-V"), ("V-c", "EviTrust-V − (c) 생성기 제외"), ("V-b", "EviTrust-V − (b) 생성기 제외")]:
            if not M.get(name):
                continue
            tab[name] = {r: st.mean(A.metric(M[name][s], cat, s, r, "observed")["AP"] for s in SEEDS) for r, _ in regs}
            L.append(f"| {lab} | " + " | ".join(f"{tab[name][r]:.4f}" for r, _ in regs) + " |")
        best = {r: max(((m, tab[m][r]) for m, _ in BASE), key=lambda x: x[1]) for r, _ in regs}
        L.append("\n최강 기준선: " + ", ".join(f"{k} {dict(BASE)[best[r][0]]} {best[r][1]:.4f}" for r, k in regs) + "\n")
        # paired tests
        L.append("| 비교 (observed, 쌍대 bootstrap) | 음성 | 차이 | CI>0 / CI<0 seed 수 |")
        L.append("|---|---|---|---|")
        tests = [("V-c", "c_twohop_time", "held-out (c)"), ("V-b", "b_degree_matched", "held-out (b)"),
                 ("V", "e_coreview", "unseen (e)"), ("V", "f_active_dm", "unseen (f)"),
                 ("U", "e_coreview", "unseen (e)"), ("U", "f_active_dm", "unseen (f)")]
        res = {}
        for m, r, lab in tests:
            bm = best[r][0]
            rr = [A.paired(M[m][s], M[bm][s], cat, s, r, "observed") for s in SEEDS]
            mu = st.mean(x[0] for x in rr); up = sum(x[1] > 0 for x in rr); dn = sum(x[2] < 0 for x in rr)
            res[f"{m}|{r}"] = dict(vs=bm, diff=mu, up=int(up), dn=int(dn))
            L.append(f"| {m} − {dict(BASE)[bm]} | {lab} | {mu:+.4f} | {up}/{dn} |")
        # held-out cost relative to the full V
        L.append(f"\n생성기 제외의 비용: (c)에서 V−c {tab['V-c']['c_twohop_time']:.4f} vs V {tab['V']['c_twohop_time']:.4f} "
                 f"(Δ {tab['V-c']['c_twohop_time'] - tab['V']['c_twohop_time']:+.4f}); (b)에서 V−b {tab['V-b']['b_degree_matched']:.4f} vs V "
                 f"{tab['V']['b_degree_matched']:.4f} (Δ {tab['V-b']['b_degree_matched'] - tab['V']['b_degree_matched']:+.4f})\n")
        out[ds] = dict(table=tab, best={k: [v[0], v[1]] for k, v in best.items()}, paired=res, chosen=chosen)
    (R / "E12_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump(out, open(R / "E12_summary.json", "w"), indent=1)
    print("wrote", R / "E12_summary.md")


if __name__ == "__main__":
    main()
