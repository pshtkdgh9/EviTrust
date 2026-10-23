"""E11 (review M5) - EviTrust-V: mixture ratio rho chosen on validation only (pre-registered, prereg_v2 §E11).

For every dataset and seed, among rho in {1:0:0 (uniform_only), 3:1:1 (full_m311mean), 1:1:1 (full)} - each
already selected internally by the mean validation AP over (a, b, c) - pick the rho with the highest
mean validation AP over (a, b, c). Writes scores_evitrust_v_s<seed>.npz next to the other score files
and a report with the pre-registered verdicts C-E11a/b/c.

    python analyze_e11_m5.py
"""
from __future__ import annotations

import json
import shutil
import statistics as st
from pathlib import Path

import numpy as np

import analyze_e8 as A

R = A.R
CANDS = [("uniform_only", "1:0:0"), ("full_m311mean", "3:1:1"), ("full", "1:1:1")]
BASE = [("cat", "CAT"), ("heur_lr", "Heuristic-LR"), ("guardian", "Guardian"), ("trustguard", "TrustGuard"), ("tgn", "TGN")]


def ddir(ds):
    return R / "e6" if ds == "epinions" else R / "e7" / ds


def main():
    L = ["# E11 (M5) — EviTrust-V: 검증셋만으로 혼합 비율 ρ 선택 (사전 등록 prereg_v2 §E11)\n"]
    verdict = {"a": [], "b": [], "c": [], "close": []}
    for ds, dsn in A.DS:
        D = ddir(ds); chosen = {}
        L.append(f"## {dsn}\n")
        L.append("| seed | ρ 후보별 검증 AP 평균 (a,b,c) | 선택 ρ |")
        L.append("|---|---|---|")
        for s in (42, 43, 44):
            crit = {}
            for tag, rho in CANDS:
                f = D / f"evitrust_{tag}_s{s}.json"
                if not f.exists():
                    continue
                va = json.loads(f.read_text())["selected"]["val_ap"]
                crit[tag] = (va["a"] + va["b"] + va["c"]) / 3
            best = max(crit, key=crit.get); chosen[s] = best
            shutil.copy(D / f"scores_{best}_s{s}.npz", D / f"scores_evitrust_v_s{s}.npz")
            L.append(f"| {s} | " + ", ".join(f"{dict(CANDS)[t]}: {v:.4f}" for t, v in crit.items()) + f" | **{dict(CANDS)[best]}** |")
        # test results
        Dm = A.load(ds); cat = Dm["cat"]
        V = {s: dict(np.load(D / f"scores_evitrust_v_s{s}.npz")) for s in (42, 43, 44)}
        regs = [(r, k) for r, k in A.REG if r in cat[42]]
        L.append("\n| 방법 | " + " | ".join(f"{k} AP" for _, k in regs) + " |")
        L.append("|---|" + "---|" * len(regs))
        for name, src in [("EviTrust-V", V), ("EviTrust-H (사후 설계)", Dm.get("full_m311af")), ("EviTrust-U", Dm.get("uniform_only"))] + \
                         [(n, Dm.get(m)) for m, n in BASE]:
            if not src:
                continue
            cells = [f"{st.mean(A.metric(src[s], cat, s, r, 'observed')['AP'] for s in (42, 43, 44)):.4f}" for r, _ in regs]
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        # paired: V vs each baseline, V vs H
        L.append("\n쌍대 bootstrap (observed): EviTrust-V − 비교 대상, AP 차이 평균 [CI>0 seed 수 / CI<0 seed 수]\n")
        L.append("| 비교 | " + " | ".join(k for _, k in regs) + " |")
        L.append("|---|" + "---|" * len(regs))
        for m, mn in BASE + [("full_m311af", "EviTrust-H")]:
            if m not in Dm:
                continue
            cells = []
            for r, k in regs:
                res = [A.paired(V[s], Dm[m][s], cat, s, r, "observed") for s in (42, 43, 44)]
                up = sum(lo > 0 for _, lo, _ in res); dn = sum(hi < 0 for _, _, hi in res); mu = st.mean(x for x, _, _ in res)
                cells.append(f"{mu:+.4f} [{up}/{dn}]")
                if m == "cat" and k == "(a)":
                    verdict["a"].append((dsn, up == 3))
                if ds in ("epinions", "ciaodvd") and k in ("(b)", "(c)") and m != "full_m311af":
                    verdict[k[1]].append((dsn, mn, up == 3))
                if m == "full_m311af" and k in ("(a)", "(b)", "(c)"):
                    verdict["close"].append((dsn, k, abs(mu) <= 0.02))
            L.append(f"| V − {mn} | " + " | ".join(cells) + " |")
        L.append(f"\n선택된 ρ: {', '.join(f's{s} {dict(CANDS)[t]}' for s, t in chosen.items())}\n")
    L.append("## 사전 고정 주장 판정\n")
    a_ok = all(x for _, x in verdict["a"]); b_ok = all(x for *_, x in verdict["b"]); c_ok = all(x for *_, x in verdict["c"])
    close_ok = all(x for *_, x in verdict["close"])
    L.append(f"- C-E11a (세 데이터셋 (a)에서 CAT보다 유의하게 높음, 3/3): {'✅' if a_ok else '❌'} {verdict['a']}")
    L.append(f"- C-E11b (Epinions·CiaoDVD (b)(c)에서 모든 기준선보다 유의하게 높음, 3/3): {'✅' if (b_ok and c_ok) else '❌'} "
             f"실패: {[x for x in verdict['b'] + verdict['c'] if not x[-1]]}")
    L.append(f"- C-E11c (EviTrust-H와 |Δ| ≤ 0.02): {'✅' if close_ok else '❌'} 실패: {[x for x in verdict['close'] if not x[-1]]}")
    out = R / "e11" / "E11_M5_summary.md"; out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
