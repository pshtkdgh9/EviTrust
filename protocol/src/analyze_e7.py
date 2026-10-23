"""E7 summary: CAT vs EviTrust-U / EviTrust-H on CiaoDVD and wiki-RfA (identical test pairs).

    python analyze_e7.py --root ../results_v2/e7 --out ../results_v2/e7/E7_summary.md
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

import e3_negsampling as N3

REG_KO = {"a_cat_own": "(a) CAT 프로토콜", "b_degree_matched": "(b) 차수 일치", "c_twohop_time": "(c) 2-hop+시간",
          "d_oppose": "(d) 실제 반대표"}
METHODS = [("cat", "CAT (원본)"), ("uniform_only", "EviTrust-U"), ("full_m311af", "EviTrust-H")]
DS_KO = {"ciaodvd": "CiaoDVD", "wikirfa": "wiki-RfA"}


def load(d: Path):
    out = {}
    for key, _ in METHODS:
        pat = "cat_s*.json" if key == "cat" else f"evitrust_{key}_s*.json"
        js = {int(p.stem.rsplit("_s", 1)[1]): json.loads(p.read_text()) for p in d.glob(pat)}
        sc = {}
        for s in js:
            f = d / (f"scores_cat_s{s}.npz" if key == "cat" else f"scores_{key}_s{s}.npz")
            if f.exists():
                sc[s] = dict(np.load(f))
        out[key] = (js, sc)
    return out


def fmt(v):
    return f"{st.mean(v):.4f} ± {st.stdev(v):.4f}" if len(v) > 1 else (f"{v[0]:.4f}" if v else "—")


def paired_ci(sp_e, sn_e, sp_c, sn_c, seed, B=1000):
    rng = np.random.default_rng(seed); n, m = len(sp_e), len(sn_e); d = []
    for _ in range(B):
        i = rng.integers(0, n, n); j = rng.integers(0, m, m)
        d.append(N3.fast_ap(sp_e[i], sn_e[j]) - N3.fast_ap(sp_c[i], sn_c[j]))
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../results_v2/e7")
    ap.add_argument("--out", default="../results_v2/e7/E7_summary.md")
    a = ap.parse_args()
    root = Path(a.root)
    L = ["# E7 — CiaoDVD·wiki-RfA에서 CAT vs EviTrust (같은 분할·같은 테스트 쌍, seed 42·43·44)\n"]
    for ds in ("ciaodvd", "wikirfa"):
        d = root / ds
        if not d.exists():
            continue
        R = load(d)
        cat_js = R["cat"][0]
        if not cat_js:
            continue
        regs = [r for r in REG_KO if r in next(iter(cat_js.values()))["regimes"]]
        for subset in ("observed", "all"):
            L.append(f"## {DS_KO[ds]} — {subset} 부분집합 (AP, seed 평균 ± 표준편차)\n")
            hdr = "| 방법 | seed | " + " | ".join(f"{REG_KO[r]} AP" for r in regs) + " | (a) AUC | (a) MRR | 학습 시간(초) |"
            L += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
            for key, label in METHODS:
                js = R[key][0]
                if not js:
                    continue
                cells = []
                for r in regs:
                    v = [j["regimes"][r][subset]["AP"] for j in js.values() if subset in j["regimes"].get(r, {})]
                    cells.append(fmt(v))
                auc = [j["regimes"]["a_cat_own"][subset]["AUC"] for j in js.values()]
                mrr = [j["regimes"]["a_cat_own"][subset]["MRR"] for j in js.values()]
                tt = [j.get("train_s", j.get("runtime_s", float("nan"))) for j in js.values()]
                L.append(f"| {label} | {len(js)} | " + " | ".join(cells) + f" | {fmt(auc)} | {fmt(mrr)} | {st.mean(tt):.0f} |")
            ch = {r: next(iter(cat_js.values()))["regimes"][r].get(subset, {}).get("prevalence") for r in regs}
            L.append("| 우연 수준 (양성 비율) | | " + " | ".join(f"{ch[r]:.3f}" if ch[r] is not None else "—" for r in regs) + " | 0.5 | | |")
            n_pos = next(iter(cat_js.values()))["regimes"]["a_cat_own"][subset]["n_pos"]
            L.append(f"\n테스트 양성 {n_pos:,}개. 음성 수: " + ", ".join(
                f"{REG_KO[r]} {next(iter(cat_js.values()))['regimes'][r][subset]['n_neg']:,}" for r in regs if subset in next(iter(cat_js.values()))['regimes'][r]) + "\n")
        # paired bootstrap vs CAT (observed and all)
        L.append(f"## {DS_KO[ds]} — 쌍대 bootstrap: EviTrust − CAT, AP 차이 (95% CI, B=1000)\n")
        L.append("| 방법 | 부분집합 | " + " | ".join(REG_KO[r] for r in regs) + " |")
        L.append("|---|---|" + "---|" * len(regs))
        cat_sc = R["cat"][1]
        for key, label in METHODS[1:]:
            js, sc = R[key]
            for subset in ("observed", "all"):
                cells = []
                for r in regs:
                    diffs, cis = [], []
                    for s in sorted(set(sc) & set(cat_sc)):
                        E, Cc = sc[s], cat_sc[s]
                        pm = Cc["observed"] if subset == "observed" else np.ones(len(Cc["pos"]), bool)
                        nm = Cc[f"{r}_obs"] if subset == "observed" else np.ones(len(Cc[r]), bool)
                        if pm.sum() == 0 or nm.sum() == 0:
                            continue
                        args = (E["pos"][pm], E[r][nm], Cc["pos"][pm], Cc[r][nm])
                        diffs.append(N3.fast_ap(args[0], args[1]) - N3.fast_ap(args[2], args[3]))
                        cis.append(paired_ci(*args, seed=s))
                    if diffs:
                        npos = sum(lo > 0 for lo, _ in cis); nneg = sum(hi < 0 for _, hi in cis)
                        cells.append(f"{st.mean(diffs):+.4f} (CI>0 {npos}/{len(cis)}" + (f", CI<0 {nneg}" if nneg else "") + ")")
                    else:
                        cells.append("—")
                L.append(f"| {label} | {subset} | " + " | ".join(cells) + " |")
        L.append("")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
