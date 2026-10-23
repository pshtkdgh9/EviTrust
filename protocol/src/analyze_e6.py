"""Paired comparison EviTrust vs CAT (E6) on identical test pairs, observed subset.

    python analyze_e6.py --dir ../results_v2/e6 --out ../results_v2/e6/E6_summary.md
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import e3_negsampling as N3

REG = ["a_cat_own", "b_degree_matched", "c_twohop_time", "a_uniform_e3"]
REG_KO = {"a_cat_own": "(a) CAT 자체 음성 [대표]", "b_degree_matched": "(b) 차수 일치", "c_twohop_time": "(c) 2-hop+시간",
          "a_uniform_e3": "(a') 무작위(E3 생성기)"}
B = 1000


def obs_mask_neg(d, reg):
    return d[f"{reg}_obs"] if f"{reg}_obs" in d else None


def paired(sp_e, sn_e, sp_c, sn_c, seed=0):
    rng = np.random.default_rng(seed)
    n, m = len(sp_e), len(sn_e)
    d_ap = []
    for _ in range(B):
        i = rng.integers(0, n, n); j = rng.integers(0, m, m)
        d_ap.append(N3.fast_ap(sp_e[i], sn_e[j]) - N3.fast_ap(sp_c[i], sn_c[j]))
    return float(np.percentile(d_ap, 2.5)), float(np.percentile(d_ap, 97.5)), float(np.mean(np.array(d_ap) <= 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="../results_v2/e6")
    ap.add_argument("--out", default="../results_v2/e6/E6_summary.md")
    ap.add_argument("--train-nodes-from", default=None)
    a = ap.parse_args()
    D = Path(a.dir)
    # observed masks for negatives are recomputed from the json n_neg is not enough -> use stored scores
    cat = {s: dict(np.load(D / f"scores_cat_s{s}.npz")) for s in (42, 43, 44) if (D / f"scores_cat_s{s}.npz").exists()}
    names = sorted({p.name.split("_s")[0].replace("evitrust_", "") for p in D.glob("evitrust_*_s*.json")})
    js = {nm: {int(p.stem.rsplit("_s", 1)[1]): json.loads(p.read_text()) for p in D.glob(f"evitrust_{nm}_s*.json")} for nm in names}
    e5 = {s: json.loads((D.parent / "e5" / f"cat_hardneg_s{s}.json").read_text()) for s in (42, 43, 44)}

    L = ["# E6 — EviTrust vs CAT (Epinions, CAT 분할·CAT 테스트 쌍 동일, observed 부분집합)\n"]
    L.append("## 1. 대표 성능 (seed 평균 ± 표준편차)\n")
    hdr = "| 방법 | seed 수 | " + " | ".join(f"{REG_KO[r]} AP" for r in REG[:3]) + " | (a) AUC | (a) MRR | Spearman(점수, log PA) |"
    L += [hdr, "|" + "---|" * (hdr.count("|") - 1)]

    def row(label, per_seed):
        ss = sorted(per_seed)
        cells = []
        for r in REG[:3]:
            v = [per_seed[s]["regimes"][r]["observed"]["AP"] for s in ss]
            cells.append(f"{st.mean(v):.4f} ± {st.stdev(v) if len(v) > 1 else 0:.4f}")
        auc = [per_seed[s]["regimes"]["a_cat_own"]["observed"]["AUC"] for s in ss]
        mrr = [per_seed[s]["regimes"]["a_cat_own"]["observed"]["MRR"] for s in ss]
        spr = [per_seed[s].get("spearman_score_logPA_observed", float("nan")) for s in ss]
        L.append(f"| {label} | {len(ss)} | " + " | ".join(cells) + f" | {st.mean(auc):.4f} | {st.mean(mrr):.4f} | {st.mean(spr):.3f} |")

    row("CAT (원본, E5)", e5)
    label = {"uniform_only": "**EviTrust-U** (CAT식 무작위 음성 학습)", "full_m311af": "**EviTrust-H** (혼합 난이도 3:1:1)",
             "full": "EviTrust 사전 등록 기본형 (1:1:1, 평균 기준)", "no_behavior": "절제: 행동 피처 제거 (1:1:1)",
             "no_emb": "절제: 임베딩 제거 (1:1:1)", "no_behavior_m311af": "절제: H − 행동 피처", "no_emb_m311af": "절제: H − 임베딩",
             "struct_only_m311af": "절제: H − 행동·임베딩 (구조만)", "full_w5af": "변형: 창 5개 (1:1:1)",
             "no_emb_u": "절제: U − 임베딩", "no_behavior_u": "절제: U − 행동 피처"}
    for nm in ["uniform_only", "full_m311af", "full", "full_w5af", "no_behavior", "no_emb", "no_behavior_m311af", "no_emb_m311af", "struct_only_m311af", "no_emb_u", "no_behavior_u"]:
        if nm in js:
            row(label.get(nm, nm), js[nm])
    L.append("\nCAT 기존 논문 보고치(Epinions): AP 0.9407, AUC 0.9688, MRR 0.6758(observed).\n")

    L.append("## 2. 쌍대 bootstrap: EviTrust − CAT (같은 테스트 쌍, B=1000, observed)\n")
    L.append("| 방법 | seed | 음성 집합 | EviTrust AP | CAT AP | 차이 | 95% CI | p(차이≤0) |")
    L.append("|---|---|---|---|---|---|---|---|")
    verdict = {}
    for nm in ["uniform_only", "full_m311af", "full"]:
        if nm not in js:
            continue
        for s in sorted(js[nm]):
            f = D / f"scores_{nm}_s{s}.npz"
            if not f.exists() or s not in cat:
                continue
            E = dict(np.load(f)); Cc = cat[s]
            po = E["observed"]; assert (po == Cc["observed"]).all()
            for r in REG[:3]:
                # observed negatives: both endpoints are training nodes; recover from CAT/EviTrust json counts
                nm_mask = Cc[f"{r}_obs"]          # identical negatives for both methods (same seed)
                sp_e, sn_e, sp_c, sn_c = E["pos"][po], E[r][nm_mask], Cc["pos"][po], Cc[r][nm_mask]
                lo, hi, p = paired(sp_e, sn_e, sp_c, sn_c, seed=s)
                ae, ac = N3.fast_ap(sp_e, sn_e), N3.fast_ap(sp_c, sn_c)
                verdict.setdefault((nm, r), []).append(lo > 0)
                L.append(f"| {label[nm]} | {s} | {REG_KO[r]} | {ae:.4f} | {ac:.4f} | {ae - ac:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {p:.3f} |")
    L.append("\n## 3. 사전 고정 기준 판정 (C-E6a: a ≥ CAT−0.005, C-E6b/c: b,c ≥ CAT+0.05, 세 seed 모두 + 쌍대 CI)\n")
    cat_m = {r: st.mean(e5[s]["regimes"][r]["observed"]["AP"] for s in e5) for r in REG[:3]}
    L.append("| 방법 | C-E6a | C-E6b | C-E6c | 쌍대 CI>0 (a,b,c seed 수) |")
    L.append("|---|---|---|---|---|")
    for nm in ["uniform_only", "full_m311af", "full"]:
        if nm not in js:
            continue
        ok = []
        for r, thr in (("a_cat_own", cat_m["a_cat_own"] - 0.005), ("b_degree_matched", cat_m["b_degree_matched"] + 0.05), ("c_twohop_time", cat_m["c_twohop_time"] + 0.05)):
            v = [js[nm][s]["regimes"][r]["observed"]["AP"] for s in js[nm]]
            ok.append(("✅" if all(x >= thr for x in v) else "❌") + f" (min {min(v):.4f} vs {thr:.4f})")
        ci = ", ".join(f"{sum(verdict.get((nm, r), []))}/{len(verdict.get((nm, r), []))}" for r in REG[:3])
        L.append(f"| {label[nm]} | " + " | ".join(ok) + f" | {ci} |")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
