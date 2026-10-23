"""E11 (review M4) - diagnosing CAT's transfer to CiaoDVD and wiki-RfA, from saved per-pair scores.

Run from Reference code/CAT/code:
    python ../../../protocol/src/e11_cat_diag.py --out ../../../protocol/results_v2/e11

(1) wiki-RfA new users: test pairs are split by how many endpoints are training nodes
    (2 = both seen, 1 = one new, 0 = both new). For every method we report the share of positives
    and of CAT-protocol negatives in each stratum, the mean score per stratum, and the AUC within
    each stratum and over all pairs. Also: how many test-pair users have pre-test interactions
    (i.e. a metapath2vec embedding trained on real walks).
(2) CiaoDVD random split: test positives whose reverse edge is in the train+val graph vs not,
    AP of each group against the same CAT-protocol negatives.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import e3_negsampling as N3                       # noqa: E402
import e8_baselines as E8                         # noqa: E402

R = Path(__file__).resolve().parents[1] / "results_v2"
METHODS = [("cat", "CAT"), ("guardian", "Guardian"), ("trustguard", "TrustGuard"), ("tgn", "TGN"),
           ("heur_lr", "Heuristic-LR"), ("uniform_only", "EviTrust-U"), ("full_m311af", "EviTrust-H")]


def score_file(ds, m, s):
    if m in ("heur_lr", "guardian", "trustguard", "tgn"):
        return R / "e8" / ds / f"scores_{m}_s{s}.npz"
    return R / "e7" / ds / f"scores_{m}_s{s}.npz"


def auc(p, n):
    return N3.fast_auc(p, n) if len(p) and len(n) else float("nan")


def wikirfa(seeds, L):
    import e6_evitrust as E6
    L.append("## 1. wiki-RfA — 신규 사용자 층별 진단 (CAT 프로토콜 음성 (a), 모든 쌍)\n")
    rows = {m: [] for m, _ in METHODS}
    share = []
    for s in seeds:
        S, first_seen, G, pos, pos_t, regimes, obs, vpos = E8.setup(s, "wikirfa")
        tn = S["train_nodes"]; neg = regimes["a_cat_own"]
        kp = np.array([(a in tn) + (b in tn) for a, b in pos]); kn = np.array([(a in tn) + (b in tn) for a, b in neg])
        rv = E6.load_reviews("wikirfa"); pre = set(rv["u"][rv["t"] < S["test_time"]].tolist())
        endp = lambda pairs: np.mean([x in pre for a, b in pairs for x in (a, b)])
        share.append(dict(pos=[float(np.mean(kp == k)) for k in (2, 1, 0)], neg=[float(np.mean(kn == k)) for k in (2, 1, 0)],
                          pre_pos=endp(pos), pre_neg=endp(neg)))
        for m, _ in METHODS:
            f = score_file("wikirfa", m, s)
            if not f.exists():
                continue
            Z = np.load(f); sp, sn = Z["pos"], Z["a_cat_own"]
            rows[m].append(dict(all=auc(sp, sn), **{f"auc{k}": auc(sp[kp == k], sn[kn == k]) for k in (2, 1, 0)},
                                **{f"mp{k}": float(sp[kp == k].mean()) if (kp == k).any() else np.nan for k in (2, 1, 0)},
                                **{f"mn{k}": float(sn[kn == k].mean()) if (kn == k).any() else np.nan for k in (2, 1, 0)}))
    sh = {k: np.mean([x[k] for x in share], axis=0) for k in ("pos", "neg")}
    L.append("| | 양 끝점 모두 학습 노드 (2) | 한쪽만 (1) | 둘 다 신규 (0) |")
    L.append("|---|---|---|---|")
    L.append("| 테스트 양성 비율 | " + " | ".join(f"{x:.1%}" for x in sh["pos"]) + " |")
    L.append("| CAT 프로토콜 음성 비율 | " + " | ".join(f"{x:.1%}" for x in sh["neg"]) + " |")
    L.append(f"\n테스트 이전 상호작용(투표)이 있는 끝점 비율: 양성 {np.mean([x['pre_pos'] for x in share]):.1%}, 음성 {np.mean([x['pre_neg'] for x in share]):.1%}. "
             "상호작용이 없는 사용자는 metapath2vec에서 더미 노드에만 연결되어 정보 없는 임베딩을 받는다.\n")
    L.append("| 방법 | AUC 전체 | AUC 층 2 | AUC 층 1 | AUC 층 0 | 평균 점수 양성 (2/1/0) | 평균 점수 음성 (2/1/0) |")
    L.append("|---|---|---|---|---|---|---|")
    for m, mn in METHODS:
        if not rows[m]:
            continue
        g = lambda k: np.nanmean([r[k] for r in rows[m]])
        L.append(f"| {mn} | {g('all'):.3f} | {g('auc2'):.3f} | {g('auc1'):.3f} | {g('auc0'):.3f} | "
                 f"{g('mp2'):.3g} / {g('mp1'):.3g} / {g('mp0'):.3g} | {g('mn2'):.3g} / {g('mn1'):.3g} / {g('mn0'):.3g} |")
    L.append("")


def ciaodvd(seeds, L):
    L.append("## 2. CiaoDVD — 역방향 엣지가 학습 그래프에 있는 양성 vs 없는 양성 (observed, 음성 (a))\n")
    rows = {m: [] for m, _ in METHODS}; frac = []
    for s in seeds:
        S, first_seen, G, pos, pos_t, regimes, obs, vpos = E8.setup(s, "ciaodvd")
        rev = np.array([(b, a) in G.edges for a, b in pos]); po = obs(pos)
        neg = regimes["a_cat_own"]; no = obs(neg)
        frac.append(float(rev[po].mean()))
        for m, _ in METHODS:
            f = score_file("ciaodvd", m, s)
            if not f.exists():
                continue
            Z = np.load(f); sp, sn = Z["pos"], Z["a_cat_own"][no]
            rows[m].append(dict(rev=N3.fast_ap(sp[po & rev], sn), norev=N3.fast_ap(sp[po & ~rev], sn),
                                all=N3.fast_ap(sp[po], sn)))
    L.append(f"observed 테스트 양성 중 역방향 엣지가 학습+검증 그래프에 있는 비율: **{np.mean(frac):.1%}**\n")
    L.append("| 방법 | AP 전체 | AP 역방향 있음 | AP 역방향 없음 | 차이 |")
    L.append("|---|---|---|---|---|")
    for m, mn in METHODS:
        if not rows[m]:
            continue
        g = lambda k: np.mean([r[k] for r in rows[m]])
        L.append(f"| {mn} | {g('all'):.3f} | {g('rev'):.3f} | {g('norev'):.3f} | {g('rev') - g('norev'):+.3f} |")
    L.append("\n(층별 AP는 양성 비율이 달라 층 간 절대 비교보다 같은 층 안의 방법 간 비교가 의미 있음)\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(R / "e11"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    L = ["# E11 (M4) — CAT 이식 진단\n"]
    wikirfa(a.seeds, L)
    ciaodvd(a.seeds, L)
    Path(a.out, "E11_diag.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", Path(a.out, "E11_diag.md"))


if __name__ == "__main__":
    main()
