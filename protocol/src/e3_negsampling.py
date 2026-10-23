"""E3 - what does CAT's Epinions protocol measure?

Re-creates CAT's split and negative sampling (CAT_main.py: chronological 70/85 quantile
split of trust-edge timestamps, mask_ratio=0, uniform random 1:1 negatives drawn against
train+val edges, metrics AUC/AP on pos+neg and MRR with 100 random negatives per
positive), then scores the test set with heuristics that never look at a label or train
anything, under three negative-sampling regimes:

    (a) uniform random 1:1      - CAT's protocol
    (b) degree-matched           - negative trustee drawn from the same in-degree decile
    (c) 2-hop + time-consistent  - negative trustee is a 2-hop neighbour of the trustor
                                   that already existed when the positive edge was formed

Pre-registered claims: HYPOTHESES_prereg_v2.md §E3.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

B = 1000


# ------------------------------------------------------------------ data ----------

def load_epinions(path):
    arr = np.loadtxt(path)
    src = arr[:, 0].astype(np.int64); dst = arr[:, 1].astype(np.int64); ts = arr[:, 2]
    return src, dst, ts


def cat_split(src, dst, ts, val_ratio=0.7, test_ratio=0.85):
    val_time, test_time = np.quantile(ts, [val_ratio, test_ratio])
    tr = ts <= val_time
    va = (ts > val_time) & (ts <= test_time)
    te = ts > test_time
    return tr, va, te, val_time, test_time


class Graph:
    """train+val directed graph with the aggregates the heuristics need."""

    def __init__(self, src, dst, ts):
        self.edges = set(zip(src.tolist(), dst.tolist()))
        self.out = defaultdict(set); self.inn = defaultdict(set); self.und = defaultdict(set)
        for u, v in zip(src.tolist(), dst.tolist()):
            self.out[u].add(v); self.inn[v].add(u)
            self.und[u].add(v); self.und[v].add(u)
        self.outdeg = {u: len(s) for u, s in self.out.items()}
        self.indeg = {v: len(s) for v, s in self.inn.items()}
        self.nodes = set(self.und)

    def score(self, name, u, v):
        if name == "reciprocity":
            return 1.0 if (v, u) in self.edges else 0.0
        if name == "pref_attach":
            return float(self.outdeg.get(u, 0) * self.indeg.get(v, 0))
        nu, nv = self.und.get(u, set()), self.und.get(v, set())
        if name == "common_neigh":
            return float(len(nu & nv))
        if name == "adamic_adar":
            return float(sum(1.0 / np.log(len(self.und[w])) for w in (nu & nv) if len(self.und[w]) > 1))
        if name == "jaccard":
            un = len(nu | nv)
            return float(len(nu & nv) / un) if un else 0.0
        raise ValueError(name)


HEURISTICS = ["reciprocity", "pref_attach", "common_neigh", "adamic_adar", "jaccard"]


# ------------------------------------------------------------- negatives ----------

def neg_uniform(n, num_nodes, forbid, rng):
    """CAT: torch_geometric.utils.negative_sampling(method='sparse') - uniform pairs that
    are not in the given edge_index, self loops excluded."""
    out = []
    while len(out) < n:
        u = rng.integers(0, num_nodes, n); v = rng.integers(0, num_nodes, n)
        for a, b in zip(u.tolist(), v.tolist()):
            if a != b and (a, b) not in forbid:
                out.append((a, b))
                if len(out) == n:
                    break
    return out


def neg_degree_matched(pos, G, forbid, rng, num_nodes, n_bins=10):
    indeg = np.array([G.indeg.get(i, 0) for i in range(num_nodes)])
    # decile bins over nodes that appear in the graph; isolated ids form their own bin
    appear = np.array(sorted(G.nodes))
    qs = np.quantile(indeg[appear], np.linspace(0, 1, n_bins + 1)[1:-1])
    bin_of = np.digitize(indeg, qs)                   # 0..n_bins-1
    bin_of[indeg == 0] = -1                           # unobserved trustees
    pools = {b: np.where(bin_of == b)[0] for b in np.unique(bin_of)}
    out, fallback = [], 0
    for u, v in pos:
        pool = pools[bin_of[v]]
        ok = False
        for _ in range(50):
            w = int(pool[rng.integers(0, len(pool))])
            if w != u and (u, w) not in forbid:
                out.append((u, w)); ok = True; break
        if not ok:
            fallback += 1
            out.append(neg_uniform(1, num_nodes, forbid, rng)[0])
    return out, fallback / len(pos)


def neg_twohop_time(pos, pos_t, G, forbid, first_seen, rng, num_nodes):
    out, fallback = [], 0
    dm_rng = np.random.default_rng(rng.integers(0, 2 ** 31))
    und_list = {u: list(s) for u, s in G.und.items()}
    for (u, v), t in zip(pos, pos_t):
        n1 = und_list.get(u)
        found = None
        if n1 and first_seen.get(u, np.inf) <= t:
            # two random steps (u -> w -> x); accept x if it is a genuine 2-hop node that
            # existed at time t and is not already linked from u. 40 tries.
            for _ in range(40):
                w = n1[rng.integers(0, len(n1))]
                n2 = und_list[w]
                x = n2[rng.integers(0, len(n2))]
                if x != u and x not in G.und[u] and (u, x) not in forbid and first_seen.get(x, np.inf) <= t:
                    found = (u, x); break
        if found is None:
            fallback += 1
        out.append(found)
    # fallback to degree-matched for the rest
    miss = [i for i, x in enumerate(out) if x is None]
    if miss:
        dm, _ = neg_degree_matched([pos[i] for i in miss], G, forbid, dm_rng, num_nodes)
        for i, x in zip(miss, dm):
            out[i] = x
    return out, fallback / len(pos)


# --------------------------------------------------------------- metrics ----------

def mrr_cat(pos_s, neg_s, rng, k=100):
    """CAT's mrr(): each positive against k negatives drawn uniformly from the negative
    score pool; ties get the average of optimistic and pessimistic rank."""
    idx = rng.integers(0, len(neg_s), size=(len(pos_s), k))
    ns = neg_s[idx]
    p = pos_s.reshape(-1, 1)
    opt = (ns > p).sum(1); pes = (ns >= p).sum(1)
    return 1.0 / (0.5 * (opt + pes) + 1)          # per-positive reciprocal ranks


def fast_auc(pos_s, neg_s):
    """Mann-Whitney AUC with ties counted 1/2 (identical to sklearn.roc_auc_score)."""
    from scipy.stats import rankdata
    r = rankdata(np.r_[pos_s, neg_s])
    n, m = len(pos_s), len(neg_s)
    return float((r[:n].sum() - n * (n + 1) / 2) / (n * m))


def fast_ap(pos_s, neg_s):
    """average_precision_score as sklearn defines it (step-wise, ties grouped)."""
    s = np.r_[pos_s, neg_s]; y = np.r_[np.ones(len(pos_s)), np.zeros(len(neg_s))]
    order = np.argsort(-s, kind="stable"); s = s[order]; y = y[order]
    # group ties: evaluate precision/recall at the end of each distinct-score block
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(y)[distinct]; fp = (distinct + 1) - tp
    prec = tp / (tp + fp); rec = tp / y.sum()
    rec_prev = np.r_[0.0, rec[:-1]]
    return float(np.sum((rec - rec_prev) * prec))


def metrics(pos_s, neg_s, rng):
    rr = mrr_cat(pos_s, neg_s, rng)
    return dict(AUC=fast_auc(pos_s, neg_s), AP=fast_ap(pos_s, neg_s), MRR=float(rr.mean())), rr


def bootstrap(pos_s, neg_s, rr, rng):
    aps, aucs, mrrs = [], [], []
    n, m = len(pos_s), len(neg_s)
    for _ in range(B):
        i = rng.integers(0, n, n); j = rng.integers(0, m, m)
        aps.append(fast_ap(pos_s[i], neg_s[j])); aucs.append(fast_auc(pos_s[i], neg_s[j]))
        mrrs.append(rr[i].mean())
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return dict(AP=ci(aps), AUC=ci(aucs), MRR=ci(mrrs))


# ------------------------------------------------------------------ main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CAT/data/Epinions/user_user_time.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--neg-seed", type=int, default=2025)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    src, dst, ts = load_epinions(args.data)
    n_lines = len(src)
    num_nodes = int(max(src.max(), dst.max())) + 1
    rep = {"n_lines": n_lines, "num_nodes_maxid": num_nodes}

    # ---- forensics (C-E3c) ----
    pairs = Counter(zip(src.tolist(), dst.tolist()))
    uniq = set(pairs)
    dup_lines = n_lines - len(uniq)
    tmap = {}
    for u, v, t in zip(src.tolist(), dst.tolist(), ts.tolist()):
        tmap.setdefault((u, v), t)
    has_rev = [(v, u) in uniq for (u, v) in uniq]
    same_t = [tmap[(v, u)] == tmap[(u, v)] for (u, v) in uniq if (v, u) in uniq]
    rep["forensics"] = dict(unique_edges=len(uniq), duplicate_lines=int(dup_lines),
                            frac_with_reverse=float(np.mean(has_rev)),
                            frac_reverse_same_timestamp=float(np.mean(same_t)),
                            one_directional=int(len(uniq) - sum(has_rev)))
    print("forensics:", rep["forensics"])

    # ---- CAT split ----
    tr, va, te, val_time, test_time = cat_split(src, dst, ts)
    G = Graph(src[tr | va], dst[tr | va], ts[tr | va])
    train_nodes = set(src[tr].tolist()) | set(dst[tr].tolist())
    forbid = G.edges
    pos = list(zip(src[te].tolist(), dst[te].tolist())); pos_t = ts[te]
    observed = np.array([u in train_nodes and v in train_nodes for u, v in pos])
    first_seen = {}
    for u, v, t in zip(src.tolist(), dst.tolist(), ts.tolist()):
        first_seen[u] = min(first_seen.get(u, np.inf), t); first_seen[v] = min(first_seen.get(v, np.inf), t)
    rep["split"] = dict(train=int(tr.sum()), val=int(va.sum()), test=int(te.sum()),
                        test_observed=int(observed.sum()), val_time=float(val_time), test_time=float(test_time))
    print("split:", rep["split"])

    # ---- negatives ----
    rng = np.random.default_rng(args.neg_seed)
    regimes = {}
    # (a) reproduces CAT exactly: negatives avoid train+val edges only (a future test
    # positive can be drawn as a negative, as in torch_geometric.negative_sampling).
    # (b)/(c) are *hard* negatives and must be genuine non-edges over the whole
    # observation window, so they also avoid the test positives themselves.
    forbid_all = forbid | set(pos)
    regimes["a_uniform"] = (neg_uniform(len(pos), num_nodes, forbid, rng), 0.0)
    regimes["b_degree_matched"] = neg_degree_matched(pos, G, forbid_all, rng, num_nodes)
    regimes["c_twohop_time"] = neg_twohop_time(pos, pos_t, G, forbid_all, first_seen, rng, num_nodes)
    pos_set = set(pos)
    rep["negatives_that_are_test_positives"] = {k: int(sum((u, v) in pos_set for u, v in neg)) for k, (neg, _) in regimes.items()}
    rep["fallback_rate"] = {k: v[1] for k, v in regimes.items()}
    print("fallback:", rep["fallback_rate"])

    # ---- score ----
    rep["results"] = {}
    for reg, (neg, _) in regimes.items():
        neg_obs = np.array([u in train_nodes and v in train_nodes for u, v in neg])
        rep["results"][reg] = {}
        for h in HEURISTICS + ["recip+pa"]:
            if h == "recip+pa":
                pa_p = np.array([G.score("pref_attach", u, v) for u, v in pos])
                pa_n = np.array([G.score("pref_attach", u, v) for u, v in neg])
                mx = max(pa_p.max(), pa_n.max(), 1.0)
                ps = np.array([G.score("reciprocity", u, v) for u, v in pos]) + pa_p / mx
                ns = np.array([G.score("reciprocity", u, v) for u, v in neg]) + pa_n / mx
            else:
                ps = np.array([G.score(h, u, v) for u, v in pos]); ns = np.array([G.score(h, u, v) for u, v in neg])
            res = {}
            for subset, pm, nm in (("all", np.ones(len(pos), bool), np.ones(len(neg), bool)),
                                   ("observed", observed, neg_obs),
                                   ("unobserved", ~observed, ~neg_obs)):
                if pm.sum() == 0 or nm.sum() == 0:
                    continue
                mrng = np.random.default_rng(args.neg_seed + 7)
                m, rr = metrics(ps[pm], ns[nm], mrng)
                m["ci"] = bootstrap(ps[pm], ns[nm], rr, np.random.default_rng(args.neg_seed + 11))
                m["n_pos"] = int(pm.sum()); m["n_neg"] = int(nm.sum())
                m["prevalence"] = float(pm.sum() / (pm.sum() + nm.sum()))
                res[subset] = m
            rep["results"][reg][h] = res
            o = res.get("observed", res["all"])
            print(f"  {reg:<18} {h:<14} obs: AP {o['AP']:.4f} [{o['ci']['AP'][0]:.3f},{o['ci']['AP'][1]:.3f}]  "
                  f"AUC {o['AUC']:.4f}  MRR {o['MRR']:.4f} [{o['ci']['MRR'][0]:.3f},{o['ci']['MRR'][1]:.3f}]")
    rep["runtime_s"] = time.time() - t0
    (out / "e3_results.json").write_text(json.dumps(rep, indent=1))
    print(f"wrote {out / 'e3_results.json'} ({rep['runtime_s']:.0f}s)")


if __name__ == "__main__":
    main()
