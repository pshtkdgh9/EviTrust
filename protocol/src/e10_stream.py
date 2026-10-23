"""E10 (review M3) - incremental EviTrust state: equivalence, update/query latency, scalability.

The dense State of e6_evitrust materializes A, A^T, U, A^2 as n x n GPU matrices (O(n^2)
memory). StreamState keeps only adjacency sets, per-user counters/timestamp lists and per-user
review dictionaries, updated in O(1) per trust record / interaction; pair features are computed
on demand by set intersections. Feature formulas are copied from State.features (embedding
features F5 are excluded - they are unnecessary, Table VII).

Sub-commands (run from Reference code/CAT/code for the real datasets):
    verify   --dataset D        streaming vs dense features on the CAT test pairs (max abs/rel diff)
    latency  --dataset D        per-update and per-query latency, memory; CAT refresh and TGN update cost
    scale    --sizes 10000 ...  synthetic temporal graphs: ingest time, memory, query latency vs #users
"""
from __future__ import annotations

import argparse
import bisect
import gc
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

DAY = 86400.0
FEATS = ["log_out_u", "log_in_u", "log_out_v", "log_in_v", "log_pa", "recip", "log_a2", "log_a3", "a2_norm_u",
         "a2_norm_v", "a3_norm", "log_coout", "coout_norm", "log_coin", "coin_norm", "log_cn", "aa", "ra", "jacc",
         "log_rec_in_v", "log_rec_out_u", "rec_in_frac_v", "days_since_u", "days_since_v", "age_u", "age_v",
         "log_nrev_u", "log_nrev_v", "rating_v", "rating_u", "help_v", "cat_cos", "log_coreview", "log_vfirst",
         "log_ufirst", "coreview_norm", "rating_diff"]


def rss_mb():
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 2**20


class StreamState:
    """Everything EviTrust needs, maintained incrementally."""

    __slots__ = ("out", "inn", "und", "in_t", "out_t", "last", "first", "items", "n_rev", "s_rat", "s_help", "cat")

    def __init__(self):
        self.out, self.inn = {}, {}                 # adjacency sets (deduplicated, no self loops)
        self.und = {}                               # undirected neighbour sets, kept so that |N(w)| is O(1)
        self.in_t, self.out_t = {}, {}              # all trust-record times per user (append-only, sorted)
        self.last, self.first = {}, {}
        self.items = {}                             # u -> {item: (first time, rating)}
        self.n_rev, self.s_rat, self.s_help = {}, {}, {}
        self.cat = {}                               # u -> {category: count}

    # ---- O(1) updates ----
    def add_trust(self, u, v, t):
        self.in_t.setdefault(v, []).append(t); self.out_t.setdefault(u, []).append(t)
        if u != v:
            self.out.setdefault(u, set()).add(v); self.inn.setdefault(v, set()).add(u)
            self.und.setdefault(u, set()).add(v); self.und.setdefault(v, set()).add(u)
        for x in (u, v):
            if t > self.last.get(x, -math.inf): self.last[x] = t
            if t < self.first.get(x, math.inf): self.first[x] = t

    def add_review(self, u, i, c, r, h, t):
        d = self.items.setdefault(u, {}); prev = d.get(i)
        if prev is None or t < prev[0]:
            d[i] = (t, r)
        self.n_rev[u] = self.n_rev.get(u, 0) + 1; self.s_rat[u] = self.s_rat.get(u, 0.0) + r
        self.s_help[u] = self.s_help.get(u, 0.0) + h
        cu = self.cat.setdefault(u, {}); cu[c] = cu.get(c, 0) + 1
        if t > self.last.get(u, -math.inf): self.last[u] = t
        if t < self.first.get(u, math.inf): self.first[u] = t

    # ---- on-demand pair features ----

    def features(self, u, v, cutoff):
        E = set()
        ou, iu, ov, iv = self.out.get(u, E), self.inn.get(u, E), self.out.get(v, E), self.inn.get(v, E)
        od_u, id_u, od_v, id_v = len(ou), len(iu), len(ov), len(iv)
        a2 = len(ou & iv) if od_u < id_v else len(iv & ou)
        a3 = 0
        if ou and iv:
            # (A^3)_uv = sum_{x in in(v)} |out(u) & in(x)| = sum_{w in out(u)} |out(w) & in(v)|; loop over the smaller side
            if id_v <= od_u:
                for x in iv:
                    ix = self.inn.get(x, E)
                    a3 += len(ou & ix) if od_u < len(ix) else len(ix & ou)
            else:
                for w in ou:
                    ow = self.out.get(w, E)
                    a3 += len(iv & ow) if id_v < len(ow) else len(ow & iv)
        co_o = len(ou & ov); co_i = len(iu & iv)
        uu, uv = self.und.get(u, E), self.und.get(v, E)
        cnset = uu & uv if len(uu) < len(uv) else uv & uu; cn = len(cnset); aa = ra = 0.0
        und = self.und
        for w in cnset:
            dw = len(und[w])
            if dw > 1: aa += 1.0 / math.log(dw)
            if dw > 0: ra += 1.0 / dw
        lo = cutoff - 180 * DAY
        tin = self.in_t.get(v, ()); tout = self.out_t.get(u, ())
        rec_in = len(tin) - bisect.bisect_right(tin, lo); rec_out = len(tout) - bisect.bisect_right(tout, lo)
        nan = float("nan")
        lu, lv, fu, fv = self.last.get(u), self.last.get(v), self.first.get(u), self.first.get(v)
        nu, nv = self.n_rev.get(u, 0), self.n_rev.get(v, 0)
        cu, cv = self.cat.get(u), self.cat.get(v)
        if cu and cv:
            dot = sum(x * cv.get(k, 0) for k, x in cu.items())
            cc = dot / (math.sqrt(sum(x * x for x in cu.values())) * math.sqrt(sum(x * x for x in cv.values())))
        else:
            cc = 0.0
        ia, ib = self.items.get(u, {}), self.items.get(v, {})
        nco = vfirst = ufirst = 0; rdiff = nan
        if ia and ib:
            com = ia.keys() & ib.keys() if len(ia) < len(ib) else ib.keys() & ia.keys()
            if com:
                s_d = 0.0
                for it in com:
                    ta, ra_ = ia[it]; tb, rb_ = ib[it]
                    if tb < ta: vfirst += 1
                    elif ta < tb: ufirst += 1
                    s_d += abs(ra_ - rb_)
                nco = len(com); rdiff = s_d / nco
        return [math.log1p(od_u), math.log1p(id_u), math.log1p(od_v), math.log1p(id_v), math.log1p(od_u * id_v),
                1.0 if u in ov else 0.0,
                math.log1p(a2), math.log1p(a3), a2 / max(od_u, 1), a2 / max(id_v, 1), a3 / max(od_u * id_v, 1),
                math.log1p(co_o), co_o / max(math.sqrt(od_u * od_v), 1), math.log1p(co_i), co_i / max(math.sqrt(id_u * id_v), 1),
                math.log1p(cn), aa, ra, cn / max(len(uu) + len(uv) - cn, 1),
                math.log1p(rec_in), math.log1p(rec_out), rec_in / max(id_v, 1),
                (cutoff - lu) / DAY if lu is not None else nan, (cutoff - lv) / DAY if lv is not None else nan,
                (cutoff - fu) / DAY if fu is not None else nan, (cutoff - fv) / DAY if fv is not None else nan,
                math.log1p(nu), math.log1p(nv),
                self.s_rat[v] / nv if nv else nan, self.s_rat[u] / nu if nu else nan, self.s_help[v] / nv if nv else nan,
                cc, math.log1p(nco), math.log1p(vfirst), math.log1p(ufirst),
                nco / max(math.sqrt(max(nu, 1) * max(nv, 1)), 1), rdiff]


def ingest(st, trust, reviews, t_cut, r_cut):
    """Replay both streams in time order up to the cutoffs; return per-type update latencies (s)."""
    tr = [(t, 0, u, v) for u, v, t in trust if t <= t_cut]
    rv = [(t, 1, u, i, c, r, h) for u, i, c, r, h, t in reviews if t < r_cut]
    ev = sorted(tr + rv, key=lambda e: e[0])
    lat_t, lat_r = [], []
    pc = time.perf_counter
    for e in ev:
        t0 = pc()
        if e[1] == 0:
            st.add_trust(e[2], e[3], e[0]); lat_t.append(pc() - t0)
        else:
            st.add_review(e[2], e[3], e[4], e[5], e[6], e[0]); lat_r.append(pc() - t0)
    return np.array(lat_t), np.array(lat_r)


def pct(a):
    a = np.asarray(a) * 1e6
    return dict(mean_us=float(a.mean()), p50_us=float(np.percentile(a, 50)), p99_us=float(np.percentile(a, 99)), n=int(len(a)))


# --------------------------------------------------------------- real data ----------

def load_real(dataset, seed=42):
    import e6_evitrust as E6
    import e3_negsampling as N3
    S = E6.cat_split_and_own_negatives(seed, dataset)
    src, dst, ts, n = S["src"], S["dst"], S["ts"], S["n_users"]
    first_seen = {}
    for a, b, t in zip(src.tolist(), dst.tolist(), ts.tolist()):
        first_seen[a] = min(first_seen.get(a, np.inf), t); first_seen[b] = min(first_seen.get(b, np.inf), t)
    tvm = S["tr"] | S["va"]
    G = N3.Graph(src[tvm], dst[tvm], ts[tvm])
    pos = list(zip(src[S["te"]].tolist(), dst[S["te"]].tolist())); pos_t = ts[S["te"]]
    regimes = E6.test_regimes(S, G, pos, pos_t, first_seen, n, dataset)
    rv = E6.load_reviews(dataset)
    trust = list(zip(src.tolist(), dst.tolist(), ts.tolist()))
    reviews = list(zip(rv["u"].tolist(), rv["i"].tolist(), rv["c"].tolist(), rv["r"].tolist(), rv["h"].tolist(), rv["t"].tolist()))
    return S, rv, trust, reviews, pos, regimes, n, tvm


def cmd_verify(a):
    import torch
    import e6_evitrust as E6
    S, rv, trust, reviews, pos, regimes, n, tvm = load_real(a.dataset)
    cut = S["test_time"]
    pairs = pos + [p for neg in regimes.values() for p in neg]
    rng = np.random.default_rng(0); idx = rng.choice(len(pairs), size=min(a.n_pairs, len(pairs)), replace=False)
    pairs = [pairs[i] for i in idx]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dense = E6.State(S["src"], S["dst"], S["ts"], tvm, rv, cut, cut, n, None, dev)
    Xd, names = dense.features(pairs)
    assert names == FEATS, (names, FEATS)
    st = StreamState()
    # identical inputs: trust records in tvm (t <= test_time for this split), reviews before test_time
    tr_sel = [(u, v, t) for (u, v, t), m in zip(trust, tvm) if m]
    ingest(st, tr_sel, reviews, math.inf, cut)
    Xs = np.array([st.features(u, v, cut) for u, v in pairs], dtype=np.float64)
    Xd = Xd.astype(np.float64)
    both_nan = np.isnan(Xd) & np.isnan(Xs)
    diff = np.where(both_nan, 0.0, np.abs(Xd - Xs))
    rel = diff / np.maximum(np.abs(Xd), 1e-6)
    per = {f: dict(max_abs=float(np.nanmax(diff[:, j])), max_rel=float(np.nanmax(np.where(np.abs(Xd[:, j]) > 1e-6, rel[:, j], 0))),
                   nan_mismatch=int((np.isnan(Xd[:, j]) != np.isnan(Xs[:, j])).sum())) for j, f in enumerate(FEATS)}
    worst = max(per.items(), key=lambda kv: kv[1]["max_rel"])
    rep = dict(dataset=a.dataset, n_pairs=len(pairs), per_feature=per, worst=worst,
               all_within_1e4_rel=bool(all(v["max_rel"] < 1e-4 and v["nan_mismatch"] == 0 for v in per.values())))
    os.makedirs(a.out, exist_ok=True)
    json.dump(rep, open(os.path.join(a.out, f"verify_{a.dataset}.json"), "w"), indent=1)
    print(a.dataset, "pairs", len(pairs), "all within 1e-4 rel:", rep["all_within_1e4_rel"], "worst:", worst)


def cmd_latency(a):
    import lightgbm as lgb
    import torch
    S, rv, trust, reviews, pos, regimes, n, tvm = load_real(a.dataset)
    cut = S["test_time"]
    gc.collect(); m0 = rss_mb()
    st = StreamState()
    tr_sel = [(u, v, t) for (u, v, t), m in zip(trust, tvm) if m]
    t0 = time.perf_counter(); lt, lr = ingest(st, tr_sel, reviews, math.inf, cut); t_ing = time.perf_counter() - t0
    gc.collect(); m1 = rss_mb()
    pairs = pos + [p for neg in regimes.values() for p in neg]
    rng = np.random.default_rng(1); pairs = [pairs[i] for i in rng.choice(len(pairs), size=min(a.n_pairs, len(pairs)), replace=False)]
    q = []
    for u, v in pairs:
        t0 = time.perf_counter(); st.features(u, v, cut); q.append(time.perf_counter() - t0)
    # LightGBM scoring cost of a model of the selected size (31 leaves, 600 trees)
    X = np.array([st.features(u, v, cut) for u, v in pairs[:20000]], dtype=np.float32)
    y = (np.random.default_rng(2).random(len(X)) < 0.5).astype(int)
    mdl = lgb.LGBMClassifier(n_estimators=600, num_leaves=31, learning_rate=0.03, min_child_samples=5, verbose=-1).fit(X, y)
    t0 = time.perf_counter(); mdl.predict_proba(X, raw_score=True); gbm_batch = (time.perf_counter() - t0) / len(X)
    one = []
    for i in range(200):
        t0 = time.perf_counter(); mdl.predict_proba(X[i:i + 1], raw_score=True); one.append(time.perf_counter() - t0)
    rep = dict(dataset=a.dataset, users=n, trust_records=len(tr_sel), reviews_ingested=int(len(lr)), ingest_s=t_ing,
               update_trust=pct(lt), update_review=pct(lr), query_features=pct(q),
               gbm_us_per_pair_batch=gbm_batch * 1e6, gbm_single=pct(one), state_rss_mb=m1 - m0)
    # ---- comparators ----
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rep["dense_state"] = dense_cost(a.dataset, S, rv, tvm, n, cut, dev)
    rep["tgn_update"] = tgn_update_cost(n, dev)
    rep["cat_refresh"] = cat_refresh_cost(a.dataset, S, pos, dev)
    os.makedirs(a.out, exist_ok=True)
    json.dump(rep, open(os.path.join(a.out, f"latency_{a.dataset}.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


def dense_cost(dataset, S, rv, tvm, n, cut, dev):
    import torch
    import e6_evitrust as E6
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter(); st = E6.State(S["src"], S["dst"], S["ts"], tvm, rv, cut, cut, n, None, dev); torch.cuda.synchronize()
    return dict(build_s=time.perf_counter() - t0, gpu_peak_mb=torch.cuda.max_memory_allocated() / 2**20,
                note="rebuilt from scratch for any change (A^2 recomputed)")


def tgn_update_cost(n, dev, reps=200):
    import torch
    from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, LastNeighborLoader, TGNMemory
    mem = TGNMemory(n, 1, 100, 100, message_module=IdentityMessage(1, 100, 100), aggregator_module=LastAggregator()).to(dev)
    mem.reset_state()
    loader = LastNeighborLoader(n, size=10, device=dev)
    mem.eval(); out = {}
    for bs in (1, 200):
        lat = []
        for k in range(reps):
            s = torch.randint(0, n, (bs,), device=dev); d = torch.randint(0, n, (bs,), device=dev)
            t = torch.full((bs,), k, dtype=torch.long, device=dev); m = torch.ones(bs, 1, device=dev)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad():
                mem.update_state(s, d, t, m); loader.insert(s, d)
            torch.cuda.synchronize(); lat.append((time.perf_counter() - t0) / bs)
        out[f"batch{bs}"] = pct(lat[10:])
    out["note"] = "memory+neighbour-store update per event (GPU); untrained weights (latency independent of weights)"
    return out


def cat_refresh_cost(dataset, S, pos, dev):
    """CAT has no incremental path: a new edge is reflected only by re-running the GNN over the
    test graph. We time one full scoring pass (forward over the whole graph) for 1 pair and for all test positives."""
    import torch
    import e5_cat_hardneg as E5
    import e6_evitrust as E6
    import CAT_main as C
    from CAT_model import cat_model
    sys.argv = [sys.argv[0]]; args = C.parameter_parser(); args.data_path = E6.DATASETS[dataset]["data_path"]
    graph, item_type = C.get_ori_graph(args); item_type = item_type.long()
    test_data = C.creat_edge_time(C.get_dataset("test", graph, S["tr"], S["va"], S["te"], 0, S["test_time"], []))
    test_data["user"].x = torch.from_numpy(np.loadtxt(C.root_path + E6.DATASETS[dataset]["emb"])).float()[:graph["user"].num_nodes]
    test_data["item"].x = torch.from_numpy(np.loadtxt(C.root_path + E6.DATASETS[dataset]["item_emb"])).float()[:graph["item"].num_nodes]
    model = cat_model(num_layers=1, in_dim=64, hidden_dim=64, ntypes=["user", "item"], negative_slope=0.2, seed=42).to(dev)
    K = int(item_type[1].max())
    if K != model.gate.numel():
        model.gate = torch.nn.Parameter(torch.ones(K, device=dev) / K)
    E5.cat_scores(model, test_data, pos[:10], item_type, dev)               # warm-up
    torch.cuda.synchronize(); t0 = time.perf_counter(); E5.cat_scores(model, test_data, pos[:1], item_type, dev); torch.cuda.synchronize()
    one = time.perf_counter() - t0
    torch.cuda.synchronize(); t0 = time.perf_counter(); E5.cat_scores(model, test_data, pos, item_type, dev); torch.cuda.synchronize()
    allp = time.perf_counter() - t0
    return dict(refresh_1pair_s=one, score_all_test_pos_s=allp, n_test_pos=len(pos),
                note="graph conversion + full heterogeneous forward pass; untrained weights")


# --------------------------------------------------------------- synthetic ----------

def synth(N, deg=25, rev=25, n_cat=25, seed=0, span_days=3650):
    """Temporal directed graph with preferential attachment and heavy-tailed activity, plus reviews.
    Trustors ~ Zipf activity; trustees: copy the target of a random earlier edge (prob 0.6, PA) or uniform."""
    rng = np.random.default_rng(seed)
    E = deg * N; R = rev * N; M = max(N, 1000)
    act = rng.zipf(1.8, N).astype(float); act /= act.sum()
    src = rng.choice(N, size=E, p=act)
    dst = np.empty(E, dtype=np.int64)
    copy = rng.random(E) < 0.6; uni = rng.integers(0, N, E)
    for k in range(E):
        dst[k] = dst[rng.integers(0, k)] if (copy[k] and k > 0) else uni[k]
    ts = np.sort(rng.random(E)) * span_days * DAY
    iu = rng.choice(N, size=R, p=act); ipop = rng.zipf(1.6, M).astype(float); ipop /= ipop.sum()
    ii = rng.choice(M, size=R, p=ipop); ic = (ii % n_cat) + 1
    rt = np.sort(rng.random(R)) * span_days * DAY
    trust = list(zip(src.tolist(), dst.tolist(), ts.tolist()))
    reviews = list(zip(iu.tolist(), ii.tolist(), ic.tolist(), rng.integers(1, 6, R).astype(float).tolist(),
                       np.ones(R).tolist(), rt.tolist()))
    return trust, reviews


def cmd_scale(a):
    res = []
    for N in a.sizes:
        trust, reviews = synth(N, seed=N)
        cut = max(t for _, _, t in trust) + 1
        gc.collect(); m0 = rss_mb()
        st = StreamState()
        t0 = time.perf_counter(); lt, lr = ingest(st, trust, reviews, math.inf, cut); t_ing = time.perf_counter() - t0
        gc.collect(); m1 = rss_mb()
        rng = np.random.default_rng(N)
        rand_pairs = list(zip(rng.integers(0, N, a.n_pairs).tolist(), rng.integers(0, N, a.n_pairs).tolist()))
        idx = rng.integers(0, len(trust), a.n_pairs); edge_pairs = [(trust[i][0], trust[i][1]) for i in idx]
        qr, qe = [], []
        for (u, v), (x, y) in zip(rand_pairs, edge_pairs):
            t0 = time.perf_counter(); st.features(u, v, cut); qr.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); st.features(x, y, cut); qe.append(time.perf_counter() - t0)
        dense_gb = 4 * N * N * 4 / 2**30           # A, A^T, U, A^2 in float32
        row = dict(users=N, trust_records=len(trust), reviews=len(reviews), ingest_s=t_ing,
                   update_trust=pct(lt), update_review=pct(lr), query_random_pair=pct(qr), query_edge_pair=pct(qe),
                   state_rss_mb=m1 - m0, dense_state_gb_required=dense_gb)
        res.append(row)
        print(f"N={N:>8}: ingest {t_ing:7.1f}s  upd {row['update_trust']['mean_us']:.2f}us  q_rand {row['query_random_pair']['mean_us']:.1f}us "
              f"q_edge {row['query_edge_pair']['mean_us']:.1f}us (p99 {row['query_edge_pair']['p99_us']:.0f})  RSS {m1 - m0:.0f}MB  dense {dense_gb:.1f}GB", flush=True)
        del st, trust, reviews; gc.collect()
        os.makedirs(a.out, exist_ok=True)
        json.dump(res, open(os.path.join(a.out, "scale.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["verify", "latency", "scale"])
    ap.add_argument("--dataset", default="epinions")
    ap.add_argument("--n-pairs", type=int, default=20000)
    ap.add_argument("--sizes", nargs="+", type=int, default=[10000, 30000, 100000, 300000, 1000000])
    ap.add_argument("--out", default="../../../protocol/results_v2/e10")
    a = ap.parse_args()
    {"verify": cmd_verify, "latency": cmd_latency, "scale": cmd_scale}[a.cmd](a)


if __name__ == "__main__":
    main()
