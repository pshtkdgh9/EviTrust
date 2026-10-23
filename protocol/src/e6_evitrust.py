"""E6 - EviTrust: evidence-based trust prediction that does not rely on the popularity shortcut.

Run from  Reference code/CAT/code/  (like e5_cat_hardneg.py) so that CAT_main's own split and
CAT's own test negatives are reproduced with the identical random sequence:

    python ../../../protocol/src/e6_evitrust.py --seeds 42 43 44 --out ../../../protocol/results_v2/e6

Every pair feature is computed from a graph *state* that only contains information strictly
before the labels it is used for:
    train labels : trust edges in time windows inside the training period, state = window start
    val labels   : CAT val edges,  state = CAT val graph  (trust <= val_time, reviews < val_time)
    test labels  : CAT test edges, state = CAT test graph (trust <= test_time, reviews < test_time)
Model selection (number of trees, leaves) uses the validation split only.

Pre-registered claims: protocol/HYPOTHESES_prereg_v2.md, section E6.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())             # CAT/code
import e3_negsampling as N3                # noqa: E402  same negative generators as E3/E5

import CAT_main as C                       # noqa: E402  (must run from CAT/code)
from utils import setup_seed               # noqa: E402

NEG_SEED = 2025
DAY = 86400.0


# ------------------------------------------------------------- CAT split ----------

DATASETS = {
    "epinions": dict(data_path="/data/Epinions", emb="/code/uitiu_user_emb_meta_64.txt", item_emb="/code/uitiu_item_emb_meta_64.txt"),
    "ciaodvd": dict(data_path="/../../protocol/data_cat/ciaodvd"),
    "wikirfa": dict(data_path="/../../protocol/data_cat/wikirfa"),
    # M4: same data, stronger metapath2vec embeddings (20 epochs, 10 walks/node, walk length 40)
    "ciaodvd_mp2": dict(data_path="/../../protocol/data_cat/ciaodvd",
                        emb="/../../protocol/data_cat/ciaodvd/user_emb_meta_64_mp2.txt",
                        item_emb="/../../protocol/data_cat/ciaodvd/item_emb_meta_64_mp2.txt"),
    "wikirfa_mp2": dict(data_path="/../../protocol/data_cat/wikirfa",
                        emb="/../../protocol/data_cat/wikirfa/user_emb_meta_64_mp2.txt",
                        item_emb="/../../protocol/data_cat/wikirfa/item_emb_meta_64_mp2.txt"),
}
for _k, _v in DATASETS.items():
    _v.setdefault("emb", _v["data_path"] + "/user_emb_meta_64.txt")
    _v.setdefault("item_emb", _v["data_path"] + "/item_emb_meta_64.txt")


def cat_split_and_own_negatives(seed, dataset="epinions"):
    """Replicates e5_cat_hardneg.py up to CAT's test negative sampling (same RNG order)."""
    sys.argv = [sys.argv[0]]
    args = C.parameter_parser()
    args.seed = seed
    args.data_path = DATASETS[dataset]["data_path"]
    setup_seed(seed)
    graph, item_type = C.get_ori_graph(args)
    trustor = np.array(graph["user", "to", "user"].edge_index[0])
    trustee = np.array(graph["user", "to", "user"].edge_index[1])
    ts = np.array(graph["user", "to", "user"].edge_label_time)
    val_time, test_time = list(np.quantile(ts, [args.val_ratio, args.test_ratio]))
    total_node_set = set(np.unique(np.hstack([trustor, trustee])))
    mask_node_set = random.sample(list(set(trustor[ts > val_time]).union(set(trustee[ts > val_time]))),
                                  int(args.mask_ratio * len(total_node_set)))
    mask_tr_flag = np.isin(trustor, mask_node_set); mask_te_flag = np.isin(trustee, mask_node_set)
    none_node_flag = (1 - mask_tr_flag) * (1 - mask_te_flag)
    valid_train_flag = (ts <= val_time) * (none_node_flag > 0)
    train_node_set = set(trustor[valid_train_flag]).union(trustee[valid_train_flag])
    valid_val_flag = (ts > val_time) * (ts <= test_time)
    valid_test_flag = ts > test_time
    val_data = C.get_dataset("val", graph, valid_train_flag, valid_val_flag, valid_test_flag, val_time, test_time, mask_node_set)
    test_data = C.get_dataset("test", graph, valid_train_flag, valid_val_flag, valid_test_flag, val_time, test_time, mask_node_set)
    val_data = C.creat_edge_time(val_data); test_data = C.creat_edge_time(test_data)
    lab_v, idx_v = C.negative_sampler(val_data["user", "to", "user"], val_data["user"].num_nodes, args.neg_seed)
    lab_t, idx_t = C.negative_sampler(test_data["user", "to", "user"], test_data["user"].num_nodes, args.neg_seed + 1)
    own_val_neg = [tuple(x) for x in idx_v.numpy()[:, lab_v.numpy() == 0].T.tolist()]
    own_test_neg = [tuple(x) for x in idx_t.numpy()[:, lab_t.numpy() == 0].T.tolist()]
    return dict(src=trustor, dst=trustee, ts=ts.astype(np.float64), val_time=float(val_time), test_time=float(test_time),
                tr=valid_train_flag.astype(bool), va=valid_val_flag.astype(bool), te=valid_test_flag.astype(bool),
                train_nodes=train_node_set, n_users=int(graph["user"].num_nodes),
                own_val_neg=own_val_neg, own_test_neg=own_test_neg)


def load_reviews(dataset="epinions"):
    r = np.genfromtxt(C.root_path + DATASETS[dataset]["data_path"] + "/user_item_time.txt", delimiter=",", usecols=(0, 1, 2, 3, 4, 5))
    return dict(u=r[:, 0].astype(np.int64), i=r[:, 1].astype(np.int64), c=r[:, 2].astype(np.int64),
                r=r[:, 3].astype(np.float32), h=r[:, 4].astype(np.float32), t=r[:, 5].astype(np.float64))


# --------------------------------------------------------------- state ------------

class State:
    """Everything a scorer may know at time `cutoff`: trust edges in `tmask`, reviews before cutoff."""

    def __init__(self, src, dst, ts, tmask, rv, r_cutoff, cutoff, n, emb, device):
        self.n, self.cutoff, self.dev = n, cutoff, device
        s, d, t = src[tmask], dst[tmask], ts[tmask]
        A = torch.zeros(n, n, device=device)
        A[torch.as_tensor(s, device=device), torch.as_tensor(d, device=device)] = 1.0
        A.fill_diagonal_(0)
        self.A = A
        self.outdeg = A.sum(1); self.indeg = A.sum(0)
        U = ((A + A.T) > 0).float(); degU = U.sum(1)
        # pair statistics are computed on demand as row dot products (memory: A, A^T, U, A^2 only)
        self.AT = A.T.contiguous(); self.U = U
        self.A2 = A @ A                       # u -> w -> v
        self.w_aa = torch.where(degU > 1, 1.0 / torch.log(degU.clamp(min=2)), torch.zeros_like(degU))
        self.w_ra = torch.where(degU > 0, 1.0 / degU.clamp(min=1), torch.zeros_like(degU))
        self.degU = degU
        # recent activity (last 180 days before cutoff)
        rec = t > cutoff - 180 * DAY
        self.rec_in = torch.as_tensor(np.bincount(d[rec], minlength=n), device=device).float()
        self.rec_out = torch.as_tensor(np.bincount(s[rec], minlength=n), device=device).float()
        last = np.full(n, np.nan); first = np.full(n, np.nan)
        for arr_u, arr_t in ((s, t), (d, t)):
            o = np.argsort(arr_t)
            last_u = np.full(n, -np.inf); np.maximum.at(last_u, arr_u, arr_t); last = np.fmax(last, np.where(np.isfinite(last_u), last_u, np.nan))
            first_u = np.full(n, np.inf); np.minimum.at(first_u, arr_u, arr_t); first = np.fmin(first, np.where(np.isfinite(first_u), first_u, np.nan))
        # reviews strictly before r_cutoff
        rm = rv["t"] < r_cutoff
        ru, ri, rc, rr, rh, rt = (rv[k][rm] for k in ("u", "i", "c", "r", "h", "t"))
        lr = np.full(n, -np.inf); np.maximum.at(lr, ru, rt); lr = np.where(np.isfinite(lr), lr, np.nan)
        self.last_act = np.fmax(last, lr)
        fr = np.full(n, np.inf); np.minimum.at(fr, ru, rt); fr = np.where(np.isfinite(fr), fr, np.nan)
        self.first_act = np.fmin(first, fr)
        self.n_rev = np.bincount(ru, minlength=n).astype(np.float64)
        self.mean_rating = np.bincount(ru, weights=rr, minlength=n) / np.maximum(self.n_rev, 1)
        self.mean_help = np.bincount(ru, weights=rh, minlength=n) / np.maximum(self.n_rev, 1)
        self.mean_rating[self.n_rev == 0] = np.nan; self.mean_help[self.n_rev == 0] = np.nan
        cat = np.zeros((n, int(rv["c"].max()) + 1)); np.add.at(cat, (ru, rc), 1.0)
        cat = cat / np.maximum(np.linalg.norm(cat, axis=1, keepdims=True), 1e-12)
        self.cat = torch.as_tensor(cat, device=device, dtype=torch.float32)
        # per-user {item: (time, rating)} for exact co-review statistics
        self.items = [dict() for _ in range(n)]
        for a, b, c_, d_ in zip(ru.tolist(), ri.tolist(), rt.tolist(), rr.tolist()):
            prev = self.items[a].get(b)
            if prev is None or c_ < prev[0]:
                self.items[a][b] = (c_, d_)
        self.keys = [set(x) for x in self.items]
        self.emb = emb

    def features(self, pairs):
        P = np.asarray(pairs, dtype=np.int64)
        u = torch.as_tensor(P[:, 0], device=self.dev); v = torch.as_tensor(P[:, 1], device=self.dev)
        g = lambda M: M[u, v].float().cpu().numpy()
        gu = lambda x: x[u].float().cpu().numpy()
        gv = lambda x: x[v].float().cpu().numpy()

        def dot(L, R, w=None, bs=4096):       # sum_k L[u,k] * w[k] * R[v,k]  in batches
            out = []
            for i in range(0, len(u), bs):
                a_, b_ = L[u[i:i + bs]], R[v[i:i + bs]]
                out.append(((a_ * w) if w is not None else a_).mul_(b_).sum(1).cpu())
            return torch.cat(out).numpy()
        od_u, id_u, od_v, id_v = gu(self.outdeg), gu(self.indeg), gv(self.outdeg), gv(self.indeg)
        a2 = g(self.A2)
        a3 = dot(self.A2, self.AT)            # (A^2 A)[u,v] = A2[u,:] . A[:,v]
        co_o = dot(self.A, self.A)            # common trustees
        co_i = dot(self.AT, self.AT)          # common trustors
        cn = dot(self.U, self.U); aa = dot(self.U, self.U, self.w_aa); ra = dot(self.U, self.U, self.w_ra)
        F = {
            "log_out_u": np.log1p(od_u), "log_in_u": np.log1p(id_u), "log_out_v": np.log1p(od_v), "log_in_v": np.log1p(id_v),
            "log_pa": np.log1p(od_u * id_v),
            "recip": g(self.A.T),
            "log_a2": np.log1p(a2), "log_a3": np.log1p(a3), "a2_norm_u": a2 / np.maximum(od_u, 1), "a2_norm_v": a2 / np.maximum(id_v, 1),
            "a3_norm": a3 / np.maximum(od_u * id_v, 1),
            "log_coout": np.log1p(co_o), "coout_norm": co_o / np.maximum(np.sqrt(od_u * od_v), 1),
            "log_coin": np.log1p(co_i), "coin_norm": co_i / np.maximum(np.sqrt(id_u * id_v), 1),
            "log_cn": np.log1p(cn), "aa": aa, "ra": ra,
            "jacc": cn / np.maximum(gu(self.degU) + gv(self.degU) - cn, 1),
            "log_rec_in_v": np.log1p(gv(self.rec_in)), "log_rec_out_u": np.log1p(gu(self.rec_out)),
            "rec_in_frac_v": gv(self.rec_in) / np.maximum(id_v, 1),
        }
        cu, cv = P[:, 0], P[:, 1]
        F["days_since_u"] = (self.cutoff - self.last_act[cu]) / DAY
        F["days_since_v"] = (self.cutoff - self.last_act[cv]) / DAY
        F["age_u"] = (self.cutoff - self.first_act[cu]) / DAY
        F["age_v"] = (self.cutoff - self.first_act[cv]) / DAY
        F["log_nrev_u"] = np.log1p(self.n_rev[cu]); F["log_nrev_v"] = np.log1p(self.n_rev[cv])
        F["rating_v"] = self.mean_rating[cv]; F["rating_u"] = self.mean_rating[cu]
        F["help_v"] = self.mean_help[cv]
        F["cat_cos"] = (self.cat[u] * self.cat[v]).sum(1).cpu().numpy()
        # exact co-review statistics
        nco = np.zeros(len(P)); vfirst = np.zeros(len(P)); ufirst = np.zeros(len(P)); rdiff = np.full(len(P), np.nan)
        for k, (a, b) in enumerate(P.tolist()):
            ka, kb = self.keys[a], self.keys[b]
            if not ka or not kb:
                continue
            com = ka & kb if len(ka) < len(kb) else kb & ka
            if not com:
                continue
            ia, ib = self.items[a], self.items[b]
            nco[k] = len(com); s_d = 0.0
            for it in com:
                ta, ra_ = ia[it]; tb, rb_ = ib[it]
                if tb < ta: vfirst[k] += 1
                elif ta < tb: ufirst[k] += 1
                s_d += abs(ra_ - rb_)
            rdiff[k] = s_d / len(com)
        F["log_coreview"] = np.log1p(nco); F["log_vfirst"] = np.log1p(vfirst); F["log_ufirst"] = np.log1p(ufirst)
        F["coreview_norm"] = nco / np.maximum(np.sqrt(np.maximum(self.n_rev[cu], 1) * np.maximum(self.n_rev[cv], 1)), 1)
        F["rating_diff"] = rdiff
        if self.emb is not None:
            eu, ev = self.emb[u], self.emb[v]
            F["emb_dot"] = (eu * ev).sum(1).cpu().numpy()
            F["emb_cos"] = torch.nn.functional.cosine_similarity(eu, ev, dim=1).cpu().numpy()
        names = list(F)
        return np.column_stack([F[k] for k in names]).astype(np.float32), names


# ------------------------------------------------------------- negatives ----------

def hard_negatives(pos, pos_t, G, forbid, first_seen, rng, n, mix=(1, 1, 1)):
    """mix[0] uniform + mix[1] degree-matched + mix[2] two-hop negatives per positive (E3 generators)."""
    out = []
    for _ in range(mix[0]):
        out.append(N3.neg_uniform(len(pos), n, forbid, rng))
    for _ in range(mix[1]):
        out.append(N3.neg_degree_matched(pos, G, forbid, rng, n)[0])
    for _ in range(mix[2]):
        out.append(N3.neg_twohop_time(pos, pos_t, G, forbid, first_seen, rng, n)[0])
    return tuple(out)


def test_regimes(S, G, pos, pos_t, first_seen, n, dataset):
    """Test negative sets shared by every method (same generator, same seed, same order as E5).
    wiki-RfA additionally gets (d) real oppose votes cast in the test period."""
    rng = np.random.default_rng(NEG_SEED); forbid = G.edges; forbid_all = forbid | set(pos)
    reg = {"a_uniform_e3": N3.neg_uniform(len(pos), n, forbid, rng),
           "b_degree_matched": N3.neg_degree_matched(pos, G, forbid_all, rng, n)[0],
           "c_twohop_time": N3.neg_twohop_time(pos, pos_t, G, forbid_all, first_seen, rng, n)[0],
           "a_cat_own": S["own_test_neg"]}
    f = C.root_path + DATASETS[dataset]["data_path"] + "/oppose_time.txt"
    if os.path.exists(f):
        o = np.loadtxt(f).astype(np.int64)
        o = o[o[:, 2] > S["test_time"]]
        seen, d = set(), []
        for a, b in o[:, :2].tolist():
            if a != b and (a, b) not in forbid_all and (a, b) not in seen:
                seen.add((a, b)); d.append((a, b))
        reg["d_oppose"] = d
    if os.environ.get("EVI_EXTRA_NEG") == "1":
        reg.update(extra_regimes(S, G, pos, pos_t, n, dataset, forbid_all))
    return reg


def extra_regimes(S, G, pos, pos_t, n, dataset, forbid_all, window_days=30):
    """N1 - hard negatives that no model is trained on (own RNG stream, so (a)-(d) are unchanged).
    (e) co-review: the negative trustee reviewed (voted on) an item that u reviewed, both before t.
    (f) active degree-matched: same in-degree decile as v AND active in (t - 30 days, t].
    Unfilled positions fall back to degree-matched."""
    import bisect
    rng = np.random.default_rng(NEG_SEED + 100)
    rv = load_reviews(dataset)
    ru, ri, rt = rv["u"].tolist(), rv["i"].tolist(), rv["t"].tolist()
    by_user, by_item = {}, {}
    for u_, i_, t_ in zip(ru, ri, rt):
        by_user.setdefault(u_, []).append((t_, i_)); by_item.setdefault(i_, []).append((t_, u_))
    for d_ in (by_user, by_item):
        for k in d_:
            d_[k].sort()
    act = {}
    for a_, b_, t_ in zip(S["src"].tolist(), S["dst"].tolist(), S["ts"].tolist()):
        act.setdefault(a_, []).append(t_); act.setdefault(b_, []).append(t_)
    for u_, t_ in zip(ru, rt):
        act.setdefault(u_, []).append(t_)
    for k in act:
        act[k].sort()
    indeg = np.array([G.indeg.get(i, 0) for i in range(n)]); appear = np.array(sorted(G.nodes))
    qs = np.quantile(indeg[appear], np.linspace(0, 1, 11)[1:-1]); bin_of = np.digitize(indeg, qs); bin_of[indeg == 0] = -1
    pools = {b: np.where(bin_of == b)[0] for b in np.unique(bin_of)}
    lo_w = window_days * 86400.0
    e_out, f_out, miss_e, miss_f = [], [], [], []
    for k, ((u, v), t) in enumerate(zip(pos, pos_t)):
        got = None
        ul = by_user.get(u, [])
        nu = bisect.bisect_left(ul, (t, -1))
        if nu:
            for _ in range(40):
                it = ul[rng.integers(0, nu)][1]; il = by_item[it]; ni = bisect.bisect_left(il, (t, -1))
                if ni == 0:
                    continue
                w = il[rng.integers(0, ni)][1]
                if w != u and (u, w) not in forbid_all:
                    got = (u, w); break
        if got is None:
            miss_e.append(k)
        e_out.append(got)
        got = None; pool = pools[bin_of[v]]
        for _ in range(50):
            w = int(pool[rng.integers(0, len(pool))])
            if w == u or (u, w) in forbid_all:
                continue
            a = act.get(w)
            if a and bisect.bisect_right(a, t) - bisect.bisect_right(a, t - lo_w) > 0:
                got = (u, w); break
        if got is None:
            miss_f.append(k)
        f_out.append(got)
    for out, miss in ((e_out, miss_e), (f_out, miss_f)):
        if miss:
            fb, _ = N3.neg_degree_matched([pos[i] for i in miss], G, forbid_all, rng, n)
            for i, x in zip(miss, fb):
                out[i] = x
    print(f"[extra negatives] co-review fallback {len(miss_e) / len(pos):.1%}, active-dm fallback {len(miss_f) / len(pos):.1%}", flush=True)
    return {"e_coreview": e_out, "f_active_dm": f_out}


def ap_obs(sp, sn, pm, nm):
    return N3.fast_ap(sp[pm], sn[nm])


# ------------------------------------------------------------------ main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--out", default="../../../protocol/results_v2/e6")
    ap.add_argument("--variant", default="full", choices=["full", "uniform_only", "no_behavior", "no_emb", "struct_only"])
    ap.add_argument("--windows", nargs="+", type=float, default=[0.5, 0.6, 0.7])
    ap.add_argument("--mix", nargs=3, type=int, default=[1, 1, 1], help="uniform / degree-matched / 2-hop negatives per positive")
    ap.add_argument("--select", default="mean", choices=["mean", "a_floor", "mean_ab", "mean_ac"],
                    help="mean: mean val AP over a,b,c; a_floor: mean of b,c subject to val AP(a) >= best val AP(a) - 0.005")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dataset", default="epinions", choices=list(DATASETS))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import lightgbm as lgb
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rv = load_reviews(args.dataset)
    emb_np = np.loadtxt(C.root_path + DATASETS[args.dataset]["emb"]).astype(np.float32)

    for seed in args.seeds:
        name = args.variant + (("_" + args.tag) if args.tag else "")
        f_out = os.path.join(args.out, f"evitrust_{name}_s{seed}.json")
        if os.path.exists(f_out):
            print("exists, skip:", f_out); continue
        t0 = time.time()
        S = cat_split_and_own_negatives(seed, args.dataset)
        src, dst, ts, n = S["src"], S["dst"], S["ts"], S["n_users"]
        emb = None if args.variant in ("no_emb", "struct_only") else torch.as_tensor(emb_np[:n], device=dev)
        first_seen = {}
        for a, b, t in zip(src.tolist(), dst.tolist(), ts.tolist()):
            first_seen[a] = min(first_seen.get(a, np.inf), t); first_seen[b] = min(first_seen.get(b, np.inf), t)
        rng = np.random.default_rng(10_000 + seed)

        # ---- training windows inside the training period ----
        qs = np.quantile(ts, args.windows)
        Xs, ys = [], []
        for w0, w1 in zip(qs[:-1], qs[1:]):
            smask = ts <= w0
            wmask = (ts > w0) & (ts <= w1) & S["tr"]
            st = State(src, dst, ts, smask, rv, w0, w0, n, emb, dev)
            Gw = N3.Graph(src[smask], dst[smask], ts[smask])
            seen = set(); pos = []; pos_t = []
            for a, b, t in zip(src[wmask].tolist(), dst[wmask].tolist(), ts[wmask].tolist()):
                if (a, b) not in seen and (a, b) not in Gw.edges and a != b:
                    seen.add((a, b)); pos.append((a, b)); pos_t.append(t)
            forbid = Gw.edges | seen
            mix = (1, 0, 0) if args.variant == "uniform_only" else tuple(args.mix)
            negs = hard_negatives(pos, np.array(pos_t), Gw, forbid, first_seen, rng, n, mix)
            Xp, names = st.features(pos)
            Xs.append(Xp); ys.append(np.ones(len(pos)))
            for ng in negs:
                Xn, _ = st.features(ng); Xs.append(Xn); ys.append(np.zeros(len(ng)))
            print(f"[seed {seed}] window ({w0:.0f},{w1:.0f}]: pos {len(pos)}, negs {[len(x) for x in negs]}", flush=True)
            del st; torch.cuda.empty_cache()
        Xtr = np.vstack(Xs); ytr = np.concatenate(ys)
        wtr = np.where(ytr == 1, (ytr == 0).sum() / max((ytr == 1).sum(), 1), 1.0)

        # ---- validation: CAT val graph, 3 negative regimes + CAT's own val negatives ----
        trm = S["tr"]
        sv = State(src, dst, ts, trm, rv, S["val_time"], S["val_time"], n, emb, dev)
        Gv = N3.Graph(src[trm], dst[trm], ts[trm])
        vpos = list(zip(src[S["va"]].tolist(), dst[S["va"]].tolist())); vpos_t = ts[S["va"]]
        vforbid = Gv.edges | set(vpos)
        vrng = np.random.default_rng(20_000 + seed)
        vneg = dict(zip(["a", "b", "c"], hard_negatives(vpos, vpos_t, Gv, vforbid, first_seen, vrng, n)))
        vneg["own"] = S["own_val_neg"]
        tn = S["train_nodes"]
        obs = lambda pairs: np.array([a in tn and b in tn for a, b in pairs])
        Xvp, _ = sv.features(vpos); vpm = obs(vpos)
        Xvn = {k: sv.features(x)[0] for k, x in vneg.items()}; vnm = {k: obs(x) for k, x in vneg.items()}
        del sv; torch.cuda.empty_cache()

        keep = list(range(len(names)))
        if args.variant in ("no_behavior", "struct_only"):
            beh = {"days_since_u", "days_since_v", "age_u", "age_v", "log_nrev_u", "log_nrev_v", "rating_v", "rating_u",
                   "help_v", "cat_cos", "log_coreview", "log_vfirst", "log_ufirst", "coreview_norm", "rating_diff"}
            keep = [i for i, nm in enumerate(names) if nm not in beh]
        fn = [names[i] for i in keep]

        # model selection on validation: mean observed AP over the three regimes (a, b, c)
        cands = []
        for leaves in (31, 127):
            m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=leaves, min_child_samples=50,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                   random_state=seed, verbose=-1)
            m.fit(Xtr[:, keep], ytr, sample_weight=wtr)
            # pick the number of trees by the validation criterion
            for nt in range(100, 2001, 100):
                p = m.predict_proba(Xvp[:, keep], raw_score=True, num_iteration=nt)
                aps = {k: ap_obs(p, m.predict_proba(Xvn[k][:, keep], raw_score=True, num_iteration=nt), vpm, vnm[k]) for k in ("a", "b", "c")}
                cands.append((aps, leaves, nt, m))
        if args.select == "mean":
            key = lambda c: np.mean([c[0][k] for k in "abc"])
            pool = cands
        elif args.select in ("mean_ab", "mean_ac"):          # N1: the held-out generator is never used for selection
            ks = "ab" if args.select == "mean_ab" else "ac"
            key = lambda c, ks=ks: np.mean([c[0][k] for k in ks])
            pool = cands
        else:
            amax = max(c[0]["a"] for c in cands)
            pool = [c for c in cands if c[0]["a"] >= amax - 0.005]
            key = lambda c: (c[0]["b"] + c[0]["c"]) / 2
        aps, leaves, nt, model = max(pool, key=key)
        crit = float(key((aps, leaves, nt, model)))
        vres = {}
        p = model.predict_proba(Xvp[:, keep], raw_score=True, num_iteration=nt)
        for k in Xvn:
            vres[k] = ap_obs(p, model.predict_proba(Xvn[k][:, keep], raw_score=True, num_iteration=nt), vpm, vnm[k])
        print(f"[seed {seed}] selected leaves={leaves} trees={nt} val mean AP(a,b,c)={crit:.4f} val AP {vres}", flush=True)

        # ---- test: CAT test graph, same negative sets as E5 ----
        tvm = S["tr"] | S["va"]
        stt = State(src, dst, ts, tvm, rv, S["test_time"], S["test_time"], n, emb, dev)
        G = N3.Graph(src[tvm], dst[tvm], ts[tvm])
        pos = list(zip(src[S["te"]].tolist(), dst[S["te"]].tolist())); pos_t = ts[S["te"]]
        regimes = test_regimes(S, G, pos, pos_t, first_seen, n, args.dataset)
        Xp, _ = stt.features(pos)
        sp = model.predict_proba(Xp[:, keep], raw_score=True, num_iteration=nt)
        observed = obs(pos)
        pa = np.log1p(np.array([G.outdeg.get(a, 0) * G.indeg.get(b, 0) for a, b in pos]))
        rep = dict(seed=seed, dataset=args.dataset, variant=args.variant, tag=args.tag, mix=list(args.mix), windows=list(args.windows), select=args.select, selected=dict(leaves=leaves, trees=nt, val_crit=crit, val_ap=vres),
                   n_train=int(len(ytr)), features=fn,
                   spearman_score_logPA_observed=float(spearmanr(sp[observed], pa[observed]).statistic), regimes={})
        scores = {"pos": sp}
        for reg, neg in regimes.items():
            Xn, _ = stt.features(neg)
            sn = model.predict_proba(Xn[:, keep], raw_score=True, num_iteration=nt); scores[reg] = sn
            nob = obs(neg); res = {}
            for subset, pm, nm in (("all", np.ones(len(pos), bool), np.ones(len(neg), bool)), ("observed", observed, nob), ("unobserved", ~observed, ~nob)):
                if pm.sum() == 0 or nm.sum() == 0:
                    continue
                m_, rr = N3.metrics(sp[pm], sn[nm], np.random.default_rng(NEG_SEED + 7))
                m_["ci"] = N3.bootstrap(sp[pm], sn[nm], rr, np.random.default_rng(NEG_SEED + 11))
                m_.update(n_pos=int(pm.sum()), n_neg=int(nm.sum()), prevalence=float(pm.sum() / (pm.sum() + nm.sum())))
                res[subset] = m_
            rep["regimes"][reg] = res
            o = res.get("observed", res["all"])
            print(f"[seed {seed}] {reg:<18} observed: AP {o['AP']:.4f} [{o['ci']['AP'][0]:.3f},{o['ci']['AP'][1]:.3f}] "
                  f"AUC {o['AUC']:.4f} MRR {o['MRR']:.4f} (n_neg {o['n_neg']}, chance {o['prevalence']:.3f})", flush=True)
        imp = model.booster_.feature_importance("gain")
        rep["importance_gain"] = dict(sorted(zip(fn, map(float, imp)), key=lambda x: -x[1]))
        rep["runtime_s"] = time.time() - t0
        np.savez_compressed(os.path.join(args.out, f"scores_{name}_s{seed}.npz"), observed=observed, **scores)
        print(f"[seed {seed}] spearman(score, log PA) = {rep['spearman_score_logPA_observed']:.3f}  runtime {rep['runtime_s']:.0f}s")
        with open(f_out, "w") as fh:
            json.dump(rep, fh, indent=1)
        print("wrote", f_out, flush=True)
        del stt; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
