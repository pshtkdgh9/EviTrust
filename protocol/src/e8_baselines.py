"""E8 (M2) - additional baselines under the CAT/EviTrust protocol.

Run from Reference code/CAT/code:
    python ../../../protocol/src/e8_baselines.py --dataset epinions --method guardian --seeds 42 43 44 \
        --out ../../../protocol/results_v2/e8/epinions

Methods (every one scores the identical test pairs of e6_evitrust.test_regimes):
    heur_lr     logistic regression on 5 classic heuristics (recip, PA, CN, AA, Jaccard), trained like
                EviTrust-U (temporal label windows, uniform negatives), features from the state S_tau
    guardian    Guardian (INFOCOM'20) layer algebra from models.py, link-prediction head, full batch
    trustguard  TrustGuard (TDSC'24) from models.py, 4 equal-time snapshots + temporal attention
    tgn         TGN (Rossi et al.) with PyG's TGNMemory / LastNeighborLoader / graph attention
Guardian/TrustGuard/TGN: message-passing graph = train edges during training/validation and
train+val edges at test time (CAT's graphs); negatives uniform 1:1 resampled per epoch (CAT's
regime); epoch chosen by validation AP on observed pairs against CAT's own validation negatives.
Guardian/TrustGuard get the same metapath2vec user embeddings as CAT.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import e3_negsampling as N3                       # noqa: E402
import e6_evitrust as E6                          # noqa: E402
import CAT_main as C                              # noqa: E402
from models import Guardian, TrustGuard           # noqa: E402


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------- shared ----------

def setup(seed, dataset):
    S = E6.cat_split_and_own_negatives(seed, dataset)
    src, dst, ts, n = S["src"], S["dst"], S["ts"], S["n_users"]
    first_seen = {}
    for a, b, t in zip(src.tolist(), dst.tolist(), ts.tolist()):
        first_seen[a] = min(first_seen.get(a, np.inf), t); first_seen[b] = min(first_seen.get(b, np.inf), t)
    tvm = S["tr"] | S["va"]
    G = N3.Graph(src[tvm], dst[tvm], ts[tvm])
    pos = list(zip(src[S["te"]].tolist(), dst[S["te"]].tolist())); pos_t = ts[S["te"]]
    regimes = E6.test_regimes(S, G, pos, pos_t, first_seen, n, dataset)
    tn = S["train_nodes"]
    obs = lambda pairs: np.array([a in tn and b in tn for a, b in pairs])
    vpos = list(zip(src[S["va"]].tolist(), dst[S["va"]].tolist()))
    return S, first_seen, G, pos, pos_t, regimes, obs, vpos


def report(method, dataset, seed, sp, sn_by_reg, pos, regimes, obs, extra, out_dir):
    observed = obs(pos)
    rep = dict(method=method, dataset=dataset, seed=seed, regimes={}, **extra)
    scores = {"pos": sp, "observed": observed}
    for reg, neg in regimes.items():
        sn = sn_by_reg[reg]; nob = obs(neg); scores[reg] = sn; scores[f"{reg}_obs"] = nob
        res = {}
        for subset, pm, nm in (("all", np.ones(len(pos), bool), np.ones(len(neg), bool)), ("observed", observed, nob)):
            if pm.sum() == 0 or nm.sum() == 0:
                continue
            m_, rr = N3.metrics(sp[pm], sn[nm], np.random.default_rng(E6.NEG_SEED + 7))
            m_["ci"] = N3.bootstrap(sp[pm], sn[nm], rr, np.random.default_rng(E6.NEG_SEED + 11))
            m_.update(n_pos=int(pm.sum()), n_neg=int(nm.sum()), prevalence=float(pm.sum() / (pm.sum() + nm.sum())))
            res[subset] = m_
        rep["regimes"][reg] = res
        o = res.get("observed", res["all"])
        print(f"[{dataset} {method} s{seed}] {reg:<18} observed AP {o['AP']:.4f} AUC {o['AUC']:.4f} MRR {o['MRR']:.4f}", flush=True)
    np.savez_compressed(os.path.join(out_dir, f"scores_{method}_s{seed}.npz"), **scores)
    with open(os.path.join(out_dir, f"{method}_s{seed}.json"), "w") as fh:
        json.dump(rep, fh, indent=1)


# ------------------------------------------------------------ heuristic LR ----------

HEUR = ["log_pa", "recip", "log_cn", "aa", "jacc"]


def run_heur_lr(seed, dataset, out_dir, dev):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    t0 = time.time()
    S, first_seen, G, pos, pos_t, regimes, obs, vpos = setup(seed, dataset)
    src, dst, ts, n = S["src"], S["dst"], S["ts"], S["n_users"]
    rv = E6.load_reviews(dataset)
    rng = np.random.default_rng(10_000 + seed)
    qs = np.quantile(ts, [0.5, 0.6, 0.7]); Xs, ys = [], []
    for w0, w1 in zip(qs[:-1], qs[1:]):
        sm = ts <= w0; wm = (ts > w0) & (ts <= w1) & S["tr"]
        st = E6.State(src, dst, ts, sm, rv, w0, w0, n, None, dev)
        Gw = N3.Graph(src[sm], dst[sm], ts[sm]); seen, P = set(), []
        for a, b in zip(src[wm].tolist(), dst[wm].tolist()):
            if (a, b) not in seen and (a, b) not in Gw.edges and a != b:
                seen.add((a, b)); P.append((a, b))
        Ng = N3.neg_uniform(len(P), n, Gw.edges | seen, rng)
        Xp, names = st.features(P); Xn, _ = st.features(Ng)
        Xs += [Xp, Xn]; ys += [np.ones(len(P)), np.zeros(len(Ng))]
        del st; torch.cuda.empty_cache()
    keep = [names.index(h) for h in HEUR]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    model.fit(np.vstack(Xs)[:, keep], np.concatenate(ys))
    tvm = S["tr"] | S["va"]
    stt = E6.State(src, dst, ts, tvm, rv, S["test_time"], S["test_time"], n, None, dev)
    f = lambda pairs: model.decision_function(stt.features(pairs)[0][:, keep])
    sp = f(pos); sn = {k: f(v) for k, v in regimes.items()}
    report("heur_lr", dataset, seed, sp, sn, pos, regimes, obs,
           dict(runtime_s=time.time() - t0, coef=model[-1].coef_.tolist(), features=HEUR), out_dir)


# --------------------------------------------------------- Guardian / TrustGuard ----------

def load_x(dataset, n):
    X = np.loadtxt(C.root_path + E6.DATASETS[dataset]["emb"]).astype(np.float32)[:n]
    return torch.from_numpy(X)


def run_gnn(kind, seed, dataset, out_dir, dev, epochs=200, eval_every=5, lr=0.01, wd=1e-5, n_slots=4):
    t0 = time.time()
    S, first_seen, G, pos, pos_t, regimes, obs, vpos = setup(seed, dataset)
    set_seed(seed)
    src, dst, ts, n = S["src"], S["dst"], S["ts"], S["n_users"]
    X = load_x(dataset, n).to(dev)

    def graph(mask):
        pairs = {}
        for a, b, t in zip(src[mask].tolist(), dst[mask].tolist(), ts[mask].tolist()):
            if a != b and (a, b) not in pairs:
                pairs[(a, b)] = t
        P = np.array(list(pairs), dtype=np.int64); T = np.array(list(pairs.values()))
        return P, T

    def inputs(P, T):
        idx = torch.as_tensor(P.T, device=dev)
        oh = torch.zeros(idx.shape[1], 2, device=dev); oh[:, 1] = 1.0      # every edge is a trust edge
        if kind == "guardian":
            return (idx, oh)
        lo, hi = T.min(), T.max(); span = (hi - lo) / n_slots; snaps = []
        for k in range(n_slots):
            a = lo + k * span; b = hi + 1 if k == n_slots - 1 else lo + (k + 1) * span
            m = (T >= a) & (T < b)
            if m.sum() == 0:
                m = np.ones(len(T), bool)
            snaps.append((idx[:, torch.as_tensor(m, device=dev)], oh[torch.as_tensor(m, device=dev)]))
        return snaps

    Ptr, Ttr = graph(S["tr"]); Ptv, Ttv = graph(S["tr"] | S["va"])
    in_tr, in_tv = inputs(Ptr, Ttr), inputs(Ptv, Ttv)
    model = (Guardian(X.shape[1], 2, layers=(64,)) if kind == "guardian"
             else TrustGuard(X.shape[1], 2, layers=(32, 64, 32), n_slots=n_slots, n_heads=8)).to(dev)
    enc = (lambda inp: model.encode(X, *inp)) if kind == "guardian" else (lambda inp: model.encode(X, inp))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    pos_tr = torch.as_tensor(Ptr.T, device=dev)
    forbid = set(map(tuple, Ptr.tolist())); rng = np.random.default_rng(seed)
    vneg = S["own_val_neg"]; vpm, vnm = obs(vpos), obs(vneg)
    vp_t = torch.as_tensor(np.array(vpos).T, device=dev); vn_t = torch.as_tensor(np.array(vneg).T, device=dev)
    sc = lambda z, P: torch.softmax(model.score(z, P), 1)[:, 1]
    best = (-1, None, -1)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        neg = torch.as_tensor(np.array(N3.neg_uniform(len(Ptr), n, forbid, rng)).T, device=dev)
        z = enc(in_tr)
        logits = torch.cat([model.score(z, pos_tr), model.score(z, neg)])
        y = torch.cat([torch.ones(pos_tr.shape[1]), torch.zeros(neg.shape[1])]).long().to(dev)
        loss = F.cross_entropy(logits, y); loss.backward(); opt.step()
        if (ep + 1) % eval_every:
            continue
        model.eval()
        with torch.no_grad():
            z = enc(in_tr)
            vap = N3.fast_ap(sc(z, vp_t).cpu().numpy()[vpm], sc(z, vn_t).cpu().numpy()[vnm])
        if vap > best[0]:
            best = (vap, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
        print(f"  [{kind} s{seed}] ep {ep + 1} loss {loss.item():.4f} val AP {vap:.4f}", flush=True)
    model.load_state_dict(best[1]); model.eval()
    with torch.no_grad():
        z = enc(in_tv)
        f = lambda pairs: torch.cat([sc(z, torch.as_tensor(np.array(pairs[i:i + 65536]).T, device=dev))
                                     for i in range(0, len(pairs), 65536)]).cpu().numpy()
        sp = f(pos); sn = {k: f(v) for k, v in regimes.items()}
    report(kind, dataset, seed, sp, sn, pos, regimes, obs,
           dict(runtime_s=time.time() - t0, best_epoch=best[2] + 1, val_ap=best[0]), out_dir)


# --------------------------------------------------------------------- TGN ----------

class GraphAttentionEmbedding(nn.Module):
    """PyG TGN example (examples/tgn.py)."""

    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        from torch_geometric.nn import TransformerConv
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2, dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = (last_update[edge_index[0]] - t).float()
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        return self.conv(x, edge_index, torch.cat([rel_t_enc, msg], dim=-1))


class LinkPredictor(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.lin_src = nn.Linear(c, c); self.lin_dst = nn.Linear(c, c); self.lin_final = nn.Linear(c, 1)

    def forward(self, zs, zd):
        return self.lin_final((self.lin_src(zs) + self.lin_dst(zd)).relu())


def run_tgn(seed, dataset, out_dir, dev, epochs=30, bs=200, mem=100, patience=5):
    from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, LastNeighborLoader, TGNMemory
    t0 = time.time()
    S, first_seen, G, pos, pos_t, regimes, obs, vpos = setup(seed, dataset)
    set_seed(seed)
    src, dst, ts, n = S["src"], S["dst"], S["ts"], S["n_users"]
    t_scale = 3600.0                                     # integer hours (TGNMemory keeps long timestamps)

    def events(mask):
        o = np.argsort(ts[mask], kind="stable")
        return (torch.as_tensor(src[mask][o], device=dev), torch.as_tensor(dst[mask][o], device=dev),
                torch.as_tensor(np.floor(ts[mask][o] / t_scale).astype(np.int64), device=dev))
    tr_ev, va_ev = events(S["tr"]), events(S["va"])
    msg_dim = 1
    memory = TGNMemory(n, msg_dim, mem, mem, message_module=IdentityMessage(msg_dim, mem, mem),
                       aggregator_module=LastAggregator()).to(dev)
    gnn = GraphAttentionEmbedding(mem, mem, msg_dim, memory.time_enc).to(dev)
    lp = LinkPredictor(mem).to(dev)
    opt = torch.optim.Adam(set(memory.parameters()) | set(gnn.parameters()) | set(lp.parameters()), lr=1e-4)
    loader = LastNeighborLoader(n, size=10, device=dev)
    assoc = torch.empty(n, dtype=torch.long, device=dev)
    # global event table indexed by LastNeighborLoader's e_id (= insertion order since reset)
    T_buf = torch.zeros(len(tr_ev[0]) + len(va_ev[0]), dtype=torch.long, device=dev)
    M_buf = torch.ones(len(T_buf), msg_dim, device=dev)
    cur = [0]

    def push(s, d, t):
        msg = M_buf[:len(s)]
        memory.update_state(s, d, t, msg); loader.insert(s, d)
        T_buf[cur[0]:cur[0] + len(s)] = t; cur[0] += len(s)

    def embed(n_id):
        n_id, edge_index, e_id = loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=dev)
        z, last_update = memory(n_id)
        return gnn(z, last_update, edge_index, T_buf[e_id], M_buf[e_id])

    @torch.no_grad()
    def score(pairs):
        out = []
        for i in range(0, len(pairs), 2000):
            P = torch.as_tensor(np.array(pairs[i:i + 2000]), device=dev)
            n_id = torch.cat([P[:, 0], P[:, 1]]).unique()
            z = embed(n_id)
            out.append(lp(z[assoc[P[:, 0]]], z[assoc[P[:, 1]]]).squeeze(-1).cpu())
        return torch.cat(out).numpy()

    vneg = S["own_val_neg"]; vpm, vnm = obs(vpos), obs(vneg)
    best = (-1, None, None, -1)
    for ep in range(epochs):
        memory.train(); gnn.train(); lp.train()
        memory.reset_state(); loader.reset_state(); cur[0] = 0
        s_, d_, t_ = tr_ev; tot = 0.0
        for i in range(0, len(s_), bs):
            s, d, t = s_[i:i + bs], d_[i:i + bs], t_[i:i + bs]
            opt.zero_grad()
            neg = torch.randint(0, n, (len(s),), device=dev)
            n_id = torch.cat([s, d, neg]).unique()
            z = embed(n_id)
            ps = lp(z[assoc[s]], z[assoc[d]]); ns = lp(z[assoc[s]], z[assoc[neg]])
            loss = F.binary_cross_entropy_with_logits(ps, torch.ones_like(ps)) + F.binary_cross_entropy_with_logits(ns, torch.zeros_like(ns))
            push(s, d, t)
            loss.backward(); opt.step(); memory.detach(); tot += loss.item() * len(s)
        memory.eval(); gnn.eval(); lp.eval()
        vp_s, vn_s = score(vpos), score(vneg)                  # state at val_time
        vap = N3.fast_ap(vp_s[vpm], vn_s[vnm])
        print(f"  [tgn s{seed}] ep {ep + 1} loss {tot / len(s_):.4f} val AP {vap:.4f}", flush=True)
        if vap > best[0]:
            with torch.no_grad():                               # advance the state through val events
                vs, vd, vt = va_ev
                for i in range(0, len(vs), bs):
                    push(vs[i:i + bs], vd[i:i + bs], vt[i:i + bs])
                sp = score(pos); sn = {k: score(v) for k, v in regimes.items()}
            best = (vap, sp, sn, ep)
        elif ep - best[3] >= patience:
            break
    report("tgn", dataset, seed, best[1], best[2], pos, regimes, obs,
           dict(runtime_s=time.time() - t0, best_epoch=best[3] + 1, val_ap=best[0]), out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(E6.DATASETS))
    ap.add_argument("--method", required=True, choices=["heur_lr", "guardian", "trustguard", "tgn"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for s in a.seeds:
        if os.path.exists(os.path.join(a.out, f"{a.method}_s{s}.json")):
            print("exists, skip", a.method, s); continue
        if a.method == "heur_lr":
            run_heur_lr(s, a.dataset, a.out, dev)
        elif a.method in ("guardian", "trustguard"):
            run_gnn(a.method, s, a.dataset, a.out, dev)
        else:
            run_tgn(s, a.dataset, a.out, dev)


if __name__ == "__main__":
    main()
