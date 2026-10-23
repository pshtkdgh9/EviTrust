"""E9 (review M1) - CAT and TGN re-trained with EviTrust-H's mixed-difficulty negatives.

Run from Reference code/CAT/code:
    python ../../../protocol/src/e9_mixed_neural.py --dataset epinions --method cat_h --seeds 42 43 44 \
        --out ../../../protocol/results_v2/e9/epinions

Question: is EviTrust-H's hard-negative robustness due to its evidence/model, or only to its
training negatives? We give the neural models the same negative distribution and the same
model-selection rule:
  * training negatives: one negative per positive (CAT's 1:1 balance), of type
    uniform / degree-matched / two-hop time-consistent with probabilities 3/5, 1/5, 1/5
    (= EviTrust-H's 3:1:1 mixture), drawn once from the training graph with the E3 generators
  * selection: among evaluated epochs whose validation AP(a) is within 0.005 of the best,
    take the one with the best mean validation AP over (b) and (c) (EviTrust-H's rule);
    validation negatives (a = CAT's own, b, c) are generated exactly as in e6_evitrust
  * CAT-H: CAT_model, neighbour sampling, loss and optimiser unchanged; only the labelled
    edges fed to LinkNeighborLoader carry our negatives (neg_sampling_ratio disabled);
    patience 10 on the mean validation AP over (a, b, c); max 50 epochs
  * TGN-H: run_tgn of e8_baselines with the mixed destination negatives
Test pairs and output format are identical to E7/E8 (e6_evitrust.test_regimes, e8_baselines.report).
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import e3_negsampling as N3                       # noqa: E402
import e5_cat_hardneg as E5                       # noqa: E402
import e6_evitrust as E6                          # noqa: E402
import e8_baselines as E8                         # noqa: E402
import CAT_main as C                              # noqa: E402
from CAT_model import cat_model                   # noqa: E402
from torch_geometric.loader import LinkNeighborLoader  # noqa: E402
from utils import setup_seed                      # noqa: E402

P_MIX = np.array([3, 1, 1]) / 5.0


def mixed_negatives(pos, pos_t, G, forbid, first_seen, n, rng):
    """One negative per positive; type drawn with P_MIX from (uniform, degree-matched, two-hop)."""
    a = N3.neg_uniform(len(pos), n, forbid, rng)
    b, _ = N3.neg_degree_matched(pos, G, forbid, rng, n)
    c, _ = N3.neg_twohop_time(pos, pos_t, G, forbid, first_seen, rng, n)
    kind = rng.choice(3, size=len(pos), p=P_MIX)
    return [(a, b, c)[k][i] for i, k in enumerate(kind)], np.bincount(kind, minlength=3).tolist()


def val_sets(S, first_seen, n, seed):
    src, dst, ts = S["src"], S["dst"], S["ts"]; trm = S["tr"]
    Gv = N3.Graph(src[trm], dst[trm], ts[trm])
    vpos = list(zip(src[S["va"]].tolist(), dst[S["va"]].tolist())); vpos_t = ts[S["va"]]
    vrng = np.random.default_rng(20_000 + seed)                    # identical to e6_evitrust
    vneg = dict(zip(["a", "b", "c"], E6.hard_negatives(vpos, vpos_t, Gv, Gv.edges | set(vpos), first_seen, vrng, n)))
    vneg["own"] = S["own_val_neg"]
    tn = S["train_nodes"]; obs = lambda pairs: np.array([x in tn and y in tn for x, y in pairs])
    return vpos, vneg, obs(vpos), {k: obs(v) for k, v in vneg.items()}


def select(hist):
    """hist: list of (epoch, {'own','b','c'} val AP, payload). EviTrust-H's a-floor rule."""
    amax = max(h[1]["own"] for h in hist)
    pool = [h for h in hist if h[1]["own"] >= amax - 0.005]
    return max(pool, key=lambda h: (h[1]["b"] + h[1]["c"]) / 2)


# --------------------------------------------------------------------- CAT-H ----------

def run_cat_h(seed, dataset, out_dir, dev, epochs=50, patience=10):
    t0 = time.time()
    sys.argv = [sys.argv[0]]
    args = C.parameter_parser(); args.device = dev; args.seed = seed
    D = E6.DATASETS[dataset]; args.data_path = D["data_path"]
    setup_seed(seed)
    graph, item_type = C.get_ori_graph(args); item_type = item_type.long()
    trustor = np.array(graph["user", "to", "user"].edge_index[0]); trustee = np.array(graph["user", "to", "user"].edge_index[1])
    ts = np.array(graph["user", "to", "user"].edge_label_time)
    val_time, test_time = list(np.quantile(ts, [args.val_ratio, args.test_ratio]))
    total_node_set = set(np.unique(np.hstack([trustor, trustee])))
    mask_node_set = random.sample(list(set(trustor[ts > val_time]).union(set(trustee[ts > val_time]))), int(args.mask_ratio * len(total_node_set)))
    none_node_flag = (1 - np.isin(trustor, mask_node_set)) * (1 - np.isin(trustee, mask_node_set))
    vtr = (ts <= val_time) * (none_node_flag > 0); vva = (ts > val_time) * (ts <= test_time); vte = ts > test_time
    train_node_set = set(trustor[vtr]).union(trustee[vtr])
    train_data = C.get_dataset("train", graph, vtr, vva, vte, val_time, test_time, mask_node_set)
    val_data = C.get_dataset("val", graph, vtr, vva, vte, val_time, test_time, mask_node_set)
    test_data = C.get_dataset("test", graph, vtr, vva, vte, val_time, test_time, mask_node_set)
    train_data = C.creat_edge_time(train_data); val_data = C.creat_edge_time(val_data); test_data = C.creat_edge_time(test_data)
    n = int(graph["user"].num_nodes)
    ue = torch.from_numpy(np.loadtxt(C.root_path + D["emb"])).float()[:n]
    ie = torch.from_numpy(np.loadtxt(C.root_path + D["item_emb"])).float()[:graph["item"].num_nodes]
    for d_ in (train_data, val_data, test_data):
        d_["user"].x = ue; d_["item"].x = ie
    lab_v, idx_v = C.negative_sampler(val_data["user", "to", "user"], n, args.neg_seed)
    lab_t, idx_t = C.negative_sampler(test_data["user", "to", "user"], n, args.neg_seed + 1)
    S = dict(src=trustor, dst=trustee, ts=ts.astype(np.float64), tr=vtr.astype(bool), va=vva.astype(bool), te=vte.astype(bool),
             train_nodes=train_node_set, test_time=float(test_time),
             own_val_neg=[tuple(x) for x in idx_v.numpy()[:, lab_v.numpy() == 0].T.tolist()],
             own_test_neg=[tuple(x) for x in idx_t.numpy()[:, lab_t.numpy() == 0].T.tolist()])
    first_seen = {}
    for x, y, t in zip(trustor.tolist(), trustee.tolist(), ts.tolist()):
        first_seen[x] = min(first_seen.get(x, np.inf), t); first_seen[y] = min(first_seen.get(y, np.inf), t)

    # ---- mixed training negatives, injected as labelled edges ----
    tr = S["tr"]; Gtr = N3.Graph(trustor[tr], trustee[tr], ts[tr])
    pos = list(zip(trustor[tr].tolist(), trustee[tr].tolist())); pos_t = ts[tr].astype(np.float64)
    neg, mix_counts = mixed_negatives(pos, pos_t, Gtr, Gtr.edges, first_seen, n, np.random.default_rng(30_000 + seed))
    et = train_data["user", "to", "user"]
    et.edge_label_index = torch.cat([et.edge_label_index, torch.as_tensor(np.array(neg).T, dtype=torch.long)], 1)
    et.edge_label = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
    et.edge_label_time = torch.cat([et.edge_label_time, et.edge_label_time])
    loader = LinkNeighborLoader(data=train_data, num_neighbors=args.num_neighbors,
                                edge_label_index=(("user", "to", "user"), et.edge_label_index), edge_label=et.edge_label,
                                edge_label_time=et.edge_label_time, time_attr="node_time", temporal_strategy="last",
                                batch_size=args.batch_size, shuffle=True)
    model = cat_model(num_layers=args.num_layers, in_dim=args.emb_size, hidden_dim=args.hidden_dim,
                      ntypes=["user", "item"], negative_slope=args.negative_slope, seed=seed).to(dev)
    K = int(item_type[1].max())
    if K != model.gate.numel():
        model.gate = torch.nn.Parameter(torch.ones(K, device=dev) / K, requires_grad=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = torch.nn.CrossEntropyLoss()
    vpos, vneg, vpm, vnm = val_sets(S, first_seen, n, seed)
    hist, best_crit, bad = [], -1, 0
    for ep in range(epochs):
        model.train(); tot = cnt = 0
        for b in loader:
            eli = b["user", "to", "user"].edge_label_index.to(dev); el = b["user", "to", "user"].edge_label.to(dev)
            hg = C.to_dgl(b).to(dev)
            opt.zero_grad(); out = model(hg, hg.ndata["x"], eli, item_type, b["item"].n_id)
            loss = crit(out, el.long()); loss.backward(); opt.step()
            tot += loss.item() * len(el); cnt += len(el)
        sp = E5.cat_scores(model, val_data, vpos, item_type, dev)
        v = {k: N3.fast_ap(sp[vpm], E5.cat_scores(model, val_data, vneg[k], item_type, dev)[vnm[k]]) for k in ("own", "b", "c")}
        hist.append((ep + 1, v, copy.deepcopy(model.state_dict())))
        m3 = (v["own"] + v["b"] + v["c"]) / 3
        print(f"  [cat_h {dataset} s{seed}] ep {ep + 1} loss {tot / cnt:.4f} val AP own {v['own']:.4f} b {v['b']:.4f} c {v['c']:.4f}", flush=True)
        if m3 > best_crit + 1e-3:
            best_crit, bad = m3, 0
        else:
            bad += 1
            if bad >= patience:
                break
    ep_sel, v_sel, state = select(hist)
    model.load_state_dict(state)
    # ---- test on the shared regimes ----
    tvm = S["tr"] | S["va"]
    G = N3.Graph(trustor[tvm], trustee[tvm], ts[tvm])
    tpos = list(zip(trustor[S["te"]].tolist(), trustee[S["te"]].tolist())); tpos_t = ts[S["te"]].astype(np.float64)
    regimes = E6.test_regimes(S, G, tpos, tpos_t, first_seen, n, dataset)
    obs = lambda pairs: np.array([x in train_node_set and y in train_node_set for x, y in pairs])
    spt = E5.cat_scores(model, test_data, tpos, item_type, dev)
    sn = {k: E5.cat_scores(model, test_data, v_, item_type, dev) for k, v_ in regimes.items()}
    E8.report("cat_h", dataset, seed, spt, sn, tpos, regimes, obs,
              dict(runtime_s=time.time() - t0, selected_epoch=ep_sel, val_ap=v_sel, n_epochs=len(hist), mix_counts=mix_counts), out_dir)


# --------------------------------------------------------------------- TGN-H ----------

def run_tgn_h(seed, dataset, out_dir, dev, epochs=30, bs=200, mem=100, patience=5):
    from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, LastNeighborLoader, TGNMemory
    t0 = time.time()
    S, first_seen, G, pos, pos_t, regimes, obs, vpos = E8.setup(seed, dataset)
    E8.set_seed(seed)
    src, dst, ts, n = S["src"], S["dst"], S["ts"], S["n_users"]
    tr = S["tr"]; o = np.argsort(ts[tr], kind="stable")
    s_np, d_np, t_np = src[tr][o], dst[tr][o], ts[tr][o]
    Gtr = N3.Graph(src[tr], dst[tr], ts[tr])
    neg, mix_counts = mixed_negatives(list(zip(s_np.tolist(), d_np.tolist())), t_np.astype(np.float64), Gtr, Gtr.edges,
                                      first_seen, n, np.random.default_rng(30_000 + seed))
    neg_dst = torch.as_tensor(np.array([b for _, b in neg]), device=dev)
    s_, d_ = torch.as_tensor(s_np, device=dev), torch.as_tensor(d_np, device=dev)
    t_ = torch.as_tensor(np.floor(t_np / 3600.0).astype(np.int64), device=dev)
    vo = np.argsort(ts[S["va"]], kind="stable")
    va_ev = (torch.as_tensor(src[S["va"]][vo], device=dev), torch.as_tensor(dst[S["va"]][vo], device=dev),
             torch.as_tensor(np.floor(ts[S["va"]][vo] / 3600.0).astype(np.int64), device=dev))
    msg_dim = 1
    memory = TGNMemory(n, msg_dim, mem, mem, message_module=IdentityMessage(msg_dim, mem, mem), aggregator_module=LastAggregator()).to(dev)
    gnn = E8.GraphAttentionEmbedding(mem, mem, msg_dim, memory.time_enc).to(dev)
    lp = E8.LinkPredictor(mem).to(dev)
    opt = torch.optim.Adam(set(memory.parameters()) | set(gnn.parameters()) | set(lp.parameters()), lr=1e-4)
    loader = LastNeighborLoader(n, size=10, device=dev)
    assoc = torch.empty(n, dtype=torch.long, device=dev)
    T_buf = torch.zeros(len(s_) + len(va_ev[0]), dtype=torch.long, device=dev); M_buf = torch.ones(len(T_buf), msg_dim, device=dev); cur = [0]

    def push(s, d, t):
        memory.update_state(s, d, t, M_buf[:len(s)]); loader.insert(s, d); T_buf[cur[0]:cur[0] + len(s)] = t; cur[0] += len(s)

    def embed(n_id):
        n_id, edge_index, e_id = loader(n_id); assoc[n_id] = torch.arange(n_id.size(0), device=dev)
        z, last_update = memory(n_id)
        return gnn(z, last_update, edge_index, T_buf[e_id], M_buf[e_id])

    @torch.no_grad()
    def score(pairs):
        out = []
        for i in range(0, len(pairs), 2000):
            P = torch.as_tensor(np.array(pairs[i:i + 2000]), device=dev); z = embed(torch.cat([P[:, 0], P[:, 1]]).unique())
            out.append(lp(z[assoc[P[:, 0]]], z[assoc[P[:, 1]]]).squeeze(-1).cpu())
        return torch.cat(out).numpy()

    vpos_, vneg, vpm, vnm = val_sets(S, first_seen, n, seed)
    hist, best_crit, bad = [], -1, 0
    for ep in range(epochs):
        memory.train(); gnn.train(); lp.train(); memory.reset_state(); loader.reset_state(); cur[0] = 0; tot = 0.0
        for i in range(0, len(s_), bs):
            s, d, t, ng = s_[i:i + bs], d_[i:i + bs], t_[i:i + bs], neg_dst[i:i + bs]
            opt.zero_grad(); z = embed(torch.cat([s, d, ng]).unique())
            ps = lp(z[assoc[s]], z[assoc[d]]); ns = lp(z[assoc[s]], z[assoc[ng]])
            loss = F.binary_cross_entropy_with_logits(ps, torch.ones_like(ps)) + F.binary_cross_entropy_with_logits(ns, torch.zeros_like(ns))
            push(s, d, t); loss.backward(); opt.step(); memory.detach(); tot += loss.item() * len(s)
        memory.eval(); gnn.eval(); lp.eval()
        vp = score(vpos_)
        v = {k: N3.fast_ap(vp[vpm], score(vneg[k])[vnm[k]]) for k in ("own", "b", "c")}
        with torch.no_grad():                         # advance to the test boundary and score the test pairs
            for i in range(0, len(va_ev[0]), bs):
                push(va_ev[0][i:i + bs], va_ev[1][i:i + bs], va_ev[2][i:i + bs])
            payload = (score(pos), {k: score(x) for k, x in regimes.items()})
        hist.append((ep + 1, v, payload))
        print(f"  [tgn_h {dataset} s{seed}] ep {ep + 1} loss {tot / len(s_):.4f} val AP own {v['own']:.4f} b {v['b']:.4f} c {v['c']:.4f}", flush=True)
        m3 = (v["own"] + v["b"] + v["c"]) / 3
        if m3 > best_crit + 1e-3:
            best_crit, bad = m3, 0
        else:
            bad += 1
            if bad >= patience:
                break
    ep_sel, v_sel, (sp, sn) = select(hist)
    E8.report("tgn_h", dataset, seed, sp, sn, pos, regimes, obs,
              dict(runtime_s=time.time() - t0, selected_epoch=ep_sel, val_ap=v_sel, n_epochs=len(hist), mix_counts=mix_counts), out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(E6.DATASETS))
    ap.add_argument("--method", required=True, choices=["cat_h", "tgn_h"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for s in a.seeds:
        if os.path.exists(os.path.join(a.out, f"{a.method}_s{s}.json")):
            print("exists, skip", a.method, s); continue
        kw = {} if a.epochs is None else {"epochs": a.epochs}
        (run_cat_h if a.method == "cat_h" else run_tgn_h)(s, a.dataset, a.out, dev, **kw)


if __name__ == "__main__":
    main()
