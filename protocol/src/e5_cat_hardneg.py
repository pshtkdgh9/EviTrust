"""E5 - the ORIGINAL CAT model under three negative-sampling regimes.

Place this file in  Reference code/CAT/code/  and run it from that directory, exactly
like CAT_main.py (CAT_main derives its root path from the working directory):

    cd "Reference code/CAT/code"
    python e5_cat_hardneg.py --seeds 42 43 44 --out ../../../protocol/results_v2/e5

Everything about data loading, splitting, model, training, validation-based model
selection and CAT's own (a) test negatives is CAT_main.py's code, imported unchanged.
After training, the best model (the checkpoint CAT itself saves) is additionally scored
on two hard-negative test sets that are generated with the SAME code and SAME random
sequence as protocol/src/e3_negsampling.py (seed 2025), so the negatives are identical
to the ones the heuristics and CAT-lite were scored on:

    (b) degree-matched  : negative trustee from the same in-degree decile as the true one
    (c) 2-hop + time    : negative trustee is a 2-hop neighbour of the trustor that already
                          existed when the positive edge formed (fallback -> (b))

Pre-registered claims: protocol/HYPOTHESES_prereg_v2.md, section E5.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import spearmanr

import CAT_main as C                      # noqa: E402  (must run from CAT/code)
from CAT_model import cat_model
from torch_geometric.loader import LinkNeighborLoader
from utils import setup_seed

NEG_SEED = 2025


# ------------------------------------------------------------- negatives ----------
# verbatim copies of protocol/src/e3_negsampling.py (same rng consumption order)

class Graph:
    def __init__(self, src, dst):
        self.edges = set(zip(src.tolist(), dst.tolist()))
        self.out = defaultdict(set); self.inn = defaultdict(set); self.und = defaultdict(set)
        for u, v in zip(src.tolist(), dst.tolist()):
            self.out[u].add(v); self.inn[v].add(u); self.und[u].add(v); self.und[v].add(u)
        self.outdeg = {u: len(s) for u, s in self.out.items()}
        self.indeg = {v: len(s) for v, s in self.inn.items()}
        self.nodes = set(self.und)


def neg_uniform(n, num_nodes, forbid, rng):
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
    appear = np.array(sorted(G.nodes))
    qs = np.quantile(indeg[appear], np.linspace(0, 1, n_bins + 1)[1:-1])
    bin_of = np.digitize(indeg, qs); bin_of[indeg == 0] = -1
    pools = {b: np.where(bin_of == b)[0] for b in np.unique(bin_of)}
    out, fallback = [], 0
    for u, v in pos:
        pool = pools[bin_of[v]]; ok = False
        for _ in range(50):
            w = int(pool[rng.integers(0, len(pool))])
            if w != u and (u, w) not in forbid:
                out.append((u, w)); ok = True; break
        if not ok:
            fallback += 1; out.append(neg_uniform(1, num_nodes, forbid, rng)[0])
    return out, fallback / len(pos)


def neg_twohop_time(pos, pos_t, G, forbid, first_seen, rng, num_nodes):
    out, fallback = [], 0
    dm_rng = np.random.default_rng(rng.integers(0, 2 ** 31))
    und_list = {u: list(s) for u, s in G.und.items()}
    for (u, v), t in zip(pos, pos_t):
        n1 = und_list.get(u); found = None
        if n1 and first_seen.get(u, np.inf) <= t:
            for _ in range(40):
                w = n1[rng.integers(0, len(n1))]; n2 = und_list[w]; x = n2[rng.integers(0, len(n2))]
                if x != u and x not in G.und[u] and (u, x) not in forbid and first_seen.get(x, np.inf) <= t:
                    found = (u, x); break
        if found is None:
            fallback += 1
        out.append(found)
    miss = [i for i, x in enumerate(out) if x is None]
    if miss:
        dm, _ = neg_degree_matched([pos[i] for i in miss], G, forbid, dm_rng, num_nodes)
        for i, x in zip(miss, dm):
            out[i] = x
    return out, fallback / len(pos)


def mrr_per_pos(pos_s, neg_s, rng, k=100):
    idx = rng.integers(0, len(neg_s), size=(len(pos_s), k)); ns = neg_s[idx]; p = pos_s.reshape(-1, 1)
    return 1.0 / (0.5 * ((ns > p).sum(1) + (ns >= p).sum(1)) + 1)


# ----------------------------------------------------------------- main -----------

@torch.no_grad()
def cat_scores(model, data, pairs, item_type, device):
    """P(trust) from CAT for arbitrary user pairs, mirroring CAT_main.test()."""
    model.eval()
    hg = C.to_dgl(data).to(device)
    item_id = torch.arange(1, data["item"].num_nodes)
    out = []
    pairs = torch.as_tensor(np.asarray(pairs).T, dtype=torch.long)
    for i in range(0, pairs.shape[1], 8192):
        o = model(hg, hg.ndata["x"], pairs[:, i:i + 8192].to(device), item_type, item_id)
        out.append(torch.softmax(o, 1)[:, 1].cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--out", default="../../../protocol/results_v2/e5")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-train", action="store_true", help="reuse models/cat_s<seed>.pkl if present")
    args0 = ap.parse_args()
    os.makedirs(args0.out, exist_ok=True)
    os.makedirs(C.root_path + "/models", exist_ok=True)

    # ---- CAT's own argument object with defaults (CAT_main.parameter_parser) ----
    import sys
    sys.argv = [sys.argv[0]]
    args = C.parameter_parser()
    args.epochs = args0.epochs
    args.device = torch.device(args0.device if torch.cuda.is_available() else "cpu")

    for seed in args0.seeds:
        f_out = os.path.join(args0.out, f"cat_hardneg_s{seed}.json")
        if os.path.exists(f_out):
            print("exists, skip:", f_out); continue
        args.seed = seed
        args.model_save_path = f"/models/cat_s{seed}.pkl"
        setup_seed(seed)

        # ======== CAT_main.main(), verbatim up to training ========
        graph, item_type = C.get_ori_graph(args)
        item_type = item_type.long()  # Windows numpy<2 gives int32; torch_scatter needs int64 index
        trustor = np.array(graph["user", "to", "user"].edge_index[0])
        trustee = np.array(graph["user", "to", "user"].edge_index[1])
        ts = np.array(graph["user", "to", "user"].edge_label_time)
        val_time, test_time = list(np.quantile(ts, [args.val_ratio, args.test_ratio]))
        total_node_set = set(np.unique(np.hstack([trustor, trustee])))
        import random
        mask_node_set = random.sample(list(set(trustor[ts > val_time]).union(set(trustee[ts > val_time]))),
                                      int(args.mask_ratio * len(total_node_set)))
        mask_tr_flag = np.isin(trustor, mask_node_set); mask_te_flag = np.isin(trustee, mask_node_set)
        none_node_flag = (1 - mask_tr_flag) * (1 - mask_te_flag)
        valid_train_flag = (ts <= val_time) * (none_node_flag > 0)
        train_node_set = set(trustor[valid_train_flag]).union(trustee[valid_train_flag])
        new_node_set = total_node_set - train_node_set
        valid_val_flag = (ts > val_time) * (ts <= test_time)
        valid_test_flag = ts > test_time
        train_data = C.get_dataset("train", graph, valid_train_flag, valid_val_flag, valid_test_flag, val_time, test_time, mask_node_set)
        val_data = C.get_dataset("val", graph, valid_train_flag, valid_val_flag, valid_test_flag, val_time, test_time, mask_node_set)
        test_data = C.get_dataset("test", graph, valid_train_flag, valid_val_flag, valid_test_flag, val_time, test_time, mask_node_set)
        train_data = C.creat_edge_time(train_data); val_data = C.creat_edge_time(val_data); test_data = C.creat_edge_time(test_data)
        user_emb = torch.from_numpy(np.loadtxt(C.root_path + '/code/uitiu_user_emb_meta_{}.txt'.format(args.emb_size))).type(torch.float32)
        item_emb = torch.from_numpy(np.loadtxt(C.root_path + '/code/uitiu_item_emb_meta_{}.txt'.format(args.emb_size))).type(torch.float32)
        train_data["user"].x = val_data["user"].x = test_data["user"].x = user_emb
        train_data["item"].x = val_data["item"].x = test_data["item"].x = item_emb
        edge_label_val, edge_label_index_val = C.negative_sampler(val_data["user", "to", "user"], val_data["user"].num_nodes, args.neg_seed)
        val_data["user", "to", "user"].edge_label_index = edge_label_index_val
        val_data["user", "to", "user"].edge_label = edge_label_val
        edge_label_test, edge_label_index_test = C.negative_sampler(test_data["user", "to", "user"], test_data["user"].num_nodes, args.neg_seed + 1)
        test_data["user", "to", "user"].edge_label_index = edge_label_index_test
        test_data["user", "to", "user"].edge_label = edge_label_test
        train_loader = LinkNeighborLoader(
            data=train_data, num_neighbors=args.num_neighbors,
            edge_label_index=(("user", "to", "user"), train_data["user", "to", "user"].edge_label_index),
            edge_label=train_data["user", "to", "user"].edge_label,
            edge_label_time=train_data["user", "to", "user"].edge_label_time,
            time_attr="node_time", temporal_strategy="last",
            neg_sampling_ratio=args.neg_sampling_ratio, batch_size=args.batch_size, shuffle=True)
        model = cat_model(num_layers=args.num_layers, in_dim=args.emb_size, hidden_dim=args.hidden_dim,
                          ntypes=["user", "item"], negative_slope=args.negative_slope, seed=args.seed).to(args.device)

        t0 = time.time()
        ckpt = C.root_path + args.model_save_path
        if args0.skip_train and os.path.exists(ckpt):
            cat_a = None
        else:
            res = C.train(model, train_loader, val_data, test_data, new_node_set, item_type, args)
            cat_a = dict(auc_old=res[0], acc_old=res[1], f1_old=res[2], ap_old=res[3], mrr_old=res[4],
                         auc_new=res[5], acc_new=res[6], f1_new=res[7], ap_new=res[8], mrr_new=res[9],
                         best_epoch=res[10], mean_epoch_time=res[11])
        train_s = time.time() - t0
        model.load_state_dict(torch.load(ckpt, map_location=args.device)["model"])   # best-val checkpoint
        # ======== end of CAT_main.main() ========

        # ---- hard negatives, identical sequence to e3_negsampling.py ----
        src, dst = trustor, trustee
        tr = valid_train_flag.astype(bool); va = valid_val_flag; te = valid_test_flag
        num_nodes = int(max(src.max(), dst.max())) + 1
        G = Graph(src[tr | va], dst[tr | va])
        pos = list(zip(src[te].tolist(), dst[te].tolist())); pos_t = ts[te]
        first_seen = {}
        for u, v, t in zip(src.tolist(), dst.tolist(), ts.tolist()):
            first_seen[u] = min(first_seen.get(u, np.inf), t); first_seen[v] = min(first_seen.get(v, np.inf), t)
        rng = np.random.default_rng(NEG_SEED); forbid = G.edges; forbid_all = forbid | set(pos)
        regimes = {"a_uniform_e3": (neg_uniform(len(pos), num_nodes, forbid, rng), 0.0),
                   "b_degree_matched": neg_degree_matched(pos, G, forbid_all, rng, num_nodes),
                   "c_twohop_time": neg_twohop_time(pos, pos_t, G, forbid_all, first_seen, rng, num_nodes)}
        observed = np.array([u in train_node_set and v in train_node_set for u, v in pos])
        pa = np.log1p(np.array([G.outdeg.get(u, 0) * G.indeg.get(v, 0) for u, v in pos]))

        # scores: CAT's P(trust) on positives and on each negative set
        sp = cat_scores(model, test_data, pos, item_type, args.device)
        rep = {"seed": seed, "cat_own_test_negatives": cat_a, "train_s": train_s,
               "spearman_score_logPA_observed": float(spearmanr(sp[observed], pa[observed]).statistic),
               "fallback": {k: v[1] for k, v in regimes.items()}, "regimes": {}}
        # also CAT's own (a) negatives re-scored with the same metric code, for a like-for-like row
        own = edge_label_index_test.numpy(); own_neg = [tuple(x) for x in own[:, edge_label_test.numpy() == 0].T.tolist()]
        regimes["a_cat_own"] = (own_neg, 0.0)
        from sklearn.metrics import average_precision_score, roc_auc_score
        for reg, (neg, _) in regimes.items():
            sn = cat_scores(model, test_data, neg, item_type, args.device)
            nob = np.array([u in train_node_set and v in train_node_set for u, v in neg])
            res = {}
            for subset, pm, nm in (("all", np.ones(len(pos), bool), np.ones(len(neg), bool)), ("observed", observed, nob), ("unobserved", ~observed, ~nob)):
                if pm.sum() == 0 or nm.sum() == 0:
                    continue
                y = np.r_[np.ones(pm.sum()), np.zeros(nm.sum())]; s = np.r_[sp[pm], sn[nm]]
                rr = mrr_per_pos(sp[pm], sn[nm], np.random.default_rng(NEG_SEED + 7))
                # bootstrap CI (B=1000) on AP and MRR
                b_rng = np.random.default_rng(NEG_SEED + 11); aps, mrrs = [], []
                P_, N_ = sp[pm], sn[nm]
                for _ in range(1000):
                    i = b_rng.integers(0, len(P_), len(P_)); j = b_rng.integers(0, len(N_), len(N_))
                    aps.append(average_precision_score(np.r_[np.ones(len(i)), np.zeros(len(j))], np.r_[P_[i], N_[j]])); mrrs.append(rr[i].mean())
                res[subset] = dict(AP=float(average_precision_score(y, s)), AUC=float(roc_auc_score(y, s)), MRR=float(rr.mean()),
                                   ci=dict(AP=[float(np.percentile(aps, 2.5)), float(np.percentile(aps, 97.5))],
                                           MRR=[float(np.percentile(mrrs, 2.5)), float(np.percentile(mrrs, 97.5))]),
                                   n_pos=int(pm.sum()), n_neg=int(nm.sum()), prevalence=float(pm.sum() / (pm.sum() + nm.sum())))
            rep["regimes"][reg] = res
            o = res["observed"]
            print(f"[seed {seed}] {reg:<18} observed: AP {o['AP']:.4f} [{o['ci']['AP'][0]:.3f},{o['ci']['AP'][1]:.3f}] "
                  f"AUC {o['AUC']:.4f} MRR {o['MRR']:.4f} (chance {o['prevalence']:.3f})", flush=True)
        print(f"[seed {seed}] spearman(CAT score, log PA) = {rep['spearman_score_logPA_observed']:.3f}; CAT own test: {cat_a}")
        with open(f_out, "w") as fh:
            json.dump(rep, fh, indent=1)
        print("wrote", f_out)


if __name__ == "__main__":
    main()
