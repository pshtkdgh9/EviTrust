"""E7 - original CAT on CiaoDVD / wiki-RfA (and Epinions) with the E5 procedure.

Run from Reference code/CAT/code:
    python ../../../protocol/src/e7_cat_run.py --dataset ciaodvd --seeds 42 43 44 --out ../../../protocol/results_v2/e7/ciaodvd

Identical to e5_cat_hardneg.py (CAT_main's split, training, validation-AP early stopping and
own test negatives, imported unchanged) except for three dataset-specific inputs that
CAT_main hard-codes for Epinions and that are set here without editing CAT's files:
    * args.data_path                     -> protocol/data_cat/<dataset>
    * metapath2vec user/item embeddings  -> protocol/data_cat/<dataset>/*_emb_meta_64.txt
    * model.gate (size 25 "for the Epinions dataset") -> number of item categories
Test negatives come from e6_evitrust.test_regimes, so CAT and EviTrust score identical pairs.
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
sys.path.insert(0, os.getcwd())
import e3_negsampling as N3                       # noqa: E402
import e5_cat_hardneg as E5                       # noqa: E402  (cat_scores)
import e6_evitrust as E6                          # noqa: E402  (DATASETS, test_regimes)
import CAT_main as C                              # noqa: E402
from CAT_model import cat_model                   # noqa: E402
from torch_geometric.loader import LinkNeighborLoader  # noqa: E402
from utils import setup_seed                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(E6.DATASETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    a0 = ap.parse_args()
    os.makedirs(a0.out, exist_ok=True); os.makedirs(C.root_path + "/models", exist_ok=True)
    sys.argv = [sys.argv[0]]
    args = C.parameter_parser()
    args.epochs = a0.epochs
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    D = E6.DATASETS[a0.dataset]
    args.data_path = D["data_path"]

    for seed in a0.seeds:
        f_out = os.path.join(a0.out, f"cat_s{seed}.json")
        if os.path.exists(f_out):
            print("exists, skip:", f_out); continue
        args.seed = seed
        args.model_save_path = f"/models/cat_{a0.dataset}_s{seed}.pkl"
        setup_seed(seed)
        # ======== CAT_main.main() / e5_cat_hardneg.py, same order ========
        graph, item_type = C.get_ori_graph(args)
        item_type = item_type.long()
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
        new_node_set = total_node_set - train_node_set
        valid_val_flag = (ts > val_time) * (ts <= test_time)
        valid_test_flag = ts > test_time
        train_data = C.get_dataset("train", graph, valid_train_flag, valid_val_flag, valid_test_flag, val_time, test_time, mask_node_set)
        val_data = C.get_dataset("val", graph, valid_train_flag, valid_val_flag, valid_test_flag, val_time, test_time, mask_node_set)
        test_data = C.get_dataset("test", graph, valid_train_flag, valid_val_flag, valid_test_flag, val_time, test_time, mask_node_set)
        train_data = C.creat_edge_time(train_data); val_data = C.creat_edge_time(val_data); test_data = C.creat_edge_time(test_data)
        user_emb = torch.from_numpy(np.loadtxt(C.root_path + D["emb"])).type(torch.float32)
        item_emb = torch.from_numpy(np.loadtxt(C.root_path + D["item_emb"])).type(torch.float32)
        assert user_emb.shape[0] >= graph["user"].num_nodes and item_emb.shape[0] >= graph["item"].num_nodes
        user_emb = user_emb[:graph["user"].num_nodes]; item_emb = item_emb[:graph["item"].num_nodes]
        train_data["user"].x = val_data["user"].x = test_data["user"].x = user_emb
        train_data["item"].x = val_data["item"].x = test_data["item"].x = item_emb
        lab_v, idx_v = C.negative_sampler(val_data["user", "to", "user"], val_data["user"].num_nodes, args.neg_seed)
        val_data["user", "to", "user"].edge_label_index = idx_v; val_data["user", "to", "user"].edge_label = lab_v
        lab_t, idx_t = C.negative_sampler(test_data["user", "to", "user"], test_data["user"].num_nodes, args.neg_seed + 1)
        test_data["user", "to", "user"].edge_label_index = idx_t; test_data["user", "to", "user"].edge_label = lab_t
        train_loader = LinkNeighborLoader(
            data=train_data, num_neighbors=args.num_neighbors,
            edge_label_index=(("user", "to", "user"), train_data["user", "to", "user"].edge_label_index),
            edge_label=train_data["user", "to", "user"].edge_label,
            edge_label_time=train_data["user", "to", "user"].edge_label_time,
            time_attr="node_time", temporal_strategy="last",
            neg_sampling_ratio=args.neg_sampling_ratio, batch_size=args.batch_size, shuffle=True)
        model = cat_model(num_layers=args.num_layers, in_dim=args.emb_size, hidden_dim=args.hidden_dim,
                          ntypes=["user", "item"], negative_slope=args.negative_slope, seed=args.seed).to(args.device)
        K = int(item_type[1].max())                  # rows of CAT's category context
        if K != model.gate.numel():
            model.gate = torch.nn.Parameter(torch.ones(K, device=args.device) / K, requires_grad=True)
        t0 = time.time()
        res = C.train(model, train_loader, val_data, test_data, new_node_set, item_type, args)
        cat_a = dict(auc_old=res[0], acc_old=res[1], f1_old=res[2], ap_old=res[3], mrr_old=res[4],
                     auc_new=res[5], acc_new=res[6], f1_new=res[7], ap_new=res[8], mrr_new=res[9],
                     best_epoch=res[10], mean_epoch_time=res[11])
        train_s = time.time() - t0
        model.load_state_dict(torch.load(C.root_path + args.model_save_path, map_location=args.device)["model"])
        # ======== scoring on the shared test negatives ========
        src, dst = trustor, trustee; n = int(graph["user"].num_nodes)
        tvm = valid_train_flag.astype(bool) | valid_val_flag
        G = N3.Graph(src[tvm], dst[tvm], ts[tvm])
        pos = list(zip(src[valid_test_flag].tolist(), dst[valid_test_flag].tolist())); pos_t = ts[valid_test_flag].astype(np.float64)
        first_seen = {}
        for x, y, t in zip(src.tolist(), dst.tolist(), ts.tolist()):
            first_seen[x] = min(first_seen.get(x, np.inf), t); first_seen[y] = min(first_seen.get(y, np.inf), t)
        own_neg = [tuple(x) for x in idx_t.numpy()[:, lab_t.numpy() == 0].T.tolist()]
        S = dict(own_test_neg=own_neg, test_time=float(test_time))
        regimes = E6.test_regimes(S, G, pos, pos_t, first_seen, n, a0.dataset)
        tn = train_node_set
        observed = np.array([x in tn and y in tn for x, y in pos])
        sp = E5.cat_scores(model, test_data, pos, item_type, args.device)
        pa = np.log1p(np.array([G.outdeg.get(x, 0) * G.indeg.get(y, 0) for x, y in pos]))
        rep = dict(dataset=a0.dataset, seed=seed, cat_own_test_negatives=cat_a, train_s=train_s, gate_size=K,
                   spearman_score_logPA_observed=float(spearmanr(sp[observed], pa[observed]).statistic), regimes={})
        scores = {"pos": sp, "observed": observed}
        for reg, neg in regimes.items():
            sn = E5.cat_scores(model, test_data, neg, item_type, args.device)
            nob = np.array([x in tn and y in tn for x, y in neg])
            scores[reg] = sn; scores[f"{reg}_obs"] = nob
            out = {}
            for subset, pm, nm in (("all", np.ones(len(pos), bool), np.ones(len(neg), bool)), ("observed", observed, nob)):
                if pm.sum() == 0 or nm.sum() == 0:
                    continue
                m_, rr = N3.metrics(sp[pm], sn[nm], np.random.default_rng(E6.NEG_SEED + 7))
                m_["ci"] = N3.bootstrap(sp[pm], sn[nm], rr, np.random.default_rng(E6.NEG_SEED + 11))
                m_.update(n_pos=int(pm.sum()), n_neg=int(nm.sum()), prevalence=float(pm.sum() / (pm.sum() + nm.sum())))
                out[subset] = m_
            rep["regimes"][reg] = out
            o = out.get("observed", out["all"])
            print(f"[{a0.dataset} seed {seed}] CAT {reg:<18} observed: AP {o['AP']:.4f} AUC {o['AUC']:.4f} MRR {o['MRR']:.4f} "
                  f"(n_neg {o['n_neg']}, chance {o['prevalence']:.3f})", flush=True)
        np.savez_compressed(os.path.join(a0.out, f"scores_cat_s{seed}.npz"), **scores)
        with open(f_out, "w") as fh:
            json.dump(rep, fh, indent=1)
        print("wrote", f_out, flush=True)


if __name__ == "__main__":
    main()
