"""E6b - per-pair CAT scores from the E5 best-validation checkpoints (models/cat_s<seed>.pkl),
for paired bootstrap tests against EviTrust on the *same* test pairs.

Run from Reference code/CAT/code:
    python ../../../protocol/src/e6b_cat_scores.py --seeds 42 43 44 --out ../../../protocol/results_v2/e6
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import e3_negsampling as N3                      # noqa: E402
import e5_cat_hardneg as E5                      # noqa: E402  (cat_scores)
import e6_evitrust as E6                         # noqa: E402  (split + CAT's own negatives, same RNG order)
import CAT_main as C                             # noqa: E402
from CAT_model import cat_model                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--out", default="../../../protocol/results_v2/e6")
    a0 = ap.parse_args()
    dev = torch.device("cuda")
    for seed in a0.seeds:
        S = E6.cat_split_and_own_negatives(seed)
        sys.argv = [sys.argv[0]]
        args = C.parameter_parser()
        graph, item_type = C.get_ori_graph(args)
        item_type = item_type.long()
        test_data = C.get_dataset("test", graph, S["tr"], S["va"], S["te"], S["val_time"], S["test_time"], [])
        test_data = C.creat_edge_time(test_data)
        test_data["user"].x = torch.from_numpy(np.loadtxt(C.root_path + "/code/uitiu_user_emb_meta_64.txt")).float()
        test_data["item"].x = torch.from_numpy(np.loadtxt(C.root_path + "/code/uitiu_item_emb_meta_64.txt")).float()
        model = cat_model(num_layers=args.num_layers, in_dim=args.emb_size, hidden_dim=args.hidden_dim,
                          ntypes=["user", "item"], negative_slope=args.negative_slope, seed=seed).to(dev)
        model.load_state_dict(torch.load(C.root_path + f"/models/cat_s{seed}.pkl", map_location=dev)["model"])
        src, dst, ts, n = S["src"], S["dst"], S["ts"], S["n_users"]
        tvm = S["tr"] | S["va"]
        G = N3.Graph(src[tvm], dst[tvm], ts[tvm])
        pos = list(zip(src[S["te"]].tolist(), dst[S["te"]].tolist())); pos_t = ts[S["te"]]
        first_seen = {}
        for a, b, t in zip(src.tolist(), dst.tolist(), ts.tolist()):
            first_seen[a] = min(first_seen.get(a, np.inf), t); first_seen[b] = min(first_seen.get(b, np.inf), t)
        rng = np.random.default_rng(E6.NEG_SEED); forbid = G.edges; forbid_all = forbid | set(pos)
        regimes = {"a_uniform_e3": N3.neg_uniform(len(pos), n, forbid, rng),
                   "b_degree_matched": N3.neg_degree_matched(pos, G, forbid_all, rng, n)[0],
                   "c_twohop_time": N3.neg_twohop_time(pos, pos_t, G, forbid_all, first_seen, rng, n)[0],
                   "a_cat_own": S["own_test_neg"]}
        tn = S["train_nodes"]
        out = {"pos": E5.cat_scores(model, test_data, pos, item_type, dev),
               "observed": np.array([a in tn and b in tn for a, b in pos])}
        for k, neg in regimes.items():
            out[k] = E5.cat_scores(model, test_data, neg, item_type, dev)
            nm = np.array([a in tn and b in tn for a, b in neg]); out[f"{k}_obs"] = nm
            print(f"[seed {seed}] CAT {k:<18} observed AP {N3.fast_ap(out['pos'][out['observed']], out[k][nm]):.4f}", flush=True)
        np.savez_compressed(os.path.join(a0.out, f"scores_cat_s{seed}.npz"), **out)


if __name__ == "__main__":
    main()
