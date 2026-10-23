"""E12 (review v2 N1) - re-score the saved CAT checkpoints on all test regimes, including the
unseen hard negatives (e) co-review and (f) active degree-matched (EVI_EXTRA_NEG=1).

Run from Reference code/CAT/code:
    EVI_EXTRA_NEG=1 python ../../../protocol/src/e12_rescore_cat.py --dataset epinions --out ../../../protocol/results_v2/e12/epinions
Checkpoints: models/cat_s<seed>.pkl (Epinions, E5) and models/cat_<dataset>_s<seed>.pkl (E7).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import e3_negsampling as N3                      # noqa: E402
import e5_cat_hardneg as E5                      # noqa: E402
import e6_evitrust as E6                         # noqa: E402
import e8_baselines as E8                        # noqa: E402
import CAT_main as C                             # noqa: E402
from CAT_model import cat_model                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(E6.DATASETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda")
    for seed in a.seeds:
        t0 = time.time()
        S, first_seen, G, pos, pos_t, regimes, obs, vpos = E8.setup(seed, a.dataset)
        sys.argv = [sys.argv[0]]; args = C.parameter_parser(); args.data_path = E6.DATASETS[a.dataset]["data_path"]
        graph, item_type = C.get_ori_graph(args); item_type = item_type.long()
        test_data = C.creat_edge_time(C.get_dataset("test", graph, S["tr"], S["va"], S["te"], S["val_time"], S["test_time"], []))
        n_u, n_i = graph["user"].num_nodes, graph["item"].num_nodes
        test_data["user"].x = torch.from_numpy(np.loadtxt(C.root_path + E6.DATASETS[a.dataset]["emb"])).float()[:n_u]
        test_data["item"].x = torch.from_numpy(np.loadtxt(C.root_path + E6.DATASETS[a.dataset]["item_emb"])).float()[:n_i]
        model = cat_model(num_layers=args.num_layers, in_dim=args.emb_size, hidden_dim=args.hidden_dim,
                          ntypes=["user", "item"], negative_slope=args.negative_slope, seed=seed).to(dev)
        K = int(item_type[1].max())
        if K != model.gate.numel():
            model.gate = torch.nn.Parameter(torch.ones(K, device=dev) / K)
        ck = f"/models/cat_s{seed}.pkl" if a.dataset == "epinions" else f"/models/cat_{a.dataset}_s{seed}.pkl"
        model.load_state_dict(torch.load(C.root_path + ck, map_location=dev)["model"])
        sp = E5.cat_scores(model, test_data, pos, item_type, dev)
        sn = {k: E5.cat_scores(model, test_data, v, item_type, dev) for k, v in regimes.items()}
        E8.report("cat", a.dataset, seed, sp, sn, pos, regimes, obs, dict(runtime_s=time.time() - t0, checkpoint=ck), a.out)


if __name__ == "__main__":
    main()
