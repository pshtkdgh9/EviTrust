"""E13 (review v2 N5) - streaming EviTrust state on the full public Epinions and Ciao releases
(Tang et al.), which are larger than the benchmark graphs.

    python e13_real_scale.py --out ../results_v2/e13

Streams
  Epinions full : rating.txt (user, product, category, rating, helpfulness, date) + trustnetwork.txt
  Ciao full     : rating.txt reviews (author -> product, date) PLUS every helpfulness vote attached to a
                  review (rater -> product of the review, at the review date; raters carry no time)
                  + trustnetwork.txt
Trust records in both releases carry no timestamp; for this latency/memory measurement they are placed
at uniformly random times within the review period. Accuracy is NOT evaluated here.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e10_stream as E10                                  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "trust dataset")
HELP = {"not helpful": 1, "somewhat helpful": 2, "helpful": 3, "very helpful": 4, "most helpful": 5, "exceptional": 5}


def day(s):
    try:
        d, m, y = s.strip().split(".")
        return dt.datetime(int(y), int(m), int(d), tzinfo=dt.timezone.utc).timestamp()
    except Exception:
        return None


def parse(name):
    sep = ":::" if name == "epinions" else "::::"
    umap, imap, cmap = {}, {}, {}
    gid = lambda m, k: m.setdefault(k, len(m))
    U, I, C, Rt, H, T = [], [], [], [], [], []
    n_help = 0
    with open(os.path.join(ROOT, "Epinions" if name == "epinions" else "ciao", "rating.txt"), encoding="utf-8", errors="replace") as fh:
        for line in fh:
            p = line.rstrip("\n").split(sep)
            if len(p) < 6:
                continue
            t = day(p[5])
            if t is None:
                continue
            u = gid(umap, p[0].strip()); i = gid(imap, p[1].strip()); c = gid(cmap, p[2].strip())
            try:
                r = float(p[3]); r = r / 10 if r > 5 else r
            except ValueError:
                r = 0.0
            U.append(u); I.append(i); C.append(c); Rt.append(r); H.append(HELP.get(p[4].strip().lower(), 0)); T.append(t)
            if name == "ciao" and len(p) >= 8:                  # helpfulness votes: rater:score:::rater:score...
                for tok in p[7].replace("</endperson>", "").split(":::"):
                    if ":" in tok:
                        a, b = tok.split(":", 1)
                        if a.strip().isdigit():
                            U.append(gid(umap, a.strip())); I.append(i); C.append(c)
                            Rt.append(float(b) if b.strip().replace(".", "").isdigit() else 0.0); H.append(0); T.append(t)
                            n_help += 1
    tr = []
    with open(os.path.join(ROOT, "Epinions" if name == "epinions" else "ciao", "trustnetwork.txt"), encoding="utf-8") as fh:
        for line in fh:
            p = [x for x in line.strip().split(":") if x]
            if len(p) >= 2:
                tr.append((gid(umap, p[0]), gid(umap, p[1])))
    reviews = np.array([U, I, C, Rt, H, T], dtype=np.float64).T
    return umap, reviews, tr, n_help


def run(name, out, n_pairs, seed=0):
    t0 = time.time()
    umap, rv, tr, n_help = parse(name)
    rng = np.random.default_rng(seed)
    lo, hi = rv[:, 5].min(), rv[:, 5].max()
    tts = rng.uniform(lo, hi, len(tr))
    trust = [(u, v, t) for (u, v), t in zip(tr, tts.tolist())]
    reviews = [tuple(x) for x in rv[:, [0, 1, 2, 3, 4, 5]].tolist()]
    reviews = [(int(a), int(b), int(c), r, h, t) for a, b, c, r, h, t in reviews]
    t_parse = time.time() - t0
    del rv; gc.collect(); m0 = E10.rss_mb()
    st = E10.StreamState()
    t1 = time.perf_counter(); lt, lr = E10.ingest(st, trust, reviews, math.inf, math.inf); t_ing = time.perf_counter() - t1
    gc.collect(); m1 = E10.rss_mb()
    cut = hi + 1
    n = len(umap)
    rand = list(zip(rng.integers(0, n, n_pairs).tolist(), rng.integers(0, n, n_pairs).tolist()))
    edge = [tr[i] for i in rng.integers(0, len(tr), n_pairs)]
    qr, qe = [], []
    for (u, v), (x, y) in zip(rand, edge):
        a = time.perf_counter(); st.features(u, v, cut); qr.append(time.perf_counter() - a)
        a = time.perf_counter(); st.features(x, y, cut); qe.append(time.perf_counter() - a)
    rep = dict(dataset=f"{name}_full", users=n, items_seen=int(len({x[1] for x in reviews})), trust_records=len(trust),
               interactions=len(reviews), helpfulness_votes=n_help, parse_s=t_parse, ingest_s=t_ing,
               update_trust=E10.pct(lt), update_interaction=E10.pct(lr), query_random_pair=E10.pct(qr), query_trust_pair=E10.pct(qe),
               state_rss_mb=m1 - m0, dense_state_gb_required=4 * n * n * 4 / 2**30,
               note="trust records have no timestamp in these releases; placed at uniform random times for this measurement")
    os.makedirs(out, exist_ok=True)
    json.dump(rep, open(os.path.join(out, f"real_scale_{name}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "note"}, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["epinions", "ciao"])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_v2", "e13"))
    ap.add_argument("--n-pairs", type=int, default=20000)
    a = ap.parse_args()
    for d in a.datasets:
        run(d, a.out, a.n_pairs)


if __name__ == "__main__":
    main()
