"""E7 data preparation: CiaoDVD and wiki-RfA -> CAT's file format + metapath2vec embeddings.

Writes  protocol/data_cat/<name>/
    user_user_time.txt   "trustor trustee time"            (CAT_main.get_ori_graph format)
    user_item_time.txt   "user,item,category,rating,help,time"
    oppose_time.txt      (wiki-RfA only) "voter candidate time" for oppose votes
    user_emb_meta_64.txt / item_emb_meta_64.txt   metapath2vec (user-item-type-item-user),
                         trained only on user-item interactions before the test boundary
    meta.json            id maps sizes, split boundaries, number of categories

Ids: users 1..N, items 1..M (contiguous, as CAT_model assumes item ids start at 1),
categories 1..K.

CiaoDVD has no trust timestamps. Trust edges get a random order placed after every review
(time = T_end + rank), so CAT's 70/85% quantile split becomes a uniformly random 70/15/15
split and every review is visible to every split. (The earlier CHTrust proxy
max(last rating date) is not used: it makes the split depend on future activity.)

wiki-RfA: trust = support vote (VOT=1) at its real time; item = election (candidate, year),
category = year; every vote (support/neutral/oppose) is a user-item interaction with
rating = vote+2.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "protocol" / "data_cat"


def write(name, uu, ui, extra=None, n_cat=None):
    """Restrict to users that occur in the trust graph (CAT sizes the user table from trust ids)
    and re-index users 1..N and items 1..M contiguously.

    CAT/DGL infer a node table's size from the largest id that has an edge in the graph being
    built, so the largest ids go to the users/items that appear FIRST (they are present in the
    train, val and test graphs alike): ids are assigned in reverse order of first appearance.
    Item categories are numbered by frequency (most frequent = largest id), because CAT sizes
    its category context from the largest category id among the items in each batch."""
    d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    users = np.unique(np.r_[uu.u, uu.v])
    n_ui0 = len(ui)
    ui = ui[ui.u.isin(users)].copy()
    first = pd.concat([uu[["u", "t"]], uu[["v", "t"]].rename(columns={"v": "u"}), ui[["u", "t"]]]).groupby("u")["t"].min()
    order = first.loc[users].sort_values(kind="stable", ascending=False).index.to_numpy()
    umap = pd.Series(np.arange(1, len(order) + 1), index=order)
    ifirst = ui.groupby("i")["t"].min().sort_values(kind="stable", ascending=False)
    imap = pd.Series(np.arange(1, len(ifirst) + 1), index=ifirst.index.to_numpy())
    cfreq = ui.drop_duplicates("i")["c"].value_counts(ascending=True)
    cmap = pd.Series(np.arange(1, len(cfreq) + 1), index=cfreq.index.to_numpy())
    ui["c"] = cmap[ui.c].to_numpy(); n_cat = len(cfreq)
    ui["u"] = umap[ui.u].to_numpy(); ui["i"] = imap[ui.i].to_numpy()
    uu = uu.assign(u=umap[uu.u].to_numpy(), v=umap[uu.v].to_numpy())
    if extra is not None:
        extra = extra[extra.u.isin(users) & extra.v.isin(users)]
        extra = extra.assign(u=umap[extra.u].to_numpy(), v=umap[extra.v].to_numpy())
    print(f"{name}: kept {len(ui)}/{n_ui0} user-item rows of trust-graph users")
    np.savetxt(d / "user_user_time.txt", uu[["u", "v", "t"]].to_numpy(np.int64), fmt="%d")
    np.savetxt(d / "user_item_time.txt", ui[["u", "i", "c", "r", "h", "t"]].to_numpy(np.int64), fmt="%d", delimiter=",")
    if extra is not None:
        np.savetxt(d / "oppose_time.txt", extra[["u", "v", "t"]].to_numpy(np.int64), fmt="%d")
    ts = uu["t"].to_numpy(np.float64)
    val_t, test_t = np.quantile(ts, [0.7, 0.85])
    meta = dict(n_users=int(max(uu.u.max(), uu.v.max(), ui.u.max())), n_items=int(ui.i.max()), n_cat=int(n_cat or ui.c.max()),
                n_trust=int(len(uu)), n_trust_unique=int(len(uu[["u", "v"]].drop_duplicates())), n_ui=int(len(ui)),
                val_time=float(val_t), test_time=float(test_t),
                n_oppose=int(len(extra)) if extra is not None else 0)
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    print(name, meta)
    return d, meta


def prep_ciaodvd(src, seed=0):
    r = pd.read_csv(src / "movie-ratings.txt", header=None, names=["u", "item", "c", "rid", "r", "date"])
    t = pd.read_csv(src / "trusts.txt", header=None, names=["u", "v", "val"])
    print("ciaodvd trust values:", t.val.value_counts().to_dict())
    t = t[(t.val == 1) & (t.u != t.v)]
    r["t"] = pd.to_datetime(r["date"], errors="coerce").astype("int64") // 10**9
    r = r[r.t > 0]
    users = np.unique(np.r_[t.u, t.v, r.u]); umap = {x: i + 1 for i, x in enumerate(users)}
    items = np.unique(r["item"]); imap = {x: i + 1 for i, x in enumerate(items)}
    ui = pd.DataFrame(dict(u=r.u.map(umap), i=r["item"].map(imap), c=r.c, r=r.r, h=1, t=r.t))
    rng = np.random.default_rng(seed)
    t_end = int(ui.t.max()) + 86400
    uu = pd.DataFrame(dict(u=t.u.map(umap), v=t.v.map(umap), t=t_end + rng.permutation(len(t))))
    return write("ciaodvd", uu, ui, n_cat=int(r.c.max()))


MONTHS = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                           "september", "october", "november", "december"])}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})


def parse_dat(s):
    m = re.search(r"(\d{1,2}):(\d{2}),?\s+(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", s)
    if not m:
        m2 = re.search(r"(\d{1,2}):(\d{2}),?\s+([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", s)
        if not m2:
            return None
        hh, mm, mon, dd, yy = m2.groups()
    else:
        hh, mm, dd, mon, yy = m.groups()
    mo = MONTHS.get(mon.lower()) or MONTHS.get(mon.lower()[:3])
    if not mo:
        return None
    try:
        return int(dt.datetime(int(yy), mo, int(dd), int(hh), int(mm), tzinfo=dt.timezone.utc).timestamp())
    except ValueError:
        return None


def prep_wikirfa(src):
    recs, cur = [], {}
    with open(src / "wiki-RfA.txt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                if cur:
                    recs.append(cur); cur = {}
                continue
            k, _, v = line.partition(":")
            cur[k] = v
    if cur:
        recs.append(cur)
    df = pd.DataFrame(recs)
    n0 = len(df)
    df["t"] = df["DAT"].fillna("").map(parse_dat)
    df["VOT"] = pd.to_numeric(df["VOT"], errors="coerce"); df["YEA"] = pd.to_numeric(df["YEA"], errors="coerce")
    df = df[(df.SRC.fillna("") != "") & (df.TGT.fillna("") != "") & df.t.notna() & df.VOT.notna() & df.YEA.notna() & (df.SRC != df.TGT)]
    print(f"wiki-RfA: {n0} votes, {len(df)} usable (SRC/TGT/time/vote present, no self vote); "
          f"votes {df.VOT.value_counts().to_dict()}")
    users = np.unique(np.r_[df.SRC, df.TGT]); umap = {x: i + 1 for i, x in enumerate(users)}
    df["u"] = df.SRC.map(umap); df["v"] = df.TGT.map(umap); df["t"] = df.t.astype(np.int64)
    el = df[["TGT", "YEA"]].drop_duplicates().sort_values(["YEA", "TGT"]).reset_index(drop=True)
    el["i"] = np.arange(1, len(el) + 1)
    df = df.merge(el, on=["TGT", "YEA"])
    y0 = int(df.YEA.min())
    # wiki-RfA has no item categories: a single category (CAT needs every category in every batch)
    ui = pd.DataFrame(dict(u=df.u, i=df.i, c=1, r=(df.VOT + 2).astype(int), h=1, t=df.t))
    uu = df[df.VOT == 1][["u", "v", "t"]].sort_values("t")
    opp = df[df.VOT == -1][["u", "v", "t"]].sort_values("t")
    return write("wikirfa", uu, ui, extra=opp, n_cat=int(ui.c.max()))


def metapath2vec(d, meta, dim=64, epochs=5, seed=0, walk_length=20, context_size=5, walks_per_node=5, suffix=""):
    """user-item-type-item-user metapath2vec on interactions before the test boundary."""
    import torch
    from torch_geometric.nn import MetaPath2Vec
    torch.manual_seed(seed)
    ui = np.loadtxt(d / "user_item_time.txt", delimiter=",").astype(np.int64)
    ui = ui[ui[:, 5] < meta["test_time"]]
    n_u, n_i, n_c = meta["n_users"] + 1, meta["n_items"] + 1, meta["n_cat"] + 1
    it = pd.DataFrame(ui[:, 1:3], columns=["i", "c"]).drop_duplicates("i")
    # PyG MetaPath2Vec needs an out-edge for every node on every metapath relation: users without
    # (pre-test) interactions, items without interactions and empty types are attached to a dummy
    # item 0 / type 0, so their walks are uninformative instead of crashing the sampler.
    no_u = np.setdiff1d(np.arange(n_u), ui[:, 0]); no_i = np.setdiff1d(np.arange(n_i), ui[:, 1])
    ui = np.vstack([ui[:, :2], np.c_[no_u, np.zeros_like(no_u)], np.c_[np.zeros_like(no_i), no_i]])
    no_c = np.setdiff1d(np.arange(n_c), it.c)
    it = pd.concat([it, pd.DataFrame(dict(i=np.r_[0, np.setdiff1d(np.arange(n_i), it.i)], c=0)),
                    pd.DataFrame(dict(i=0, c=no_c))]).drop_duplicates()
    edge = {("user", "rates", "item"): torch.as_tensor(ui[:, :2].T),
            ("item", "rated_by", "user"): torch.as_tensor(ui[:, [1, 0]].T),
            ("item", "in", "type"): torch.as_tensor(it[["i", "c"]].to_numpy().T),
            ("type", "has", "item"): torch.as_tensor(it[["c", "i"]].to_numpy().T)}
    num_nodes = {"user": n_u, "item": n_i, "type": n_c}
    mp = [("user", "rates", "item"), ("item", "in", "type"), ("type", "has", "item"), ("item", "rated_by", "user")]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetaPath2Vec(edge, embedding_dim=dim, metapath=mp, walk_length=walk_length, context_size=context_size, walks_per_node=walks_per_node,
                         num_negative_samples=5, num_nodes_dict=num_nodes, sparse=True).to(dev)
    loader = model.loader(batch_size=256, shuffle=True, num_workers=0)
    opt = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
    for ep in range(epochs):
        tot = 0.0
        for pos_rw, neg_rw in loader:
            opt.zero_grad(); loss = model.loss(pos_rw.to(dev), neg_rw.to(dev)); loss.backward(); opt.step(); tot += loss.item()
        print(f"  metapath2vec {d.name} epoch {ep} loss {tot / max(len(loader), 1):.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        np.savetxt(d / f"user_emb_meta_{dim}{suffix}.txt", model("user").cpu().numpy())
        np.savetxt(d / f"item_emb_meta_{dim}{suffix}.txt", model("item").cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ciaodvd", "wikirfa"])
    ap.add_argument("--src", default=str(ROOT / "trust dataset"))
    a = ap.parse_args()
    src = Path(a.src)
    for name in a.datasets:
        d, meta = prep_ciaodvd(src / "CiaoDVD") if name == "ciaodvd" else prep_wikirfa(src / "wiki-RfA")
        metapath2vec(d, meta)


if __name__ == "__main__":
    main()
