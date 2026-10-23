"""Numbers for the U/V-centred manuscript (EviTrust-V = validation-selected rho, prereg_v2 §E11).
Run from Reference code/CAT/code (needs the CAT split for the Spearman check)."""
import json
import os
import statistics as st
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import analyze_e8 as A          # noqa: E402
import e8_baselines as E8       # noqa: E402

R = A.R
out = {}


def vload(ds):
    d = R / "e6" if ds == "epinions" else R / "e7" / ds
    return {s: dict(np.load(d / f"scores_evitrust_v_s{s}.npz")) for s in (42, 43, 44)}


for ds, dsn in A.DS:
    D = A.load(ds); cat = D["cat"]; V = vload(ds)
    regs = [r for r, _ in A.REG if r in cat[42]]
    o = {}
    for sub in (["observed", "all"] if ds == "wikirfa" else ["observed"]):
        m = {r: [A.metric(V[s], cat, s, r, sub) for s in (42, 43, 44)] for r in regs}
        o[sub] = {r: dict(AP=st.mean(x["AP"] for x in m[r]), sd=st.stdev(x["AP"] for x in m[r]),
                          AUC=st.mean(x["AUC"] for x in m[r]), MRR=st.mean(x["MRR"] for x in m[r])) for r in regs}
    # paired V vs controls (observed)
    pc = {}
    for c in ("cat_h", "tgn_h"):
        if c in D:
            pc[c] = {r: [A.paired(V[s], D[c][s], cat, s, r, "observed") for s in (42, 43, 44)] for r in regs[:3]}
            pc[c] = {r: dict(mean=st.mean(x[0] for x in v), up=int(sum(x[1] > 0 for x in v)), dn=int(sum(x[2] < 0 for x in v))) for r, v in pc[c].items()}
    o["paired_controls"] = pc
    out[ds] = o

# Spearman with log PA on Epinions observed positives (V)
sp = []
for s in (42, 43, 44):
    S, fs, G, pos, pos_t, regimes, obs, vpos = E8.setup(s, "epinions")
    pa = np.log1p(np.array([G.outdeg.get(a, 0) * G.indeg.get(b, 0) for a, b in pos])); m = obs(pos)
    sp.append(float(spearmanr(vload("epinions")[s]["pos"][m], pa[m]).statistic))
out["spearman_V_epinions"] = sp
json.dump(out, open(R / "e11" / "stats_paper_v3.json", "w"), indent=1)
print(json.dumps(out, indent=1))
