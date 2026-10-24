"""Cell-level audit of the main result tables: recompute every cell from the per-pair score files
and compare with the manuscript.

    python audit_tables.py --tex ../../ICDE/paper/evitrust_icde2027.tex
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

import numpy as np

import analyze_e8 as A

R = A.R
SEEDS = (42, 43, 44)


def table_rows(tex, label):
    i = tex.index(r"\label{" + label + "}")
    j = tex.index(r"\end{tabular}", i)
    rows = {}
    for line in tex[i:j].split("\n"):
        if "&" not in line or "\\\\" not in line:
            continue
        cells = [c.strip() for c in line.split("\\\\")[0].split("&")]
        key = re.sub(r"\\textbf\{|\}|\\eviU\{|\\eviV\{|\$|\\pm|~\\cite\{[^}]*", "", cells[0]).strip()
        vals = [re.sub(r"\\textbf\{([^}]*)\}", r"\1", c).replace("$\\pm$", "±").replace("$-$", "-") for c in cells[1:]]
        rows[key] = vals
    return rows


def num(c):
    m = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", c)
    return float(m.group()) if m else None


def vload(ds):
    d = R / "e6" if ds == "epinions" else R / "e7" / ds
    return {s: dict(np.load(d / f"scores_evitrust_v_s{s}.npz")) for s in SEEDS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", required=True)
    a = ap.parse_args()
    tex = Path(a.tex).read_text(encoding="utf-8")
    bad, n = [], 0

    def check(table, row, col, got, exp, prec):
        nonlocal n
        n += 1
        if got is None or abs(got - round(exp, prec)) > 0.5 * 10 ** (-prec) + 1e-9:
            bad.append((table, row, col, got, round(exp, prec)))

    names = {"CAT (re-trained)": "cat", "Heuristic-LR": "heur_lr", "Heur.-LR": "heur_lr", "Guardian": "guardian",
             "TrustGuard": "trustguard", "TGN": "tgn", "EviTrust-U": "uniform_only", "EviTrust-V": "V", "CAT": "cat"}
    M = {}
    for ds, _ in A.DS:
        M[ds] = A.load(ds); M[ds]["V"] = vload(ds)
    met = lambda ds, m, r, sub, k: [A.metric(M[ds][m][s], M[ds]["cat"], s, r, sub)[k] for s in SEEDS]

    # Table III (main): per dataset AP(±sd) AUC MRR, observed
    rows = table_rows(tex, "tab:main")
    for label, cells in rows.items():
        lab = label.replace("\\", "").strip()
        m = names.get(lab) or names.get(lab.replace("{}", ""))
        key = next((k for k in names if lab.startswith(k.split(" ")[0]) and lab == k), None)
        if lab.startswith("CAT (reported"):
            continue
        m = names.get(lab)
        if m is None:
            continue
        for di, (ds, _) in enumerate(A.DS):
            ap_ = met(ds, m, "a_cat_own", "observed", "AP"); auc = met(ds, m, "a_cat_own", "observed", "AUC"); mrr = met(ds, m, "a_cat_own", "observed", "MRR")
            check("III", lab, f"{ds} AP", num(cells[3 * di]), st.mean(ap_), 4)
            check("III", lab, f"{ds} AUC", num(cells[3 * di + 1]), st.mean(auc), 4)
            check("III", lab, f"{ds} MRR", num(cells[3 * di + 2]), st.mean(mrr), 4)
    # Table IV (hard)
    rows = table_rows(tex, "tab:hard")
    cols = [("epinions", "b_degree_matched"), ("epinions", "c_twohop_time"), ("ciaodvd", "b_degree_matched"), ("ciaodvd", "c_twohop_time"),
            ("wikirfa", "b_degree_matched"), ("wikirfa", "c_twohop_time"), ("wikirfa", "d_oppose")]
    for label, cells in rows.items():
        m = names.get(label.replace("\\", "").strip())
        if m is None:
            continue
        for ci, (ds, r) in enumerate(cols):
            check("IV", label, f"{ds} {r}", num(cells[ci]), st.mean(met(ds, m, r, "observed", "AP")), 3)
    # Table X (wiki-RfA all pairs)
    rows = table_rows(tex, "tab:rfa")
    for label, cells in rows.items():
        m = names.get(label.replace("\\", "").strip())
        if m is None:
            continue
        exp = [st.mean(met("wikirfa", m, "a_cat_own", "all", "AP")), st.mean(met("wikirfa", m, "a_cat_own", "all", "AUC")),
               st.mean(met("wikirfa", m, "a_cat_own", "all", "MRR"))] + [st.mean(met("wikirfa", m, r, "all", "AP")) for r in ("b_degree_matched", "c_twohop_time", "d_oppose")]
        for ci, e in enumerate(exp):
            check("X", label, f"col{ci}", num(cells[ci]), e, 3)
    # Table VI (unseen) from E12 summary
    e12 = json.loads((R / "e12" / "E12_summary.json").read_text())
    rows = table_rows(tex, "tab:unseen")
    keymap = {"CAT": "cat", "Heur.-LR": "heur_lr", "Guardian": "guardian", "TrustGuard": "trustguard", "TGN": "tgn", "EviTrust-U": "U", "EviTrust-V": "V"}
    for label, cells in rows.items():
        m = keymap.get(label.replace("\\", "").strip())
        if m is None:
            continue
        for di, ds in enumerate(("epinions", "ciaodvd", "wikirfa")):
            for k, r in enumerate(("e_coreview", "f_active_dm")):
                check("VI", label, f"{ds} {r}", num(cells[2 * di + k]), e12[ds]["table"][m][r], 3)
    # Table VII (control)
    rows = table_rows(tex, "tab:control")
    cmap = {"CAT (uniform)": "cat", "CAT-H (mixed)": "cat_h", "TGN (uniform)": "tgn", "TGN-H (mixed)": "tgn_h", "EviTrust-V (mixed)": "V"}
    for label, cells in rows.items():
        m = cmap.get(label.replace("\\", "").strip())
        if m is None:
            continue
        for di, ds in enumerate(("epinions", "ciaodvd")):
            for k, r in enumerate(("a_cat_own", "b_degree_matched", "c_twohop_time")):
                check("VII", label, f"{ds} {r}", num(cells[3 * di + k]), st.mean(met(ds, m, r, "observed", "AP")), 3)
    print(f"checked {n} cells, mismatches {len(bad)}")
    for b in bad:
        print("  MISMATCH", b)


if __name__ == "__main__":
    main()
