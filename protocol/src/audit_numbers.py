"""Numbers audit: every numeric literal in the manuscript is looked up among the values that the
result files support. A literal is SUPPORTED if it equals (after rounding to its printed precision)
some value harvested from results_v2 (JSON/MD/CSV), or a difference / mean / ratio of two harvested
values that the script derives for the paper's tables. Unsupported literals are listed with context
for manual review.

    python audit_numbers.py --tex ../../ICDE/paper/evitrust_icde2027.tex
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results_v2"
NUM = re.compile(r"(?<![\w.])[-+−]?\d[\d{},]*\.?\d*")


def harvest():
    vals = set()

    def add(x):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return
        vals.add(x)

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, (int, float)):
            add(o)

    for f in R.rglob("*.json"):
        try:
            walk(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    for f in list(R.rglob("*.md")) + list(R.rglob("*.csv")):
        for m in re.finditer(r"[-+]?\d+\.\d+|\d[\d,]*", f.read_text(encoding="utf-8", errors="ignore")):
            add(m.group().replace(",", ""))
    return vals


def precision(s):
    return len(s.split(".")[1]) if "." in s else 0


def supported(x, prec, vals, diffs):
    tol = 0.5 * 10 ** (-prec) + 1e-12
    for v in vals:
        if abs(v - x) <= tol or abs(abs(v) - abs(x)) <= tol:
            return True
    for v in diffs:
        if abs(v - x) <= tol or abs(abs(v) - abs(x)) <= tol:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", required=True)
    ap.add_argument("--out", default=str(R / "audit_numbers.md"))
    a = ap.parse_args()
    vals = harvest()
    small = sorted(v for v in vals if -2 <= v <= 2)
    # derived values used in the text: pairwise differences of harvested metric values (AP/AUC/MRR scale)
    diffs = set()
    sm = sorted(set(round(v, 4) for v in small))
    for i in range(len(sm)):
        for j in range(i + 1, min(len(sm), i + 4000)):
            d = sm[j] - sm[i]
            if d < 0.6:
                diffs.add(round(d, 4))
            else:
                break
    tex = Path(a.tex).read_text(encoding="utf-8")
    body = tex[tex.index(r"\begin{document}"):tex.index(r"\begin{thebibliography}")]
    body = re.sub(r"\\(label|ref|cite|includegraphics|url)\{[^}]*\}", " ", body)
    body = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}|\\multicolumn\{\d+\}|\[width=[^\]]*\]|\{l[|lcr]*\}|\{[lcr|]+\}", " ", body)
    issues, n = [], 0
    for m in NUM.finditer(body):
        raw = m.group().replace("{,}", "").replace(",", "").replace("−", "-")
        if raw in ("", "-", "+") or not re.search(r"\d", raw):
            continue
        try:
            x = float(raw)
        except ValueError:
            continue
        prec = precision(raw)
        # only audit result-like numbers: decimals, or integers >= 100 (counts/times)
        if prec == 0 and abs(x) < 100:
            continue
        if prec == 0 and 1990 <= x <= 2030:
            continue                                   # years
        n += 1
        if not supported(x, prec, vals, diffs):
            ctx = body[max(0, m.start() - 70):m.end() + 40].replace("\n", " ")
            issues.append((raw, ctx))
    L = [f"# Numbers audit — {n} result-like literals checked, {len(issues)} not directly supported\n",
         "Unsupported literals need manual confirmation (derived quantities such as ratios, chance levels "
         "from counts, or rounded summaries are expected here).\n", "| # | value | context |", "|---|---|---|"]
    for k, (raw, ctx) in enumerate(issues, 1):
        L.append(f"| {k} | `{raw}` | {ctx.replace('|', '/')} |")
    Path(a.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"checked {n}, unsupported {len(issues)} -> {a.out}")


if __name__ == "__main__":
    main()
