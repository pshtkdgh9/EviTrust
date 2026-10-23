"""Summarize E5 CAT hard-negative results (observed regime) across seeds into Markdown + CSV."""
import csv, json, statistics as st, sys
from pathlib import Path

res_dir = Path(sys.argv[1])
out_md = Path(sys.argv[2])
REG = ["a_cat_own", "b_degree_matched", "c_twohop_time"]
EXTRA = ["a_uniform_e3"]

runs = []
for f in sorted(res_dir.glob("cat_hardneg_s*.json")):
    runs.append(json.loads(f.read_text(encoding="utf-8")))
if not runs:
    sys.exit("no result files")

def obs(r, k):
    return r["regimes"][k]["observed"]

def ms(xs):
    return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0)

L = []
L.append("# E5 — CAT 원본 모델의 난이도 음성 평가 (observed 구간)\n")
L.append(f"Seed: {', '.join(str(r['seed']) for r in runs)}  ·  출처: `protocol/results_v2/e5/cat_hardneg_s*.json`\n")

L.append("## 1. Seed별 observed AP (요청 요약표)\n")
L.append("| Seed | a_cat_own AP | b_degree_matched AP | c_twohop_time AP | spearman(score, log PA) | 최적 epoch |")
L.append("|---|---|---|---|---|---|")
for r in runs:
    cells = [f"{obs(r,k)['AP']:.4f}" for k in REG]
    L.append(f"| {r['seed']} | " + " | ".join(cells)
             + f" | {r['spearman_score_logPA_observed']:.3f} | {r['cat_own_test_negatives'].get('best_epoch','-')} |")
cells = []
for k in REG:
    m, s = ms([obs(r, k)["AP"] for r in runs]); cells.append(f"**{m:.4f} ± {s:.4f}**")
m, s = ms([r["spearman_score_logPA_observed"] for r in runs])
L.append("| 평균 ± 표준편차 | " + " | ".join(cells) + f" | **{m:.3f} ± {s:.3f}** | |")
L.append("")

L.append("## 2. 음성 집합별 상세 (observed): AP [95% bootstrap CI], AUC, MRR, 우연 수준 AP (= 양성 비율)\n")
L.append("| Seed | 음성 집합 | AP | AP 95% CI | AUC | MRR | 우연 AP | AP − 우연 |")
L.append("|---|---|---|---|---|---|---|---|")
for r in runs:
    for k in REG + EXTRA:
        o = obs(r, k); lo, hi = o["ci"]["AP"]
        L.append(f"| {r['seed']} | {k} | {o['AP']:.4f} | [{lo:.3f}, {hi:.3f}] | {o['AUC']:.4f} | {o['MRR']:.4f} | {o['prevalence']:.3f} | {o['AP']-o['prevalence']:+.3f} |")
L.append("")

L.append("## 3. Seed 평균 ± 표준편차 (observed)\n")
L.append("| 음성 집합 | AP | AUC | MRR | 우연 AP | a_cat_own 대비 ΔAP |")
L.append("|---|---|---|---|---|---|")
base = st.mean([obs(r, "a_cat_own")["AP"] for r in runs])
for k in REG + EXTRA:
    ap = ms([obs(r, k)["AP"] for r in runs]); auc = ms([obs(r, k)["AUC"] for r in runs])
    mrr = ms([obs(r, k)["MRR"] for r in runs]); ch = st.mean([obs(r, k)["prevalence"] for r in runs])
    L.append(f"| {k} | {ap[0]:.4f} ± {ap[1]:.4f} | {auc[0]:.4f} ± {auc[1]:.4f} | {mrr[0]:.4f} ± {mrr[1]:.4f} | {ch:.3f} | {ap[0]-base:+.4f} |")
L.append("")

L.append("## 4. CAT 자체 테스트 지표와 실행 정보\n")
L.append("| Seed | AP old | AP new | AUC old | AUC new | 최적 epoch | epoch당 평균(초) | 학습 시간(초) | c_twohop 대체 표본 비율 |")
L.append("|---|---|---|---|---|---|---|---|---|")
for r in runs:
    c = r["cat_own_test_negatives"]
    L.append(f"| {r['seed']} | {c['ap_old']:.4f} | {c['ap_new']:.4f} | {c['auc_old']:.4f} | {c['auc_new']:.4f} | {c.get('best_epoch','-')} | {c.get('mean_epoch_time',0):.0f} | {r.get('train_s',0):.0f} | {r.get('fallback',{}).get('c_twohop_time',0):.3f} |")
L.append("")

L.append("## 5. 실행 환경 / 참고\n")
L.append("- 환경 `cat_e5` (Windows에 새로 만듦): Python 3.10, torch 2.3.0+cu121, torch-geometric 2.5.1, pyg-lib 0.4.0, torch-scatter 2.1.2, torch-sparse 0.6.18, dgl 2.2.1+cu121, torchdata 0.7.1. "
         "ICDE/02_cat_hardneg_run.md §2의 버전(torch 2.0.0 / pyg-lib 0.2.0 / dgl 2.1.0)과 다름. torch 2.2 이하용 pyg-lib Windows 빌드가 없고, LinkNeighborLoader의 시간순(disjoint) 샘플링에는 pyg-lib가 필수이기 때문.")
L.append("- `e5_cat_hardneg.py`: `C.get_ori_graph(args)` 바로 뒤에 `item_type = item_type.long()` 한 줄 추가 (Windows의 numpy<2는 int32를 만들고 torch_scatter는 int64 인덱스만 받음). `protocol/src/`와 `Reference code/CAT/code/` 두 복사본에 모두 적용. CAT_main.py / CAT_model.py는 수정하지 않음.")
L.append("- `c_twohop 대체 표본 비율` = 해당 음성 집합에서 2-hop 후보가 부족해 대체 샘플링으로 채운 음성의 비율 (JSON의 `fallback` 값).")

out_md.write_text("\n".join(L) + "\n", encoding="utf-8")

with open(out_md.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["seed", "regime", "AP", "AP_ci_lo", "AP_ci_hi", "AUC", "MRR", "chance_AP", "n_pos", "n_neg", "spearman_score_logPA", "best_epoch"])
    for r in runs:
        for k in REG + EXTRA:
            o = obs(r, k)
            w.writerow([r["seed"], k, o["AP"], *o["ci"]["AP"], o["AUC"], o["MRR"], o["prevalence"], o["n_pos"], o["n_neg"],
                        r["spearman_score_logPA_observed"], r["cat_own_test_negatives"].get("best_epoch")])
print(f"wrote {out_md} and {out_md.with_suffix('.csv')} ({len(runs)} seeds)")
