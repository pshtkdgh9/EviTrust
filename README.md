# EviTrust: Evidence-Based Trust Prediction under Hard-Negative Evaluation

Artifact for the ICDE 2027 submission *"Beyond Popularity: Evidence-Based Trust Prediction under Hard-Negative Evaluation"* (`paper/`).

The repository contains all code, pre-registration, and per-pair results. **Datasets and the third-party CAT code are not redistributed**; see [Data](#data) for the original download links and where to place them.

## What is here

| Path | Content |
|---|---|
| `paper/` | Manuscript (`evitrust_icde2027.tex/.pdf`) and figures |
| `protocol/src/` | All experiment code (see table below) |
| `protocol/results_v2/` | Raw results: per-run JSON, per-pair score files (`scores_*.npz`), summary tables (`*_summary.md`) |
| `docs/HYPOTHESES_prereg_v2.md` | Pre-registration (written before the corresponding test results; in Korean) |
| `Reference code/` | Place the original CAT code here (instructions inside) |
| `trust dataset/`, `protocol/data_cat/` | Place / generate datasets here (instructions inside) |

### Code (`protocol/src/`)

| Script | Purpose | Paper |
|---|---|---|
| `e3_negsampling.py` | CAT split, negative generators (uniform, degree-matched, two-hop time-consistent), metrics, bootstrap | Sec. III-A, Table I |
| `e5_cat_hardneg.py` | Original CAT trained with its own code, scored on hard negatives (Epinions) | Table I |
| `e6_evitrust.py` | **EviTrust**: platform state, 37 evidence features, temporal label windows, mixed negatives, LightGBM, shared test regimes | Sec. III, IV |
| `prep_datasets.py` | CiaoDVD / wiki-RfA to CAT format + metapath2vec embeddings | Sec. IV-A |
| `e7_cat_run.py`, `e6b_cat_scores.py`, `e12_rescore_cat.py` | CAT on CiaoDVD / wiki-RfA; re-scoring saved CAT checkpoints | Tables III-X |
| `e8_baselines.py`, `models.py` | Heuristic-LR, Guardian, TrustGuard, TGN | Tables III-VI, X |
| `e9_mixed_neural.py` | Controls: CAT-H / TGN-H trained with EviTrust's mixed negatives | Table VII |
| `e10_stream.py` | Incremental `StreamState`: equivalence check, update/query latency, synthetic scaling | Sec. III-D, Table XII, Fig. 3 |
| `e11_cat_diag.py` | wiki-RfA new-user (Simpson's paradox) and CiaoDVD reciprocity diagnostics | Table IX |
| `e13_real_scale.py` | Streaming state on the full Epinions / Ciao releases | Sec. IV-G |
| `analyze_*.py`, `stats_for_paper_v3.py`, `make_figures.py`, `summarize_e5.py` | Tables, paired bootstrap, figures | all |

## Setup

Tested on Windows 11, Python 3.10, RTX 3060 (12 GB), 64 GB RAM.

```bash
conda create -n evitrust python=3.10 -y && conda activate evitrust
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric==2.5.1 pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install dgl==2.2.1 -f https://data.dgl.ai/wheels/cu121/repo.html
pip install --no-deps torchdata==0.7.1
pip install -r requirements.txt
```

## Data

| Dataset | Used for | Original source | Place at |
|---|---|---|---|
| Epinions (CAT release) | main benchmark | shipped with CAT: https://github.com/Jieerbobo/CAT (`data/Epinions`) | `Reference code/CAT/data/Epinions/` |
| CiaoDVD (Guo et al., ASONAM 2014) | second dataset | https://guoguibing.github.io/librec/datasets/CiaoDVD.zip | `trust dataset/CiaoDVD/` (`trusts.txt`, `movie-ratings.txt`) |
| Wikipedia Requests for Adminship (West et al., TACL 2014) | third dataset, real distrust | https://snap.stanford.edu/data/wiki-RfA.html | `trust dataset/wiki-RfA/wiki-RfA.txt` |
| Epinions and Ciao full releases (Tang et al.) | streaming scale test only (Sec. IV-G) | https://www.cse.msu.edu/~tangjili/trust.html | `trust dataset/Epinions/`, `trust dataset/ciao/` (`rating.txt`, `trustnetwork.txt`) |

Then build the converted CiaoDVD / wiki-RfA data and their metapath2vec embeddings:

```bash
cd protocol/src && python prep_datasets.py      # writes protocol/data_cat/{ciaodvd,wikirfa}/
```

## Reproducing

All experiment scripts run from the CAT code directory, because CAT's `CAT_main.py` resolves paths from the working directory:

```bash
git clone https://github.com/Jieerbobo/CAT "Reference code/CAT"
cp protocol/src/e5_cat_hardneg.py "Reference code/CAT/code/"
cd "Reference code/CAT/code"

# CAT (original code) on Epinions with hard negatives; ~1.5-3 h per seed on an RTX 3060
python e5_cat_hardneg.py --seeds 42 43 44 --out ../../../protocol/results_v2/e5
# CAT on CiaoDVD / wiki-RfA
python ../../../protocol/src/e7_cat_run.py --dataset ciaodvd --seeds 42 43 44 --out ../../../protocol/results_v2/e7/ciaodvd
# EviTrust-U and the three candidates of EviTrust-V (rho 1:0:0, 3:1:1, 1:1:1)
python ../../../protocol/src/e6_evitrust.py --dataset ciaodvd --seeds 42 43 44 --variant uniform_only --out ../../../protocol/results_v2/e7/ciaodvd
python ../../../protocol/src/e6_evitrust.py --dataset ciaodvd --seeds 42 43 44 --mix 3 1 1 --select mean --tag m311mean --out ../../../protocol/results_v2/e7/ciaodvd
python ../../../protocol/src/e6_evitrust.py --dataset ciaodvd --seeds 42 43 44 --variant full --out ../../../protocol/results_v2/e7/ciaodvd
# baselines and controls
python ../../../protocol/src/e8_baselines.py --dataset ciaodvd --method tgn --out ../../../protocol/results_v2/e8/ciaodvd
python ../../../protocol/src/e9_mixed_neural.py --dataset ciaodvd --method cat_h --out ../../../protocol/results_v2/e9/ciaodvd
# unseen hard negatives (e)/(f): prefix with EVI_EXTRA_NEG=1 and write to results_v2/e12/<dataset>
# streaming state
python ../../../protocol/src/e10_stream.py verify --dataset epinions
python ../../../protocol/src/e10_stream.py scale
```

Summary tables: `cd protocol/src && python analyze_e8.py && python analyze_e11_m5.py && python analyze_e12.py`.

Notes
- On Windows with NumPy < 2, CAT's `get_ori_graph` returns an int32 `item_type`; `e5_cat_hardneg.py` casts it to int64 (values unchanged). CAT's own files are never modified; for CiaoDVD / wiki-RfA our wrappers only set the data path, the embeddings, and the size of CAT's category gate.
- All methods score identical test pairs per dataset and seed (`e6_evitrust.test_regimes`); existing results are skipped when their JSON exists.
- Neural baselines are not bit-for-bit deterministic on GPU; re-runs differ by at most about 0.01 AP.

## Citation

Anonymous submission under review at ICDE 2027.
