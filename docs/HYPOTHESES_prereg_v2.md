# 사전 등록 v2 — ICDE 반복 1 (E1·E2·E3), 실험 실행 전 고정

> 작성: 2026-10-15, **실험 실행 전**. 결과를 본 뒤 기준을 바꾸지 않기 위해 먼저 기록합니다.
> 근거: `ICDE/00_candidates.md`(선택안 C1+C3+C2), `ARGUMENT_v2.md`, `HYPOTHESES_RESULTS.md`
> 실행 환경: 클라우드 작업공간(CPU 2코어, torch 2.14.1+cpu, numpy 2.5.3, scikit-learn 1.9.1). 이 환경에서 나온 수치를 이번 반복의 기준으로 삼습니다.

## 공통 검정 (ARGUMENT_v2 §1과 동일)

- 같은 분할에서 두 방법의 지표 차이: paired bootstrap(B=1,000, 같은 재표본 엣지) → 부트스트랩 sd
- 모델 seed가 여럿이면 sd = √(부트스트랩 분산[첫 seed 예측] + var_A/n + var_B/n), 양측 정규 검정, α=0.05
- 모델·epoch·임계값 선택은 **검증셋**만. 테스트셋 선택 금지
- QWK(순서형)와 MCC(명목형)를 항상 병기. "개선"은 QWK 유의 개선 + MCC 유의 악화 없음

---

## E1. 학습 시 목표 누수 제거(H5)의 사전 등록 확증

**배경**: HYPOTHESES_RESULTS §2 — 역할 분리 집계(`opinion`/`inn_opinion`)가 학습 엣지 자신의 라벨을 입력에 포함한 채 손실을 계산. 손실 엣지를 메시지 전달 그래프에서 제외(h5, 마스킹 50%)하면 Advogato 전체 +0.022(p=0.005), OTC cold-start 0.011→0.160. 사후 가설이었으므로 **새 분할·새 seed**에서 확증합니다.

**개입**: `h5` = 매 epoch 학습 엣지의 50%를 목표로 지정, 목표 엣지는 입력 그래프에서 제외하고 손실만 계산. `base` = 전체 엣지로 입력·손실.

**새 데이터**
- Advogato·PGP: 쌍 인지 분할 **seed 1, 2, 3, 4** (기존 결과는 seed 0). 분할 seed s에 모델 seed s를 짝지음 → 분할마다 base/h5 쌍이 하나씩
- BitcoinOTC·Alpha: 시간순 분할은 고정이므로 **모델 seed 3, 4, 5, 6** (기존은 0, 1, 2)

**E1b (계보 일반성)**: 같은 개입을 **Guardian(1층)** 에도 적용. 변형 `g_base`, `g_h5`.

**E1c (통제)**: `base_half` = 매 epoch 학습 엣지의 50%에만 손실을 계산하되 **입력 그래프는 전체 유지**. h5와 base_half의 차이가 "입력에서 제외" 효과이고, base_half와 base의 차이가 "손실 엣지 수 감소" 효과. E1c는 TrustGuard, seed/분할은 E1과 동일.

**가설**
- H-E1: h5 − base의 QWK > 0 (유의), MCC 유의 악화 없음
- H-E1b: g_h5 − g_base도 같은 방향 (계보 전체의 결함)
- H-E1c: h5 − base_half > 0 이고 base_half − base ≈ 0 (효과는 누수 제거에서 옴)
- 메커니즘: 증거 희박 계층(Bitcoin E0·E1 / Advogato·PGP 학습 in-degree 0~5)의 QWK 개선이 풍부 계층(E3)보다 큼

**채택 기준 (데이터셋 단위)**
- Advogato·PGP: 4개 분할 중 **3개 이상**에서 h5−base QWK가 paired bootstrap으로 유의 개선, **0개**에서 MCC 유의 악화. 보조로 4개 분할 차이의 평균과 sd 보고
- Bitcoin: seed 3~6의 결합 검정(부트스트랩 ⊕ seed 분산)으로 QWK 유의 개선, MCC 유의 악화 없음
- **전체 채택**: 4개 데이터셋 중 **2개 이상**이 데이터셋 기준을 만족하고, 메커니즘(희박 > 풍부)이 채택 데이터셋에서 관찰됨
- **조건부**: 1개 데이터셋만 만족, 또는 메커니즘 불일치 → 논문에서 "진단 발견"이 아닌 "분석·한계"
- **기각**: 그 외. 결과는 음성 결과 절로 보고

**예상 (기록용)**: Advogato 채택, OTC 경계, PGP 기각(풍부 계층 53%), Alpha 불명. 즉 전체 "조건부"가 가장 가능성 높음. 이 예상이 틀리는 쪽(2개 이상 채택)이면 B1을 Tier A로 승격.

**설정**: TrustGuard layers (32,64,32), Guardian (64,), Adam lr 0.01 wd 1e-5, weighted CE, epochs 120(PGP 80), eval-every 2, 검증 QWK로 epoch 선택. 기존 `hyp.py`와 동일.

---

## E2. 비용–정확도 경계: 증분 노드 통계 추정기 vs GNN

**배경**: ARGUMENT_v2 §3.2 / HYPOTHESES_RESULTS §1.3 — 역할 분리 GNN의 핵심 신호가 노드 평균(평판·관대함)이며, Advogato·PGP에서 6피처 순서형 LR이 GNN QWK의 98~102%. ICDE 독자를 위해 이를 **비용 축**으로 재서술합니다.

**추정기 (`stream_estimator.py`)**: 노드별 상태 (수신 등급 합, 수신 개수, 발신 등급 합, 발신 개수)만 유지. 엣지 (u→v, 등급 r) 도착 시 4개 값을 O(1) 갱신. 엣지 피처 = [수축 trustee 평균, 수축 trustor 평균, log1p(수신 개수), log1p(발신 개수), trustee 관측 여부, trustor 관측 여부]를 **엣지 도착 직전 상태**에서 계산(학습 엣지는 자연히 leave-one-out). 예측 = Frank&Hall 순서형 LR(C는 검증셋 QWK로 선택). 학습 = 학습 엣지 1회 스캔 + LR 적합. Bitcoin은 시간순 스트림, Advogato·PGP는 학습 엣지 순서(seed로 섞음).

**비교 대상**: Guardian·TrustGuard base (E1의 분할 seed 0 / Bitcoin 시간순, 모델 seed 0·1·2 — `hyp2.py`로 재실행해 검증 예측·시간을 함께 저장).

**측정 (4 데이터셋 × 3 seed)**
- 정확도: QWK, MCC, MAE
- 비용: (a) 학습 벽시계 시간 (b) 테스트 엣지당 추론 지연 µs (GNN은 인코딩 1회 + 점수, 추정기는 조회) (c) 신규 엣지 1건 반영 비용: 추정기는 상태 갱신 시간, GNN은 전체 재인코딩 시간(재학습 없이 그래프만 갱신하는 가장 관대한 가정) (d) 최대 RSS(MB)
- 모두 같은 CPU 2코어, 단일 프로세스, 3회 반복 중앙값

**하이브리드**: trustee 증거량(Bitcoin: 인과 카운트, Advogato·PGP: 학습 in-degree) < τ 이면 TrustGuard, 아니면 추정기. τ ∈ {1, 3, 6, 11, 21}을 **검증셋 QWK**로 선택. 평균 비용 = 라우팅 비율 가중 추론 비용.

**사전 고정 주장과 판정**
- C-E2a: Advogato·PGP에서 추정기 QWK ≥ 최고 GNN QWK × 0.95, 학습 비용 ≤ GNN 학습 비용 × 0.01. (판정: QWK는 seed 평균, 비용은 중앙값)
- C-E2b: Bitcoin에서는 추정기가 GNN에 유의하게 뒤진다 (ARGUMENT_v2 C3: OTC Δ≈0.12) — 이것이 "GNN이 필요한 영역"의 정의
- C-E2c: 하이브리드는 4개 데이터셋 모두에서 TrustGuard 대비 QWK 유의 악화 없이 평균 추론 비용을 50% 이하로 줄인다 (판정: 결합 검정 n.s. 또는 개선, 비용은 중앙값)
- 어느 주장도 결과를 본 뒤 임계값(0.95, 0.01, 50%)을 바꾸지 않음. 미달이면 그대로 보고

---

## E3. 음성 샘플링과 학습 없는 휴리스틱: CAT의 Epinions 프로토콜 재검

**배경**: Research_directions §2 F1 — CAT Epinions는 99.91% 양방향·역방향 타임스탬프 동일. CIKM R3 w3 — 1:1 무작위 음성 샘플링은 낙관적.

**CAT 프로토콜 재현 (`CAT_main.py` 기준)**: 신뢰 엣지 타임스탬프의 70%/85% 분위수로 train/val/test 시간순 분할(mask_ratio=0). 테스트 음성 = train+val 엣지에 없는 균일 무작위 쌍, 양성과 1:1(PyG `negative_sampling`, seed 2025). 지표: AUC, AP(양성+음성), MRR(양성마다 음성 100개 무작위, 평균 순위). "observed" = 양 끝점이 학습 노드 집합에 있는 테스트 엣지(CAT·TCHG 표의 Epinions(observed)에 해당).

**점수 함수 (학습 없음, train+val 그래프만 사용)**: 상호성(역방향 엣지 존재 1/0), preferential attachment(out(u)·in(v)), 공통 이웃(무방향), Adamic-Adar, Jaccard, 그리고 상호성+정규화 PA 합. 동률은 CAT의 mrr 정의대로 낙관/비관 평균.

**음성 샘플링 3종**
- (a) CAT 방식: 균일 무작위 1:1
- (b) 차수 일치: 각 양성 (u,v)에 대해 u는 그대로, v를 in-degree 분위수가 같은 노드 중에서 무작위(비연결 보장)
- (c) 2-hop 제한 + 시간 정합: v를 u의 2-hop 이웃(train+val 그래프) 중 비연결 노드에서 무작위, u·v 모두 양성 엣지 시각 이전에 등장한 노드. 2-hop 후보가 없으면 (b)로 폴백하고 폴백 비율 보고

**사전 고정 주장과 판정** (bootstrap 95% CI, B=1,000, 테스트 양성 재표본)
- C-E3a: (a)에서 상호성 단독 휴리스틱의 AP ≥ CAT 보고 AP(0.9407, CIKM 리뷰 R4 인용) × 0.90. 보조로 TCHG 표의 CAT Epinions(observed) MRR 0.6758과 휴리스틱 MRR 병기
- C-E3b: (b)와 (c)에서 모든 휴리스틱의 AP가 (a) 대비 0.10 이상 하락 (CI 하한 기준)
- C-E3c (포렌식 재확인): 양방향 비율 ≥ 99.5%, 역방향 타임스탬프 동일 비율 ≥ 99.5%
- 미달이면 그대로 보고. 특히 C-E3a가 미달이면 "휴리스틱이 SOTA에 근접"이라는 서사를 철회

**한계 (미리 기록)**: CAT 자체를 재학습하지 않으므로 (b)(c)에서 CAT의 하락폭은 측정하지 못함. 휴리스틱의 하락이 CAT의 하락을 함의하지는 않음 → 논문에서는 "프로토콜이 측정하는 것"에 대한 주장으로 한정.

---

## 실행 순서와 산출물

1. E3 (학습 불필요, 수 분) → `results_v2/e3/`
2. E2 추정기 + base 재실행 (`hyp2.py`, 분할 seed 0, 모델 seed 0~2) → `results_v2/e2/`
3. E1 (`hyp2.py`, 분할 seed 1~4 / 모델 seed 3~6, 변형 base·h5·base_half·g_base·g_h5) → `results_v2/e1/`
4. `analyze_v2.py` 로 위 기준을 기계적으로 적용 → `ITERATION_01_RESULTS.md`

결과를 본 뒤 추가하는 실험은 "사후 탐색"으로 별도 표기합니다.

---

## 반복 2 추가 — E4. CAT-lite 재학습 (E3 결과 확인 후, E4 실행 **전** 기록, 2026-10-16)

**동기**: E3는 학습 없는 휴리스틱만 다뤄 "CAT 자체가 난이도 음성에서 얼마나 떨어지는가"를 비워 두었습니다. CAT 전체(DGL·PyG·metapath2vec 재학습)는 CPU 환경에서 불가하므로, CAT가 배포한 **사전학습 metapath2vec 사용자 임베딩**(`CAT/code/uitiu_user_emb_meta_64.txt`)과 CAT의 **MLP 예측층**(hidden 64 → 2클래스)을 그대로 쓰는 "CAT-lite"를 CAT 방식으로 학습합니다. 이는 CAT에서 이종 attention·시간 인코딩·맥락 집계를 뺀 것으로, CAT의 수치를 재현하면 그 수치가 임베딩(=구조·인기도)에서 온다는 증거가 됩니다.

**학습 (CAT_main.py 기준)**: 학습 양성 = train 엣지, 학습 음성 = 매 epoch 균일 무작위 쌍 1:1 (PyG LinkNeighborLoader의 neg_sampling_ratio=1과 동일하게 기존 엣지 여부를 검사하지 않음), CE 손실, Adam lr 1e-3, wd 5e-4, batch 256, 최대 50 epoch, **검증(observed) AP**로 모델 선택·조기 종료(patience 10). seed 3개.
**변형 E4-dm**: 학습 음성을 차수 일치 샘플로 바꿈(그 외 동일).
**평가**: E3와 동일한 테스트 음성 집합 (a)(b)(c) (seed 2025 → 동일 샘플), observed 부분집합, AP/AUC/MRR + bootstrap CI.

**사전 고정 주장**
- C-E4a: CAT-lite의 (a) observed AP ≥ 0.93 이고 MRR ≥ 0.60 (CAT 보고 0.9407 / 0.6758의 각각 99% / 89%) — 이종 attention 없이 임베딩+MLP만으로 헤드라인이 재현된다
- C-E4b: (b)와 (c)에서 AP가 (a) 대비 **0.30 이상** 하락 (seed 평균, (a) CI 하한 − (b) CI 상한 기준)
- C-E4c: E4-dm(차수 일치 음성으로 재학습)의 (b) AP ≤ 0.75 — 난이도 음성으로 학습해도 CAT 수준의 수치는 회복되지 않는다
- 메커니즘: 테스트 쌍에서 CAT-lite 점수와 log(1+PA)의 Spearman 상관 ≥ 0.5 — 학습된 점수는 차수 대리변수다
- 미달 시 그대로 보고. 특히 C-E4a 미달이면 "CAT 수치는 attention·맥락 모듈에서 온다"는 반대 결론을 기록

---

## 반복 3 추가 — E5. CAT 원본 모델의 난이도 음성 평가 (E4 결과 확인 후, E5 실행 **전** 기록, 2026-10-16)

**설정**: `CAT_main.py`의 데이터 로딩·분할·학습·검증 AP 기반 모델 선택을 **그대로 import**해 CAT을 학습(seed 42·43·44, epochs 50, 기본 하이퍼파라미터)하고, CAT이 저장한 best-val 체크포인트를 (a) CAT 자체 음성, (a′) E3와 동일 seed의 균일 음성, (b) 차수 일치, (c) 2-hop+시간 정합 음성에서 채점. (b)(c)는 `e3_negsampling.py`와 **동일한 코드·동일한 난수 순서(seed 2025)** 로 생성되므로 휴리스틱·CAT-lite와 쌍 비교 가능. 실행 환경: 사용자 로컬 PC(RTX 3060), 스크립트 `src/e5_cat_hardneg.py`.

**사전 고정 주장**
- C-E5a (재현): CAT (a) observed AP ∈ [0.92, 0.96] (보고치 0.9407 ± 0.02), MRR ≥ 0.62
- C-E5b: (b)에서 AP ≤ (a) − 0.30, (c)에서 AP ≤ (a) − 0.40 (seed 평균; CI 기준으로도 보고)
- C-E5c: (b)(c)에서 CAT의 AP가 PA 단독(0.538 / 0.401)보다 **0.05 이상** 높지 않다 — 즉 CAT이 난이도 음성에서 휴리스틱 이상의 신뢰 신호를 회복하지 못한다
- 메커니즘: observed 양성에서 CAT 점수와 log PA의 Spearman ≥ 0.5
- 미달 시 그대로 보고. C-E5c가 미달(CAT이 휴리스틱보다 0.05 이상 높음)이면 "CAT은 인기도 이상의 신호를 일부 학습한다"로 A6를 완화

---

## E6. EviTrust — 인기도 지름길에 강건한 근거 기반 신뢰 예측 (방법 개선, 2026-10-16 실행 전 기록)

**배경**: E3·E4·E5 — CAT의 Epinions 헤드라인 AP(0.94)는 차수 곱으로 재현되고, CAT 원본도 차수 일치 음성 (b)에서 0.579, 2-hop+시간 음성 (c)에서 0.439로 우연(0.453/0.399) 근처까지 떨어진다(E5, seed 42·43·44). 기존 논문(CHTrust 초고)의 대표 비교 대상 수치는 CAT Epinions AP 0.9407 / AUC 0.9688.

**방법**: 쌍 (u,v)에 대해 **예측 시점 이전 정보만으로** 계산한 근거 피처 + GBDT(LightGBM).
- 구조: log 입·출차수(양 끝), PA, 방향 2-hop 경로 수(u→w→v), 3-hop 경로 수, 공동 피신뢰자/공동 신뢰자, 무방향 CN·AA·RA·Jaccard, 역방향 엣지, 경로 수의 차수 정규화 버전
- 행동(리뷰): 리뷰 수, 공동 리뷰 상품 수, **v가 먼저·u가 나중에 리뷰한 공동 상품 수(읽기 순서 신호)**와 반대 방향, 공동 상품 평점 차이, 카테고리 분포 코사인, v의 평균 평점·helpfulness
- 시간: 각 끝점의 마지막 활동으로부터 경과 시간, 최근 창의 입·출차수 증가
- 임베딩: CAT 배포 metapath2vec 코사인/내적 (CAT와 동일 입력, 절제 대상)
- **학습**: 학습 기간(≤ val_time) 안에서 시간 창 2개 (q50,q60], (q60,q70]의 엣지를 양성으로, 창 시작 시점 그래프로 피처 계산(누수 차단). 음성은 양성당 무작위 1 + 차수 일치 1 + 2-hop 1 (혼합 난이도). 검증: CAT 검증 그래프(신뢰 ≤ val_time, 리뷰 < val_time), val 양성 vs 3종 음성. 하이퍼파라미터·트리 수는 **검증 3종 평균 AP**로만 선택
- **테스트**: CAT 테스트 그래프(신뢰 train+val, 리뷰 < test_time), E5와 **동일한 음성 집합**(a_cat_own: CAT_main 난수 순서 재현, a_uniform_e3·b·c: seed 2025 동일 생성기). seed 42·43·44(학습 음성·GBDT seed)

**사전 고정 주장 (observed 부분집합, CAT = E5 seed 평균)**
- C-E6a (표준 프로토콜 비열위): a_cat_own AP ≥ CAT − 0.005 (0.9327 이상). 우위면 함께 보고
- C-E6b (난이도 음성 우위): b_degree_matched AP ≥ CAT + 0.05 (0.629 이상)
- C-E6c (난이도 음성 우위): c_twohop_time AP ≥ CAT + 0.05 (0.489 이상)
- 검정: 세 seed 모두에서 기준 충족 + CAT 점수를 같은 체크포인트로 재채점해 쌍대 bootstrap(B=1000) 차이 95% CI가 0을 포함하지 않음
- 메커니즘: 점수–log PA Spearman이 CAT(0.67)보다 낮음
- 절제(사후 아님, 사전 계획): (i) 무작위 음성만으로 학습, (ii) 행동 피처 제거, (iii) 임베딩 제거
- 채택: C-E6a·b·c 모두 충족 → "개선 방법"으로 논문 본체에 사용. 일부만 → 조건부로 보고

---

## E11 (리뷰 M5). 혼합 비율 ρ의 검증 기반 선택 — EviTrust-V (2026-10-22 실행 전 기록)

**배경**: EviTrust-H의 ρ = 3:1:1과 선택 규칙(a-floor)은 Epinions seed 42 테스트 결과를 본 뒤 정한 사후 설계다(ITERATION_03 §2). 이를 검증셋만으로 고르는 절차로 대체할 수 있는지 확인한다.

**절차 (결과 보기 전 고정)**
- 후보 ρ ∈ {1:0:0, 3:1:1, 1:1:1} (무작위 : 차수 일치 : 2-hop 음성, 양성당).
- 각 ρ 안의 설정(잎 수 ∈ {31,127}, 트리 수 100~2000)은 **검증 AP (a)(b)(c) 평균 최대**로 선택(`--select mean`, 사전 등록 E6 본안과 같은 규칙).
- ρ 선택: 데이터셋·seed마다 위 기준(검증 AP (a)(b)(c) 평균)이 가장 높은 ρ. 선택된 모델 = **EviTrust-V**.
- 테스트는 E6~E8과 같은 쌍. 3개 데이터셋 × seed 42·43·44.

**사전 고정 주장**
- C-E11a: EviTrust-V는 세 데이터셋 모두에서 (a) CAT 프로토콜 AP가 CAT보다 높다(쌍대 CI>0, 3/3 seed).
- C-E11b: Epinions·CiaoDVD에서 EviTrust-V의 (b)(c) AP가 모든 기준선(E8)보다 높다(쌍대 CI>0, 3/3 seed).
- C-E11c: EviTrust-V와 EviTrust-H의 (a)(b)(c) AP 차이는 데이터셋별로 |Δ| ≤ 0.02.
- 판정: 세 주장 모두 성립하면 원고의 EviTrust-H를 EviTrust-V로 교체(사후 설계 문제 해소). 일부만 성립하면 그대로 보고.
