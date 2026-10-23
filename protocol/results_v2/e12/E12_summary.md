# E12 (N1) — 학습하지 않은 난이도 음성에서의 일반화

(*) = 어떤 모델도 학습·선택에 쓰지 않은 음성. observed 부분집합 AP, seed 3개 평균.

## Epinions  (V가 고른 ρ: full_m311mean, full_m311mean, full)

| 방법 | (a) | (b) | (c) | (e) co-review* | (f) active-dm* |
|---|---|---|---|---|---|
| CAT | 0.9377 | 0.5790 | 0.4393 | 0.5954 | 0.5143 |
| Heuristic-LR | 0.9158 | 0.5622 | 0.4199 | 0.6345 | 0.5169 |
| Guardian | 0.8713 | 0.4896 | 0.4065 | 0.5824 | 0.5038 |
| TrustGuard | 0.9254 | 0.5611 | 0.4533 | 0.5995 | 0.4887 |
| TGN | 0.8541 | 0.7402 | 0.6162 | 0.7724 | 0.5588 |
| EviTrust-U | 0.9705 | 0.7147 | 0.5410 | 0.6742 | 0.5665 |
| EviTrust-V | 0.9384 | 0.7832 | 0.6620 | 0.6810 | 0.6006 |
| EviTrust-V − (c) 생성기 제외 | 0.9583 | 0.7841 | 0.6100 | 0.7005 | 0.6080 |
| EviTrust-V − (b) 생성기 제외 | 0.9523 | 0.7625 | 0.6646 | 0.6716 | 0.5830 |

최강 기준선: (a) CAT 0.9377, (b) TGN 0.7402, (c) TGN 0.6162, (e) co-review* TGN 0.7724, (f) active-dm* TGN 0.5588

| 비교 (observed, 쌍대 bootstrap) | 음성 | 차이 | CI>0 / CI<0 seed 수 |
|---|---|---|---|
| V-c − TGN | held-out (c) | -0.0062 | 1/2 |
| V-b − TGN | held-out (b) | +0.0223 | 3/0 |
| V − TGN | unseen (e) | -0.0914 | 0/3 |
| V − TGN | unseen (f) | +0.0418 | 3/0 |
| U − TGN | unseen (e) | -0.0982 | 0/3 |
| U − TGN | unseen (f) | +0.0077 | 2/0 |

생성기 제외의 비용: (c)에서 V−c 0.6100 vs V 0.6620 (Δ -0.0520); (b)에서 V−b 0.7625 vs V 0.7832 (Δ -0.0207)

## CiaoDVD  (V가 고른 ρ: full_m311mean, full, full)

| 방법 | (a) | (b) | (c) | (e) co-review* | (f) active-dm* |
|---|---|---|---|---|---|
| CAT | 0.9440 | 0.5427 | 0.5375 | 0.6078 | 0.5403 |
| Heuristic-LR | 0.9829 | 0.7455 | 0.7369 | 0.7612 | 0.7402 |
| Guardian | 0.9627 | 0.5431 | 0.5091 | 0.5673 | 0.5417 |
| TrustGuard | 0.9554 | 0.5306 | 0.5029 | 0.5923 | 0.5272 |
| TGN | 0.9269 | 0.6676 | 0.5654 | 0.7008 | 0.6653 |
| EviTrust-U | 0.9866 | 0.7873 | 0.7405 | 0.7787 | 0.7817 |
| EviTrust-V | 0.9712 | 0.8478 | 0.8363 | 0.7729 | 0.8418 |
| EviTrust-V − (c) 생성기 제외 | 0.9826 | 0.8560 | 0.7587 | 0.8033 | 0.8500 |
| EviTrust-V − (b) 생성기 제외 | 0.9702 | 0.7835 | 0.8458 | 0.7437 | 0.7811 |

최강 기준선: (a) Heuristic-LR 0.9829, (b) Heuristic-LR 0.7455, (c) Heuristic-LR 0.7369, (e) co-review* Heuristic-LR 0.7612, (f) active-dm* Heuristic-LR 0.7402

| 비교 (observed, 쌍대 bootstrap) | 음성 | 차이 | CI>0 / CI<0 seed 수 |
|---|---|---|---|
| V-c − Heuristic-LR | held-out (c) | +0.0219 | 3/0 |
| V-b − Heuristic-LR | held-out (b) | +0.0380 | 3/0 |
| V − Heuristic-LR | unseen (e) | +0.0117 | 3/0 |
| V − Heuristic-LR | unseen (f) | +0.1016 | 3/0 |
| U − Heuristic-LR | unseen (e) | +0.0174 | 3/0 |
| U − Heuristic-LR | unseen (f) | +0.0415 | 3/0 |

생성기 제외의 비용: (c)에서 V−c 0.7587 vs V 0.8363 (Δ -0.0776); (b)에서 V−b 0.7835 vs V 0.8478 (Δ -0.0644)

## wiki-RfA  (V가 고른 ρ: full, full_m311mean, full)

| 방법 | (a) | (b) | (c) | (d) | (e) co-review* | (f) active-dm* |
|---|---|---|---|---|---|---|
| CAT | 0.4201 | 0.4131 | 0.1360 | 0.7505 | 0.1462 | 0.3592 |
| Heuristic-LR | 0.4142 | 0.3831 | 0.1396 | 0.7778 | 0.1365 | 0.3436 |
| Guardian | 0.3223 | 0.3710 | 0.1420 | 0.7671 | 0.1708 | 0.4065 |
| TrustGuard | 0.5162 | 0.5229 | 0.2345 | 0.7440 | 0.2700 | 0.4956 |
| TGN | 0.2906 | 0.4952 | 0.2592 | 0.7938 | 0.2676 | 0.4441 |
| EviTrust-U | 0.7281 | 0.4567 | 0.2194 | 0.7789 | 0.1989 | 0.3507 |
| EviTrust-V | 0.5252 | 0.4435 | 0.3515 | 0.7653 | 0.2888 | 0.3908 |
| EviTrust-V − (c) 생성기 제외 | 0.7168 | 0.5141 | 0.2400 | 0.7669 | 0.2138 | 0.3547 |
| EviTrust-V − (b) 생성기 제외 | 0.7487 | 0.3401 | 0.4304 | 0.7655 | 0.3459 | 0.3730 |

최강 기준선: (a) TrustGuard 0.5162, (b) TrustGuard 0.5229, (c) TGN 0.2592, (d) TGN 0.7938, (e) co-review* TrustGuard 0.2700, (f) active-dm* TrustGuard 0.4956

| 비교 (observed, 쌍대 bootstrap) | 음성 | 차이 | CI>0 / CI<0 seed 수 |
|---|---|---|---|
| V-c − TGN | held-out (c) | -0.0192 | 0/2 |
| V-b − TrustGuard | held-out (b) | -0.1827 | 0/3 |
| V − TrustGuard | unseen (e) | +0.0189 | 2/0 |
| V − TrustGuard | unseen (f) | -0.1048 | 0/3 |
| U − TrustGuard | unseen (e) | -0.0710 | 0/3 |
| U − TrustGuard | unseen (f) | -0.1448 | 0/3 |

생성기 제외의 비용: (c)에서 V−c 0.2400 vs V 0.3515 (Δ -0.1115); (b)에서 V−b 0.3401 vs V 0.4435 (Δ -0.1034)

