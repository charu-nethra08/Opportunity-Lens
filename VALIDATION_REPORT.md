# OpportunityLens validation report

> Controlled synthetic data only. Passing these tests does **not** show real-world fairness, causation or discrimination.

## 1. Model quality: PASS
- Chosen model: Logistic regression (baseline)
- Held-out test: accuracy 0.857, ROC-AUC 0.929, Brier 0.104, calibration error (ECE) 0.028
- Baselines: logistic-regression ROC-AUC 0.929; majority-class accuracy 0.512
- Cross-validated ROC-AUC: Logistic regression (baseline) 0.920±0.006; Gradient-boosted trees 0.918±0.008

## 2. Detection test: PASS
| Injected issue | Injected | Detected | Recall |
|---|---|---|---|
| missing | 136 | 136 | 100% |
| out_of_range | 100 | 100 | 100% |
| scale_error | 56 | 56 | 100% |
| subtle_shift | 70 | 0 | 0% |
- False-alarm rate on clean rows: 0.0%
- Group disparity flagged when injected: True
- Group disparity flagged on no-bias control: False
- **What it misses:** subtle_shift errors (plausible but wrong values) are invisible to range checks by design; group metrics flag gaps but cannot say whether they come from the data, the labels or the model

## 3. Explanation check: PASS
- Method: Linear contributions (coefficient x deviation from average; equals SHAP for linear models)
- Rank correlation (true effect vs explained importance): 1.00
- True top-3 drivers ['academic_score', 'documentation_completeness', 'financial_need'] vs model top-3 ['academic_score', 'documentation_completeness', 'financial_need']
- Importance on features with no true effect: 3.9%

## 4. End-to-end test: PASS
- all five sections present: PASS
- outcome stated: PASS
- score shown: PASS
- limitations stated: PASS
- user text escaped: PASS
- synthetic notice: PASS
