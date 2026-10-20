# OpportunityLens: Decision Receipts for AI-assisted scholarship decisions

A prototype that lets an *affected person* understand, question and ask for review of one AI-assisted
scholarship decision. Its core output is the **Decision Receipt**, a plain-language record that answers:

1. **What happened?** Simulated outcome, score, threshold.
2. **Why?** Strongest factors (SHAP), in words and as a chart.
3. **Was the process consistent?** Similar synthetic cases + group-level outcome/error gaps.
4. **What can be challenged?** Input checks, what-if tests, a route to human review.
5. **What are the limits?** What the model can and cannot establish; what a reviewer should inspect.

> **Synthetic data only.** No real decisions are made. A receipt never proves discrimination or real-world fairness.

## Deploy on Streamlit Community Cloud
Put the **contents** of this folder at the root of your GitHub repo (so `app.py` and `requirements.txt` sit at the top level, next to the `opportunitylens/` package folder), then create the app with main file `app.py`. Do not pin old versions in `requirements.txt`: pinned old numpy/scikit-learn have no prebuilt wheels on new Python versions and the build can hang for many minutes. If `shap` is slow to install, remove that line; the app falls back to a built-in explanation method and says so in the receipt.

## Setup (local)
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py          # interactive prototype
pytest -q                     # automated tests
python -m opportunitylens.validation   # writes VALIDATION_REPORT.md
```
First launch trains the models (a few seconds). Python 3.10+.

## Using the app
Pick a synthetic applicant (tick *Only applicants with input flags* to see data-quality cases), read the tabs 1 to 5,
optionally edit values or run a what-if, add a note requesting review, then download the receipt (HTML; PDF if WeasyPrint is installed).

## Layout
| Module | Role |
|---|---|
| `config.py` | features, valid ranges, output labels, standard notices |
| `data.py` | synthetic data with injected missing/incorrect fields and a simulated group disparity |
| `model.py` | logistic-regression baseline vs gradient-boosted trees, chosen by cross-validation; test metrics + calibration |
| `explain.py` | local/global SHAP explanations translated to plain language (transparent fallback if SHAP is absent) |
| `checks.py` | input checks, similar cases, what-if, group metrics with bootstrap CIs |
| `report.py` | builds the receipt and renders self-contained HTML (+ optional PDF) |
| `pipeline.py` | wires everything into one `System` |
| `validation.py` | model quality, detection test, explanation check, end-to-end test |
| `app.py` | Streamlit UI |

Every output is labelled **Model explanation**, **Warning signal** or **Needs human investigation**.

## Design decisions worth knowing
- **The sensitive attribute is never a model input.** `group` is used only for group-level checks. `institution_tier` and `category` are
  correlated with group but have *no* true effect, so we can test whether the model leans on proxies.
- **Simpler model wins ties.** Gradient boosting is only chosen if it beats the baseline by > 0.005 CV ROC-AUC (here the data is
  near-linear, so logistic regression is chosen). SHAP's `LinearExplainer`/`TreeExplainer` is picked to match.
- **Group warnings need a CI that excludes zero** (and a ≥10-point gap); smaller-sample gaps are shown as "may be noise".
- **Receipt is model-agnostic:** swap any scikit-learn pipeline into `ModelBundle` and it works if SHAP can explain it.

## Known limitations
- Subtle wrong-but-plausible inputs (e.g. need recorded 3 points off) are *not* detectable by range checks. The validation report shows 0% recall on these by design.
- Group gaps are computed against historical labels that are themselves biased by construction; they are warning signals only.
- Reference population and thresholds are synthetic. Do not use on real applicants without consented data, audit logs and an expert-led fairness audit.
- Without SHAP installed, the tree-model path uses a clearly-labelled occlusion approximation; the receipt states which method was used.
