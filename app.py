"""OpportunityLens: Streamlit prototype. Run with:  streamlit run app.py"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from opportunitylens.checks import what_if
from opportunitylens.config import (FEATURE_LABELS, NOT_PROOF_NOTICE, NUMERIC, SYNTHETIC_NOTICE, VALID_RANGES,
                                    CATEGORY_VALUES, LABEL_INVESTIGATE, LABEL_MODEL, LABEL_WARNING)
from opportunitylens.pipeline import build_system
from opportunitylens.report import build_receipt, html_to_pdf, render_html
from opportunitylens.validation import run_validation, to_markdown

st.set_page_config(page_title="OpportunityLens", page_icon="🔎", layout="wide")
ICON = {LABEL_MODEL: "🔵", LABEL_WARNING: "🟡", LABEL_INVESTIGATE: "🔴"}
tag = lambda label: f"{ICON[label]} *{label}*"


@st.cache_resource(show_spinner="Generating synthetic data and training models...")
def load_system():
    return build_system()


@st.cache_data(show_spinner=False)
def validation_markdown():
    return to_markdown(run_validation(load_system()))


S = load_system()
test = S.bundle.test_df

# ----------------------------------------------------------------------------- sidebar: choose applicant
st.sidebar.title("🔎 OpportunityLens")
st.sidebar.warning(SYNTHETIC_NOTICE)
from opportunitylens.checks import input_checks  # noqa: E402


@st.cache_data(show_spinner=False)
def flagged_ids():
    return [r.applicant_id for r in S.bundle.test_df.itertuples() if input_checks(r._asdict())]


only_flagged = st.sidebar.checkbox("Only applicants with input flags", value=False)
ids = flagged_ids() if only_flagged else list(test["applicant_id"])
sel = st.sidebar.selectbox("Synthetic applicant", ids)
base = S.df.loc[S.df.applicant_id == sel].iloc[0].to_dict()
row = dict(base)

if st.sidebar.toggle("Edit this applicant's values"):
    st.sidebar.caption("Out-of-range values are allowed here so you can see how the input checks react.")
    for f in NUMERIC:
        lo, hi = VALID_RANGES[f]
        unknown = st.sidebar.checkbox(f"{FEATURE_LABELS[f]}: unknown", value=pd.isna(base[f]), key=f"unk_{sel}_{f}")
        default = float(base[f]) if not pd.isna(base[f]) else float((lo + hi) / 2)
        val = st.sidebar.number_input(FEATURE_LABELS[f], value=default, step=(hi - lo) / 100, key=f"num_{sel}_{f}",
                                      disabled=unknown)
        row[f] = np.nan if unknown else val
    row["category"] = st.sidebar.selectbox("Applicant category", CATEGORY_VALUES,
                                           index=CATEGORY_VALUES.index(base["category"]), key=f"cat_{sel}")

# ----------------------------------------------------------------------------- state-driven receipt inputs
note = st.session_state.get(f"note_{sel}", "")
extra = []
wf = st.session_state.get(f"wi_field_{sel}", NUMERIC[0])
wv = st.session_state.get(f"wi_value_{sel}_{wf}")
if wv is not None and not (not pd.isna(row.get(wf)) and abs(float(row[wf]) - wv) < 1e-9):
    extra.append(what_if(S.bundle, row, wf, float(wv)))
rc = build_receipt(S, row, note, extra)
html_doc = render_html(rc)

st.sidebar.divider()
st.sidebar.download_button("⬇️ Download Decision Receipt (HTML)", html_doc, f"decision_receipt_{sel}.html", "text/html")
pdf = html_to_pdf(html_doc)
if pdf:
    st.sidebar.download_button("⬇️ Download as PDF", pdf, f"decision_receipt_{sel}.pdf", "application/pdf")
else:
    st.sidebar.caption("PDF: open the HTML receipt in a browser and print to PDF (or `pip install weasyprint`).")

# ----------------------------------------------------------------------------- main
st.title("Decision Receipt")
st.caption(f"Applicant **{sel}**  ·  {NOT_PROOF_NOTICE}")
t1, t2, t3, t4, t5, t6 = st.tabs(["1 What happened", "2 Why", "3 Consistent?", "4 Challenge", "5 Limits",
                                  "Model & validation"])

o = rc["outcome"]
with t1:
    st.markdown(tag(LABEL_MODEL))
    c = st.columns(4)
    c[0].metric("Simulated outcome", o["decision"])
    c[1].metric("Prediction score", f"{o['score']:.2f}")
    c[2].metric("Decision threshold", f"{o['threshold']:.2f}")
    c[3].metric("Margin", f"{o['margin']:+.2f}")
    st.caption(f"Model: {o['model']}")
    st.subheader("Inputs used")
    st.dataframe(pd.DataFrame({"Field": [FEATURE_LABELS[k] for k in rc["inputs"]], "Value recorded": list(rc["inputs"].values())}),
                 hide_index=True)

with t2:
    st.markdown(tag(LABEL_MODEL) + " — what the model did, **not** what caused the outcome in real life.")
    ex = rc["why"]["table"].sort_values("contribution")
    fig = go.Figure(go.Bar(x=ex["contribution"], y=ex["label"], orientation="h",
                           marker_color=["#c2572b" if v < 0 else "#2a7f62" for v in ex["contribution"]]))
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Effect on approval score (log-odds)")
    st.plotly_chart(fig)
    for s in rc["why"]["table"]["sentence"].head(4):
        st.markdown(f"- {s}")
    st.caption(f"Explanation method: {rc['why']['method']}")
    with st.expander("Global view: what drives the model overall"):
        g = S.explainer.global_importance(S.bundle.test_df)
        st.bar_chart(g.set_index("label")["share"])

with t3:
    sc = rc["similar"]
    st.subheader("Similar cases")
    st.markdown(f"{tag(sc['label'])} — {sc['message']}")
    nb = sc["neighbors"].head(8).copy()
    nb["decision"] = nb["decision"].map({True: "Approved", False: "Not approved"})
    st.dataframe(nb.drop(columns=["approved"]).round(2), hide_index=True)
    st.subheader("Group-level differences (whole system, held-out test set)")
    gr = rc["group"]
    st.markdown(f"{tag(gr['label'])} — {gr['summary']}")
    st.dataframe(gr["table"].round(3))
    fig = go.Figure()
    for k, name in [("historical_approval_rate", "Historical"), ("model_selection_rate", "Model")]:
        fig.add_bar(name=name, x=list(gr["table"].index), y=gr["table"][k])
    fig.update_layout(barmode="group", height=280, yaxis_tickformat=".0%", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig)
    for f in gr["findings"]:
        st.markdown(f"- {f['message']}")

with t4:
    st.subheader("Input checks")
    if not rc["flags"]:
        st.success("No missing or implausible fields detected. (Plausible-looking wrong values cannot be detected.)")
    for f in rc["flags"]:
        (st.error if f.severity == "high" else st.warning)(f"{f.message}  \n{tag(f.label)}")
    st.subheader("What-if")
    st.caption("Change one field and see how the model's score moves. This shows sensitivity, not what the applicant would be awarded.")
    field = st.selectbox("Field", NUMERIC, format_func=FEATURE_LABELS.get, key=f"wi_field_{sel}")
    lo, hi = VALID_RANGES[field]
    cur = row.get(field)
    st.number_input("New value", min_value=float(lo), max_value=float(hi),
                    value=float(cur) if not pd.isna(cur) and lo <= cur <= hi else float((lo + hi) / 2),
                    step=(hi - lo) / 100, key=f"wi_value_{sel}_{field}")
    for w in rc["what_ifs"]:
        st.info(f"{w['message']}  \n{tag(w['output_label'])}")
    st.subheader("Request a human review")
    st.text_area("Why do you want this outcome reviewed? (included in the receipt)", key=f"note_{sel}")
    st.caption("Download the receipt in the sidebar and send it to the scholarship office with your applicant ID.")
    st.subheader("Questions for a human reviewer")
    for q in rc["reviewer_questions"]:
        st.markdown(f"1. {q}")

with t5:
    for item in rc["limits"]:
        st.markdown(f"- {item}")

with t6:
    st.subheader("Model comparison (held-out test set)")
    b = S.bundle
    st.dataframe(pd.DataFrame({k: {m: v for m, v in d.items() if m != "_curve"} for k, d in b.metrics.items()}).T.round(3),
                 )
    st.caption(" ".join(b.notes) + f" Majority-class accuracy: {b.baseline_accuracy:.3f}.")
    curve = pd.DataFrame(b.calibration_curve, columns=["predicted", "observed", "n"])
    fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(dash="dot")),
                     go.Scatter(x=curve["predicted"], y=curve["observed"], mode="lines+markers", name="Model")])
    fig.update_layout(height=300, xaxis_title="Predicted probability", yaxis_title="Observed approval rate",
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig)
    st.subheader("Validation suite")
    if st.button("Run validation (detection, explanation and end-to-end tests)"):
        with st.spinner("Running..."):
            st.markdown(validation_markdown())
