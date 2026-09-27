# ============================================================
# AERIS Phase 1 — Streamlit Demo Application
#
# Tab 1: Full AERIS Pipeline Demo
#   Loads the precomputed Phase 1 Explainable Investigation Output
#   (CMAPSS + NTSB + NGAFID fused via Dempster-Shafer) and displays
#   it as a clean, presentable dashboard.
#
# Tab 2: NTSB Semantic Retrieval Explorer
#   A live sandbox where any free-text failure description can be
#   typed in, embedded with the same Sentence Transformer used in
#   the pipeline, and searched against the real NTSB FAISS index.
#   This demonstrates that the retrieval component generalizes
#   beyond the single CMAPSS FD001 fault mode (High-Pressure
#   Compressor), even though the full end-to-end pipeline result
#   in Tab 1 is currently tied to that one fault type.
#
# Run with:  streamlit run app.py
# ============================================================

import json
import os
import glob

import streamlit as st
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="AERIS — Phase 1 Demo",
    page_icon="🛩️",
    layout="wide",
)

# ------------------------------------------------------------
# PATHS — update these if your folder layout differs
# ------------------------------------------------------------
REPORT_JSON_PATH = "Reports/Phase1_Explainable_Investigation_Output.json"

NTSB_METADATA_PATH = "NTSB_metadata.csv"
NTSB_FAISS_INDEX_PATH = "NTSB_FAISS.index"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ------------------------------------------------------------
# SHARED STYLES
# ------------------------------------------------------------
st.markdown("""
<style>
    .risk-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 15px;
    }
    .card {
        background: white;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        margin-bottom: 14px;
    }
    .small-label {
        font-size: 12px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

RISK_COLORS = {
    "Low Risk": ("#1e7e34", "#d4edda"),
    "Medium Risk": ("#b8860b", "#fff3cd"),
    "High Risk": ("#a71d2a", "#f8d7da"),
}

st.title("🛩️ AERIS — Phase 1 Demo")
st.caption("Explainable Reasoning and Investigation System for Aerospace Predictive Maintenance")

tab1, tab2 = st.tabs(["📊 Full AERIS Pipeline Demo", "🔎 NTSB Semantic Retrieval Explorer"])

# ============================================================
# TAB 1 — FULL PIPELINE DEMO (precomputed, real end-to-end case)
# ============================================================
with tab1:
    st.markdown(
        "This tab shows AERIS's complete, real, end-to-end result for one "
        "case: a CMAPSS engine's RUL prediction, the NTSB report retrieved "
        "based on it, the independent NGAFID flight evidence, and the final "
        "risk assessment produced by Dempster–Shafer fusion."
    )

    if not os.path.exists(REPORT_JSON_PATH):
        st.error(
            f"Could not find '{REPORT_JSON_PATH}'. Make sure you run this app "
            f"from your AERIS project root, or update REPORT_JSON_PATH at the "
            f"top of app.py."
        )
    else:
        with open(REPORT_JSON_PATH, "r") as f:
            report = json.load(f)

        risk_level = report["aircraft_risk_level"]
        confidence = report["overall_confidence_belief"]
        subsystem = report["most_probable_fault_subsystem"]
        cmapss = report["supporting_sensor_evidence_cmapss"]
        ntsb = report["supporting_historical_investigation_ntsb"]
        ngafid = report["supporting_flight_evidence_ngafid"]
        fusion = report["fusion_transparency"]

        risk_dark, risk_light = RISK_COLORS.get(risk_level, ("#333", "#eee"))

        # ---- Risk banner ----
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(
                f"""<div class="card" style="background:{risk_light}; border-left:6px solid {risk_dark};">
                <div class="small-label">Aircraft Risk Level</div>
                <div style="font-size:32px; font-weight:800; color:{risk_dark};">{risk_level}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""<div class="card">
                <div class="small-label">Overall Confidence</div>
                <div class="metric-value">{confidence['pignistic_probability']*100:.1f}%</div>
                <div style="font-size:12px; color:#777;">Belief {confidence['belief_lower_bound']*100:.0f}%
                &ndash; Plausibility {confidence['plausibility_upper_bound']*100:.0f}%</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""<div class="card">
                <div class="small-label">Most Probable Fault Subsystem</div>
                <div class="metric-value" style="font-size:18px;">{subsystem['subsystem']}</div>
                <div style="font-size:12px; color:#777;">Primary source: {subsystem['primary_evidence_source']}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        max_conflict = max(fusion["inter_source_conflict"].values())
        if max_conflict > 0.3:
            st.warning(
                f"⚠️ Inter-source conflict of {max_conflict:.2f} was detected during fusion. "
                f"This indicates the evidence sources describe **different underlying aircraft "
                f"cases** in this Phase 1 prototype (a known limitation of combining independent "
                f"benchmark datasets), rather than a fusion error. See the technical appendix for "
                f"the full Dempster-Shafer breakdown."
            )

        st.markdown("### Supporting Evidence by Source")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("#### 🔧 CMAPSS — Sensor Evidence")
            st.metric("Predicted RUL", f"{cmapss['predicted_rul_cycles']} cycles")
            st.metric("Prediction Confidence", f"{cmapss['prediction_confidence']*100:.1f}%")
            st.write(f"**Inferred subsystem:** {cmapss['inferred_subsystem']}")
            st.write(f"**Top SHAP features:** {', '.join(cmapss['top_contributing_sensors_shap'])}")

        with c2:
            st.markdown("#### 📄 NTSB — Historical Investigation")
            st.write(f"**Matched report:** {ntsb['matched_report_id']}")
            st.write(f"**Aircraft:** {ntsb['aircraft_involved']}")
            st.metric("Semantic Similarity", f"{ntsb['semantic_similarity']*100:.1f}%")
            with st.expander("Probable cause (full text)"):
                st.write(ntsb["probable_cause"])

        with c3:
            st.markdown("#### ✈️ NGAFID — Flight Evidence")
            st.metric("Failure Risk Probability", f"{ngafid['failure_risk_probability']*100:.1f}%")
            st.write(f"**Diagnosed subsystem:** {ngafid['diagnosed_subsystem']}")
            st.metric("Diagnosis Confidence", f"{ngafid['diagnosis_confidence']*100:.1f}%")
            st.write(f"**Top anomaly sensors:** {', '.join(ngafid['top_anomaly_sensors'])}")

        with st.expander("🔬 Technical Appendix — Full Dempster-Shafer Fusion Detail"):
            st.json(fusion)
            st.json(report)

# ============================================================
# TAB 2 — LIVE NTSB SEMANTIC RETRIEVAL EXPLORER
# ============================================================
with tab2:
    st.markdown(
        "Type **any** aircraft failure description below — it does not need "
        "to be related to compressor issues. This sandbox tests the NTSB "
        "semantic retrieval module (Sentence Transformer + FAISS) "
        "independently of the fixed CMAPSS scenario shown in Tab 1, "
        "demonstrating that retrieval generalizes across failure types "
        "even though CMAPSS FD001 itself only simulates High-Pressure "
        "Compressor degradation."
    )

    missing = []
    if not os.path.exists(NTSB_METADATA_PATH):
        missing.append(NTSB_METADATA_PATH)
    if not os.path.exists(NTSB_FAISS_INDEX_PATH):
        missing.append(NTSB_FAISS_INDEX_PATH)

    if missing:
        st.error(
            f"Missing required file(s): {', '.join(missing)}. "
            f"Update the paths at the top of app.py, or run this app from "
            f"your AERIS project root where these files live."
        )
    else:
        @st.cache_resource
        def load_retrieval_components():
            import faiss
            from sentence_transformers import SentenceTransformer
            metadata = pd.read_csv(NTSB_METADATA_PATH)
            index = faiss.read_index(NTSB_FAISS_INDEX_PATH)
            model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            return metadata, index, model

        with st.spinner("Loading NTSB FAISS index and embedding model..."):
            metadata, index, model = load_retrieval_components()

        example_queries = [
            "landing gear failure on approach",
            "fuel starvation during cruise",
            "electrical system fire in cockpit",
            "propeller blade separation",
            "hydraulic system failure",
        ]

        query = st.text_input(
            "Enter a failure description:",
            placeholder="e.g. 'engine fire during takeoff' or 'bird strike causing engine damage'",
        )

        st.caption("Or try an example: " + " · ".join(f"`{q}`" for q in example_queries))

        top_k = st.slider("Number of results to retrieve", min_value=1, max_value=10, value=5)

        if query:
            query_embedding = model.encode(
                query, convert_to_numpy=True, normalize_embeddings=True
            )
            query_embedding = np.expand_dims(query_embedding, axis=0)

            scores, indices = index.search(query_embedding, top_k)

            st.markdown(f"### Top {top_k} Retrieved NTSB Cases")

            for rank, idx in enumerate(indices[0]):
                record = metadata.iloc[idx]
                similarity = float(scores[0][rank])

                with st.container():
                    st.markdown(
                        f"""<div class="card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                          <div><strong>Rank {rank+1}</strong> &mdash; {record.get('event_id', 'N/A')}
                          &nbsp;|&nbsp; {record.get('aircraft_make', '')} {record.get('aircraft_model', '')}</div>
                          <div class="risk-badge" style="background:#e8f0fe; color:#1a4b8c;">
                            Similarity: {similarity:.4f}
                          </div>
                        </div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                    with st.expander(f"View details for {record.get('event_id', 'N/A')}"):
                        st.write(f"**Probable Cause:** {record.get('probable_cause', 'N/A')}")
                        st.write(f"**Findings:** {record.get('findings', 'N/A')}")
        else:
            st.info("Enter a query above to search the NTSB investigation database.")

st.divider()
st.caption(
    "AERIS Phase 1 Prototype — Sources: NASA CMAPSS, NGAFID, NTSB Aviation Investigation Reports. "
    "Fusion via Dempster–Shafer evidence theory."
)
