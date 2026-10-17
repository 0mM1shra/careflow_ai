import streamlit as st
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from evaluation.run_eval import run_evaluation_pipeline, RESULTS_DIR
from evaluation.fixtures import TEST_FIXTURES
from backend.services.medgemma import medgemma_service

st.set_page_config(
    page_title="Evaluation & Benchmarks | CareFlow",
    page_icon="🧪",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .eval-header {
        background: linear-gradient(135deg, #3B0764 0%, #6B21A8 100%);
        color: white;
        padding: 20px 24px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: white;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="eval-header">
    <h2 style="margin:0; font-size: 24px;">🧪 CareFlow Clinical AI Benchmark Suite</h2>
    <div style="font-size: 14px; opacity: 0.9; margin-top: 4px;">
        Empirical Evaluation of MedGemma Extraction, Deterministic Triage Protocols & Failure Safety
    </div>
</div>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div style="background-color: #FEF3C7; border-left: 4px solid #D97706; padding: 10px 14px; border-radius: 6px; color: #92400E; font-size: 13px; margin-bottom: 20px;">
    <strong>Benchmarking Disclaimer:</strong> This trial measures unfine-tuned Google MedGemma performance on permitted test fixtures.
    This evaluation is for engineering and research validation only and <strong>does NOT constitute clinical validation</strong>.
</div>
""", unsafe_allow_html=True)

st.write(f"**Current Model Service Mode:** `{medgemma_service.mode.upper()}` | **Model ID:** `{medgemma_service.model_id}`")

col_run, col_info = st.columns([1, 2])
with col_run:
    if st.button("▶️ Run Benchmark Evaluation Suite", type="primary", use_container_width=True):
        with st.spinner("Executing benchmark across test fixtures..."):
            summary_res = run_evaluation_pipeline()
            st.session_state.last_eval = summary_res
            st.success("Benchmark completed successfully!")

with col_info:
    st.caption(
        "Runs all 7 standardized fixtures: Acute Coronary P0, Meningeal P1, Severe Pain P1, "
        "Contradictory Duration P2, Unknown Dimension P3, Routine Cold P3, and Simulated Model Failure ERR."
    )

st.write("---")

# Display Results if Available
eval_data = getattr(st.session_state, "last_eval", None)

# If no in-session eval, check if any saved runs exist in evaluation_results
if not eval_data:
    summary_files = list(RESULTS_DIR.glob("*_summary.json"))
    if summary_files:
        summary_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest_file = summary_files[0]
        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
                st.caption(f"Displaying latest saved run: `{latest_file.name}`")
        except Exception:
            pass

if eval_data:
    st.subheader(f"Benchmark Performance Summary (Run: {eval_data.get('run_id')})")

    # Metrics Row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Triage Accuracy", f"{eval_data.get('triage_accuracy_pct', 0)}%")
    m2.metric("Extraction Accuracy", f"{eval_data.get('extraction_accuracy_pct', 0)}%")
    m3.metric("Conflict Preservation", f"{eval_data.get('conflict_accuracy_pct', 0)}%")
    m4.metric("Unknown Preservation", f"{eval_data.get('unknown_accuracy_pct', 0)}%")
    m5.metric("Schema Error Rate", f"{eval_data.get('schema_error_rate_pct', 0)}%")
    m6.metric("Mean Latency", f"{eval_data.get('mean_latency_ms', 0)} ms")

    st.write("---")
    st.subheader("Detailed Fixture-by-Fixture Results")

    cases = eval_data.get("cases", [])
    if cases:
        table_rows = []
        for c in cases:
            match_icon = "✅" if c.get("priority_match") else "❌"
            table_rows.append({
                "Fixture ID": c.get("fixture_id"),
                "Scenario Title": c.get("title"),
                "Expected Priority": c.get("expected_priority"),
                "Assigned Priority": c.get("assigned_priority"),
                "Match": f"{match_icon} {c.get('priority_match')}",
                "Rules Triggered": ", ".join(c.get("triggered_rule_ids", [])) or "None",
                "Schema Valid": "✅ Yes" if c.get("schema_valid") else "❌ No",
                "Latency (ms)": c.get("latency_ms")
            })

        df = pd.DataFrame(table_rows)
        st.dataframe(df, use_container_width=True)

        # Download button for JSONL
        run_id = eval_data.get("run_id")
        jsonl_path = RESULTS_DIR / f"{run_id}.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path, "r", encoding="utf-8") as f:
                jsonl_bytes = f.read()
            st.download_button(
                label="📥 Download Benchmark Results (JSONL)",
                data=jsonl_bytes,
                file_name=f"{run_id}.jsonl",
                mime="application/jsonlines"
            )

        # Expandable inspection of individual case summaries
        with st.expander("🔍 Inspect Extracted Summaries & Evidence Details", expanded=False):
            for c in cases:
                st.markdown(f"### {c.get('fixture_id')} — {c.get('title')}")
                st.markdown(f"**Assigned Priority:** `{c.get('assigned_priority')}` | **Expected:** `{c.get('expected_priority')}`")
                st.markdown(f"**Reasons:** {'; '.join(c.get('reasons', []))}")
                st.markdown(f"**Extracted Symptoms:** {', '.join(c.get('symptoms_extracted', [])) or 'None'}")
                st.markdown(f"**Summary Text:** {c.get('summary_text')}")
                st.write("---")

else:
    st.info("No benchmark run recorded yet. Click '▶️ Run Benchmark Evaluation Suite' above to execute tests.")

# Display Test Set Documentation
with st.expander("📚 Permitted Benchmark Scenarios Specification", expanded=False):
    for f in TEST_FIXTURES:
        st.markdown(f"**{f['id']}**: {f['title']}")
        st.caption(f"{f['description']} (Expected: {f['expected_priority']})")
        st.write(f"- Utterances: `{f['messages']}`")
