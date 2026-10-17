import streamlit as st
import sys
from pathlib import Path

# Add project root to sys.path so frontend can import backend / config if needed
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from frontend.api_client import api_client

st.set_page_config(
    page_title="CareFlow | MedGemma Clinical Intake & Reviewer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Clean Clinical Dashboard CSS
st.markdown("""
<style>
    /* Clean typography & spacing */
    .main {
        background-color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .header-box {
        background: linear-gradient(135deg, #1E3A8A 0%, #0284C7 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .header-title {
        font-size: 28px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        font-size: 15px;
        opacity: 0.9;
        margin-top: 6px;
    }
    .disclaimer-banner {
        background-color: #FEF3C7;
        border-left: 4px solid #D97706;
        padding: 12px 16px;
        border-radius: 6px;
        color: #92400E;
        font-size: 13.5px;
        margin-bottom: 20px;
    }
    .card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .card-title {
        font-size: 18px;
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 8px;
    }
    .card-desc {
        font-size: 14px;
        color: #64748B;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
<div class="header-box">
    <div class="header-title">🩺 CareFlow: MedGemma Clinical Intake & Reviewer</div>
    <div class="header-subtitle">Evidence-Grounded Conversational Intake, Deterministic Triage Protocols & Decision Support Dashboard</div>
</div>
""", unsafe_allow_html=True)

# Mandatory Prototype Disclaimer
st.markdown("""
<div class="disclaimer-banner">
    <strong>⚠️ Research & Hackathon Prototype Notice:</strong>
    CareFlow is an exploratory conversational assistant and reviewer dashboard powered by Google MedGemma.
    It is <strong>NOT a diagnostic tool, medical device, or clinically validated emergency triage system</strong>.
    All outputs require qualified professional human review. In a life-threatening emergency, call 911 or visit the nearest emergency department.
</div>
""", unsafe_allow_html=True)

# Sidebar System Health Status
health = api_client.get_health()
st.sidebar.title("⚙️ System Status")

if health.get("status") == "healthy":
    st.sidebar.success("🟢 Backend API: Online")
    st.sidebar.info(f"🧠 AI Service Mode: **{health.get('medgemma_mode', 'mock').upper()}**")
    st.sidebar.caption(f"Protocol: {health.get('triage_rule_version', 'v1.0.0-prototype')}")
    st.sidebar.caption(f"Database: {health.get('database', 'SQLite')}")
else:
    st.sidebar.error("🔴 Backend API: Offline or Unreachable")
    st.sidebar.caption("Ensure FastAPI is running: `uvicorn backend.main:app --port 8000`")

st.sidebar.divider()
st.sidebar.subheader("Navigation")
st.sidebar.page_link("pages/patient.py", label="Patient Interface", icon="🩺")
st.sidebar.page_link("pages/reviewer.py", label="Reviewer Dashboard", icon="📋")
st.sidebar.page_link("pages/evaluation.py", label="Evaluation & Benchmarks", icon="🧪")

st.sidebar.divider()
st.sidebar.caption("CareFlow • MedGemma 4B Instruction-Tuned Prototype")

# Landing Page Content
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-title">🩺 Patient Conversational Intake</div>
        <div class="card-desc">
            A respectful, privacy-preserving conversational interface for patients to report symptoms in plain text.
            <ul>
                <li><strong>No personal identifying information (PII) required</strong>: Random Case ID generated.</li>
                <li><strong>Adaptive follow-up</strong>: Google MedGemma identifies missing dimensions and asks one clarifying question at a time.</li>
                <li><strong>Respect for unknown answers</strong>: Skipping or 'I don't know' is preserved as unknown, never assumed absent.</li>
                <li><strong>Real-time Red Flag Detection</strong>: Emergency warnings appear immediately if critical symptoms are mentioned.</li>
                <li><strong>Review & Correction</strong>: Patients can review and edit their recorded statements before final submission.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Launch Patient Intake →", type="primary", use_container_width=True):
        st.switch_page("pages/patient.py")

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-title">📋 Healthcare Reviewer Dashboard</div>
        <div class="card-desc">
            A high-acuity priority queue and clinical decision support dashboard for triage nurses and physicians.
            <ul>
                <li><strong>Deterministic Protocol Triage</strong>: Separation of generative AI from emergency priority rules (P0, P1, P2, P3).</li>
                <li><strong>AI-Generated Clinical Summary</strong>: Structured synthesis with verbatim evidence citations to patient quotes.</li>
                <li><strong>Uncertainty & Conflict Surfacing</strong>: Clashing reports are flagged for targeted human re-interview.</li>
                <li><strong>Clinician Override & Audit Log</strong>: Reviewers can edit summaries, override priority with mandatory rationale, and record approvals.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Reviewer Dashboard →", use_container_width=True):
        st.switch_page("pages/reviewer.py")

st.divider()

col_eval, col_info = st.columns([1, 1])

with col_eval:
    st.markdown("""
    <div class="card">
        <div class="card-title">🧪 Model Evaluation & Benchmarks</div>
        <div class="card-desc">
            Run reproducible test fixtures to evaluate extraction accuracy, triage protocol consistency,
            conflict preservation, and system fallback safety across 7 standardized scenarios.
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Run Clinical Benchmark Suite →", use_container_width=True):
        st.switch_page("pages/evaluation.py")

with col_info:
    st.markdown("""
    <div class="card">
        <div class="card-title">🛡️ System Architecture & Guardrails</div>
        <div class="card-desc">
            <strong>Generative AI Model:</strong> Google MedGemma 4B (Instruction-Tuned).<br/>
            <strong>Triage Rules:</strong> Deterministic protocol (Manchester Triage System & ESI adapted).<br/>
            <strong>Prompt Injection Defense:</strong> Patient text treated strictly as untrusted clinical input inside XML tags.<br/>
            <strong>Fail-Safe State:</strong> If AI or model inference fails, case enters visible <code>ERR: Review required — AI unavailable</code> state.
        </div>
    </div>
    """, unsafe_allow_html=True)
