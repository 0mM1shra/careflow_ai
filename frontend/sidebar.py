import os
import streamlit as st
import requests
from frontend.api_client import api_client

def render_careflow_sidebar(current_page: str = ""):
    """
    Renders standardized hospital status, live Colab GPU connection widget,
    and navigation across CareFlow portals.
    """
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #0F766E 0%, #1E3A8A 100%); padding: 14px 16px; border-radius: 12px; color: white; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(15,118,110,0.2);">
        <div style="font-weight: 700; font-size: 17px; display: flex; align-items: center; gap: 8px;">
            🏥 CareFlow Health
        </div>
        <div style="font-size: 12px; opacity: 0.92; margin-top: 3px;">
            Clinical Triage & Multilingual Intake
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. System Health Status
    health = api_client.get_health()
    st.sidebar.markdown("##### ⚙️ System Status")

    backend_ok = health.get("status") == "healthy"
    if backend_ok:
        st.sidebar.success("🟢 Hospital Backend: **Online**")
    else:
        st.sidebar.error("🔴 Hospital Backend: **Offline / Reconnecting**")

    # Check active MedGemma runtime
    active_mode = health.get("medgemma_mode", "hosted")
    colab_connected = st.session_state.get("colab_gpu_connected", False)
    
    # Check if active endpoint is configured
    active_endpoint = os.getenv("MEDGEMMA_ENDPOINT", "") or st.session_state.get("colab_endpoint", "")
    
    if colab_connected or active_mode == "hosted" or "trycloudflare" in active_endpoint:
        gpu_name = st.session_state.get("colab_gpu_name", "NVIDIA Tesla T4")
        st.sidebar.info(f"🚀 Neural SLM: **Colab GPU ({gpu_name})**")
    else:
        st.sidebar.info("🧠 Neural SLM: **Local / Clinical Engine**")

    st.sidebar.caption("🌳 Urgency Model: LightGBM (41 Clinical Features)")
    st.sidebar.caption("🎙️ Voice Engine: Deepgram Nova-2 Medical STT")

    # 2. Google Colab GPU Connection Expander
    st.sidebar.divider()
    with st.sidebar.expander("⚡ Google Colab GPU Server", expanded=False):
        st.markdown(
            "Run [`CareFlow_MedGemma_GPU_Colab.ipynb`](https://colab.research.google.com/github/0mM1shra/careflow_ai/blob/main/CareFlow_MedGemma_GPU_Colab.ipynb) "
            "on a free Google Colab T4 GPU and connect it directly below:"
        )
        
        default_val = st.session_state.get(
            "colab_endpoint",
            active_endpoint or "https://refuse-overhead-efficiently-hoped.trycloudflare.com/v1/chat/completions"
        )
        
        user_url = st.text_input(
            "Colab Tunnel URL:",
            value=default_val,
            placeholder="https://xxxx.trycloudflare.com",
            key="colab_url_sidebar_input",
            help="Copy the https://xxxx.trycloudflare.com URL from Step 5 of the Colab notebook."
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔗 Connect", use_container_width=True, type="primary", key="btn_connect_colab"):
                if user_url.strip():
                    with st.spinner("Connecting to Colab GPU..."):
                        clean_url = user_url.strip()
                        if not any(s in clean_url for s in ("/v1/chat/completions", "/api/medgemma/triage")):
                            clean_url = f"{clean_url.rstrip('/')}/v1/chat/completions"
                        
                        # Ping health
                        base_h = clean_url.split("/v1/")[0]
                        try:
                            h_resp = requests.get(f"{base_h}/health", timeout=6)
                            if h_resp.status_code == 200:
                                h_data = h_resp.json()
                                gpu = h_data.get("gpu", "Tesla T4")
                                vram = h_data.get("vram_used_gb", 3.3)
                                
                                # Update backend
                                api_client.update_medgemma_config(clean_url, mode="hosted")
                                st.session_state["colab_endpoint"] = clean_url
                                st.session_state["colab_gpu_connected"] = True
                                st.session_state["colab_gpu_name"] = gpu
                                st.success(f"Connected! {gpu} ({vram} GB VRAM)")
                                st.rerun()
                            else:
                                st.error(f"Endpoint returned HTTP {h_resp.status_code}")
                        except Exception as ex:
                            st.error(f"Cannot reach endpoint: {ex}")
                else:
                    st.warning("Please enter a URL.")

        with col_btn2:
            if st.button("🔄 Local Mode", use_container_width=True, key="btn_reset_local"):
                api_client.update_medgemma_config("", mode="local")
                st.session_state["colab_gpu_connected"] = False
                st.info("Switched to Local mode.")
                st.rerun()

    # 3. Navigation Links
    st.sidebar.divider()
    st.sidebar.markdown("##### 🧭 Navigation")
    if current_page != "home":
        st.sidebar.page_link("app.py", label="Hospital Home", icon="🏥")
    if current_page != "patient":
        st.sidebar.page_link("pages/patient.py", label="Patient Intake Portal", icon="🩺")
    if current_page != "reviewer":
        st.sidebar.page_link("pages/reviewer.py", label="Clinician Reviewer Dashboard", icon="📋")
    if current_page != "architecture":
        st.sidebar.page_link("pages/model_showcase.py", label="Model Architecture & AI", icon="🧠")
