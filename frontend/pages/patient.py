import streamlit as st
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from frontend.api_client import api_client

st.set_page_config(
    page_title="Patient Intake | CareFlow",
    page_icon="🩺",
    layout="centered"
)

# Clean Clinical CSS
st.markdown("""
<style>
    .patient-header {
        background: linear-gradient(135deg, #0F766E 0%, #0D9488 100%);
        color: white;
        padding: 20px 24px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .privacy-notice {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-left: 4px solid #2563EB;
        padding: 12px 16px;
        border-radius: 6px;
        color: #1E40AF;
        font-size: 13.5px;
        margin-bottom: 20px;
    }
    .emergency-banner {
        background-color: #FEE2E2;
        border: 2px solid #DC2626;
        padding: 16px;
        border-radius: 8px;
        color: #991B1B;
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 24px;
    }
    .chat-bubble-patient {
        background-color: #E2E8F0;
        color: #1E293B;
        padding: 12px 16px;
        border-radius: 12px 12px 0 12px;
        margin-left: 20%;
        margin-bottom: 12px;
        font-size: 14.5px;
    }
    .chat-bubble-assistant {
        background-color: #CCFBF1;
        color: #115E59;
        border: 1px solid #99F6E4;
        padding: 14px 18px;
        border-radius: 12px 12px 12px 0;
        margin-right: 20%;
        margin-bottom: 12px;
        font-size: 15px;
        font-weight: 500;
    }
    .card-review {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "case_id" not in st.session_state:
    st.session_state.case_id = None
if "intake_step" not in st.session_state:
    st.session_state.intake_step = "start"  # 'start', 'chatting', 'review_edit', 'submitted'
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_question" not in st.session_state:
    st.session_state.current_question = None
if "is_emergency" not in st.session_state:
    st.session_state.is_emergency = False
if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0
if "max_turns" not in st.session_state:
    st.session_state.max_turns = 8

# Header
st.markdown("""
<div class="patient-header">
    <h2 style="margin:0; font-size: 24px;">🩺 CareFlow Clinical Intake</h2>
    <div style="font-size: 14px; opacity: 0.9; margin-top: 4px;">Private & Adaptive Symptom Assessment</div>
</div>
""", unsafe_allow_html=True)

# Privacy Warning
st.markdown("""
<div class="privacy-notice">
    <strong>🛡️ Privacy Protection Notice:</strong>
    Please do <strong>NOT</strong> enter your name, telephone number, home address, social security / government ID number,
    or any other identifying personal information. You will be assigned a random, non-identifying Case ID.
</div>
""", unsafe_allow_html=True)

# Emergency Warning Banner (Conditional)
if st.session_state.is_emergency:
    st.markdown("""
    <div class="emergency-banner">
        🚨 <strong>POTENTIAL MEDICAL EMERGENCY DETECTED:</strong><br/>
        If you are experiencing severe chest pain, trouble breathing, sudden weakness, or another life-threatening emergency,
        <strong>immediately call 911 (or your local emergency services) or proceed to the nearest Emergency Room.</strong>
        Do not wait for online clinical review.
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# STEP 1: START INTAKE (Language & Chief Complaint)
# =====================================================================
if st.session_state.intake_step == "start":
    st.subheader("Begin New Clinical Intake")

    col_lang, _ = st.columns([1, 1])
    with col_lang:
        pref_lang = st.selectbox("Preferred Language", ["English", "Spanish", "French", "Other"], index=0)

    chief_complaint = st.text_area(
        "What symptoms or health concern are you experiencing today?",
        placeholder="e.g., I have had a severe sore throat and fever for the past 2 days, and it hurts when I swallow.",
        height=120
    )

    if st.button("Start Intake Assessment →", type="primary", use_container_width=True):
        if not chief_complaint.strip():
            st.warning("Please provide a brief description of what you are experiencing.")
        else:
            with st.spinner("Initializing private intake session..."):
                try:
                    # 1. Create case in backend
                    new_case = api_client.create_case(
                        preferred_language=pref_lang,
                        chief_complaint=chief_complaint.strip()
                    )
                    case_id = new_case["id"]
                    st.session_state.case_id = case_id

                    # 2. Add first message to backend
                    msg_res = api_client.send_message(case_id, chief_complaint.strip())
                    st.session_state.messages.append({"role": "patient", "content": chief_complaint.strip()})

                    if msg_res.get("is_emergency"):
                        st.session_state.is_emergency = True

                    # 3. Request first follow-up question
                    q_res = api_client.get_next_question(case_id)
                    st.session_state.turn_count = 1

                    if q_res.get("next_question"):
                        st.session_state.current_question = q_res["next_question"]
                        st.session_state.messages.append({"role": "assistant", "content": q_res["next_question"]})
                        st.session_state.intake_step = "chatting"
                    else:
                        st.session_state.intake_step = "review_edit"

                    st.rerun()
                except Exception as e:
                    st.error(f"Error connecting to CareFlow backend: {e}")

# =====================================================================
# STEP 2: CONVERSATIONAL ADAPTIVE QUESTIONING
# =====================================================================
elif st.session_state.intake_step == "chatting":
    st.caption(f"Case ID: **{st.session_state.case_id}** | Question {st.session_state.turn_count} of {st.session_state.max_turns}")
    
    # Progress bar
    progress_val = min(st.session_state.turn_count / st.session_state.max_turns, 1.0)
    st.progress(progress_val)

    # Conversation history
    st.write("---")
    for msg in st.session_state.messages[:-1]:  # Show history except latest assistant prompt
        if msg["role"] == "patient":
            st.markdown(f'<div class="chat-bubble-patient"><strong>You:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-assistant"><strong>CareFlow Assistant:</strong> {msg["content"]}</div>', unsafe_allow_html=True)

    # Current Question Card
    current_q = st.session_state.current_question or "Could you clarify any other symptoms you are experiencing?"
    st.markdown(f'<div class="chat-bubble-assistant" style="margin-right: 0; background: #E0F2FE; border-color: #BAE6FD; color: #0369A1;">'
                f'<strong>CareFlow Question:</strong><br/><span style="font-size: 16px;">{current_q}</span></div>', unsafe_allow_html=True)

    # Patient Answer Form
    with st.form("patient_answer_form", clear_on_submit=True):
        patient_answer = st.text_input("Your Response:", placeholder="Type your answer here...", key="current_input")
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
        with col_btn1:
            submit_answer = st.form_submit_button("Send Response 💬", type="primary", use_container_width=True)
        with col_btn2:
            skip_question = st.form_submit_button("Skip ⏩", use_container_width=True)
        with col_btn3:
            unknown_answer = st.form_submit_button("I Don't Know ❓", use_container_width=True)

    # Process Form Submission
    content_to_send = None
    if submit_answer:
        if patient_answer.strip():
            content_to_send = patient_answer.strip()
        else:
            st.warning("Please enter your answer or click 'Skip' / 'I Don't Know'.")
    elif skip_question:
        content_to_send = "I choose to skip this question."
    elif unknown_answer:
        content_to_send = "I do not know or am not sure of the answer to this question."

    if content_to_send:
        with st.spinner("Processing your response..."):
            try:
                # 1. Send message to backend
                send_res = api_client.send_message(st.session_state.case_id, content_to_send)
                st.session_state.messages.append({"role": "patient", "content": content_to_send})

                if send_res.get("is_emergency"):
                    st.session_state.is_emergency = True

                # 2. Get next question
                q_res = api_client.get_next_question(st.session_state.case_id)
                st.session_state.turn_count += 1

                if q_res.get("is_terminal") or not q_res.get("next_question") or st.session_state.turn_count > st.session_state.max_turns:
                    # Proceed to Review Screen
                    st.session_state.intake_step = "review_edit"
                else:
                    st.session_state.current_question = q_res["next_question"]
                    st.session_state.messages.append({"role": "assistant", "content": q_res["next_question"]})

                st.rerun()
            except Exception as e:
                st.error(f"Error processing message: {e}")

    # Early submission option
    st.write("---")
    col_early, col_back = st.columns([2, 1])
    with col_early:
        if st.button("Finish & Review Now (Submit Early) 📋", use_container_width=True):
            st.session_state.intake_step = "review_edit"
            st.rerun()

# =====================================================================
# STEP 3: PRE-SUBMISSION REVIEW & EDIT SCREEN
# =====================================================================
elif st.session_state.intake_step == "review_edit":
    st.subheader("Review Your Intake Information")
    st.write("Please review the responses you provided. You may correct or add details before submitting for clinician review.")

    patient_responses = [m["content"] for m in st.session_state.messages if m["role"] == "patient"]

    edited_texts = []
    for idx, resp in enumerate(patient_responses):
        with st.container():
            st.markdown(f"**Item {idx+1}:**")
            new_text = st.text_area(
                f"Response #{idx+1}",
                value=resp,
                key=f"edit_resp_{idx}",
                label_visibility="collapsed"
            )
            edited_texts.append(new_text)

    col_sub, col_add = st.columns([2, 1])
    with col_sub:
        if st.button("Submit Intake for Healthcare Professional Review ✅", type="primary", use_container_width=True):
            with st.spinner("Submitting intake to clinical reviewer queue..."):
                try:
                    # Check if patient modified any responses, if so send updated statement
                    for i, (orig, edited) in enumerate(zip(patient_responses, edited_texts)):
                        if orig != edited and edited.strip():
                            api_client.send_message(
                                st.session_state.case_id,
                                f"Correction to Item #{i+1}: {edited.strip()}"
                            )

                    # Finalize submission
                    submit_res = api_client.submit_case(st.session_state.case_id)
                    st.session_state.intake_step = "submitted"
                    st.rerun()
                except Exception as e:
                    st.error(f"Submission error: {e}")

    with col_add:
        if st.button("← Return to Questions", use_container_width=True):
            st.session_state.intake_step = "chatting"
            st.rerun()

# =====================================================================
# STEP 4: SUBMISSION CONFIRMATION
# =====================================================================
elif st.session_state.intake_step == "submitted":
    st.balloons()
    st.success("🎉 Your clinical intake has been successfully submitted!")

    st.markdown(f"""
    <div class="card-review">
        <h3>Intake Confirmation</h3>
        <p><strong>Case Reference ID:</strong> <code>{st.session_state.case_id}</code></p>
        <p><strong>Status:</strong> Queued for Registered Healthcare Professional Review</p>
        <p>Your responses have been securely transferred to the clinical reviewer dashboard.
        A registered medical reviewer will inspect your case notes and clinical summary.</p>
        <div style="background: #F1F5F9; padding: 12px; border-radius: 6px; font-size: 13.5px; color: #475569;">
            <strong>Important Safety Reminder:</strong> CareFlow is an automated clinical assistant prototype.
            It does not diagnose medical conditions. If your symptoms worsen or you develop severe chest pain,
            difficulty breathing, or sudden numbness, seek emergency medical care immediately.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Start Another Patient Intake", use_container_width=True):
        st.session_state.case_id = None
        st.session_state.intake_step = "start"
        st.session_state.messages = []
        st.session_state.current_question = None
        st.session_state.is_emergency = False
        st.session_state.turn_count = 0
        st.rerun()
