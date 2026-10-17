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

# Bilingual Dictionary
UI_TEXT = {
    "English": {
        "title": "🩺 CareFlow Clinical Intake",
        "subtitle": "Private & Adaptive Symptom Assessment (Powered by MedGemma)",
        "privacy_title": "🛡️ Privacy Protection Notice:",
        "privacy_body": "Please do <strong>NOT</strong> enter your name, telephone number, home address, social security / Aadhaar number, or any other identifying personal information. You will be assigned a random, non-identifying Case ID.",
        "emergency_title": "🚨 POTENTIAL MEDICAL EMERGENCY DETECTED:",
        "emergency_body": "If you are experiencing severe chest pain, trouble breathing, sudden weakness, or another life-threatening emergency, <strong>immediately call 911 / 112 (or local emergency services) or proceed to the nearest Emergency Room.</strong> Do not wait for online clinical review.",
        "start_title": "Begin New Clinical Intake",
        "lang_label": "Preferred Language / भाषा चुनें",
        "complaint_label": "What symptoms or health concerns are you experiencing today?",
        "complaint_placeholder": "e.g., I have had a severe sore throat and high fever for 2 days, and it hurts when I swallow. / मुझे दो दिन से तेज़ बुखार और गले में दर्द है...",
        "doc_upload_header": "📄 Attach Medical Document or Prescription (Optional)",
        "doc_upload_help": "Upload an image (PNG, JPG) or PDF of a prescription, lab test, or clinical discharge report. Native OCR will securely extract the text before sending it to MedGemma.",
        "doc_extracted_alert": "✅ Medical document parsed via OCR and attached to your clinical record.",
        "start_btn": "Start Intake Assessment →",
        "case_id_label": "Case ID",
        "question_count": "Question {curr} of {max}",
        "you_label": "You",
        "assistant_label": "CareFlow Assistant",
        "q_label": "CareFlow Follow-Up Question:",
        "input_label": "Your Response:",
        "input_placeholder": "Type your answer here in English or Hindi (हिन्दी)...",
        "send_btn": "Send Response 💬",
        "skip_btn": "Skip ⏩",
        "unknown_btn": "I Don't Know ❓",
        "early_submit_btn": "Finish & Review Now (Submit Early) 📋",
        "upload_during_chat": "📎 Upload Additional Medical Document / Lab Report (PDF or Image OCR)",
        "upload_btn": "Extract & Ingest Document 📤",
        "voice_header": "🎤 Voice Input (Record or Upload Audio)",
        "voice_help": "Record symptoms in English, Hindi (हिन्दी), or Hinglish. You will review and edit the transcript before sending.",
        "voice_transcript_title": "🎤 Voice Transcript (Review & Edit before sending):",
        "voice_confirm_btn": "Confirm & Send Voice Response 💬",
        "voice_discard_btn": "Discard Recording ❌",
        "doc_preview_title": "📄 Extracted Document Text (Review & Edit before sending):",
        "doc_confirm_btn": "Confirm & Ingest Document Text 📋",
        "doc_discard_btn": "Discard Document ❌",
        "review_title": "Review Your Intake Information",
        "review_subtitle": "Please review the information you provided. You may correct or add details before submitting for clinician review.",
        "submit_btn": "Submit Intake for Healthcare Professional Review ✅",
        "return_btn": "← Return to Questions",
        "success_title": "🎉 Your clinical intake has been successfully submitted!",
        "confirm_card_title": "Intake Confirmation",
        "confirm_status": "Queued for Registered Healthcare Professional Review",
        "confirm_body": "Your responses and uploaded medical documentation have been securely transferred to the clinical reviewer dashboard. A registered medical reviewer will inspect your case notes and clinical summary.",
        "safety_reminder": "<strong>Important Safety Reminder:</strong> CareFlow is an automated clinical assistant prototype. It does not diagnose medical conditions. If your symptoms worsen or you develop severe chest pain, difficulty breathing, or sudden numbness, seek emergency medical care immediately.",
        "restart_btn": "Start Another Patient Intake"
    },
    "हिन्दी (Hindi)": {
        "title": "🩺 CareFlow क्लिनिकल इनटेक",
        "subtitle": "निजी एवं अनुकूली लक्षण मूल्यांकन (Google MedGemma द्वारा संचालित)",
        "privacy_title": "🛡️ गोपनीयता संरक्षण सूचना:",
        "privacy_body": "कृपया अपना <strong>नाम, फ़ोन नंबर, घर का पता, आधार कार्ड नंबर</strong> या कोई अन्य पहचान योग्य व्यक्तिगत जानकारी दर्ज <strong>न करें</strong>। आपको एक सुरक्षित, गैर-पहचान योग्य केस आईडी प्रदान की जाएगी।",
        "emergency_title": "🚨 संभावित आपातकालीन स्थिति पहचानी गई:",
        "emergency_body": "यदि आप सीने में तेज़ दर्द, सांस लेने में अत्यधिक तकलीफ़, अचानक कमजोरी या कोई अन्य गंभीर आपातकालीन स्थिति महसूस कर रहे हैं, तो <strong>तुरंत आपातकालीन सेवा (112 / 108) पर कॉल करें या नज़दीकी अस्पताल के इमरजेंसी वार्ड में जाएं।</strong> ऑनलाइन समीक्षा की प्रतीक्षा न करें।",
        "start_title": "नया क्लिनिकल इनटेक शुरू करें",
        "lang_label": "पसंदीदा भाषा (Preferred Language)",
        "complaint_label": "आज आप किन लक्षणों या स्वास्थ्य संबंधी परेशानियों का सामना कर रहे हैं?",
        "complaint_placeholder": "उदा: मुझे दो दिन से तेज़ बुखार, गले में ख़राश और दर्द है, और निगलने में बहुत तकलीफ़ हो रही है...",
        "doc_upload_header": "📄 मेडिकल पर्ची या दस्तावेज़ अपलोड करें (वैकल्पिक)",
        "doc_upload_help": "डॉक्टर की पर्ची, लैब टेस्ट या डिस्चार्ज समरी की फ़ोटो (PNG, JPG) या PDF अपलोड करें। हमारा OCR सिस्टम सुरक्षित रूप से टेक्स्ट निकाल कर MedGemma को प्रदान करेगा।",
        "doc_extracted_alert": "✅ दस्तावेज़ से OCR द्वारा टेक्स्ट सफलतापूर्वक प्राप्त कर लिया गया है।",
        "start_btn": "इनटेक मूल्यांकन शुरू करें →",
        "case_id_label": "केस आईडी",
        "question_count": "प्रश्न {curr} / {max}",
        "you_label": "आप",
        "assistant_label": "CareFlow सहायक",
        "q_label": "CareFlow अनुवर्ती प्रश्न:",
        "input_label": "आपका उत्तर:",
        "input_placeholder": "यहाँ अपना उत्तर हिंदी या अंग्रेजी में लिखें...",
        "send_btn": "उत्तर भेजें 💬",
        "skip_btn": "छोड़ें ⏩",
        "unknown_btn": "मुझे नहीं पता ❓",
        "early_submit_btn": "समीक्षा करें एवं अभी जमा करें 📋",
        "upload_during_chat": "📎 अतिरिक्त मेडिकल दस्तावेज़ या रिपोर्ट अपलोड करें (PDF/Image OCR)",
        "upload_btn": "दस्तावेज़ से टेक्स्ट निकालें एवं जोड़ें 📤",
        "voice_header": "🎤 बोलकर बताएं (आवाज़ रिकॉर्ड करें या ऑडियो अपलोड करें)",
        "voice_help": "हिंदी, अंग्रेजी या हिंग्लिश में बोलें। भेजने से पहले आप टेक्स्ट की समीक्षा और सुधार कर सकेंगे।",
        "voice_transcript_title": "🎤 वॉइस ट्रांसक्रिप्ट (भेजने से पहले समीक्षा एवं सुधार करें):",
        "voice_confirm_btn": "पुष्टि करें और उत्तर भेजें 💬",
        "voice_discard_btn": "रद्द करें ❌",
        "doc_preview_title": "📄 दस्तावेज़ से प्राप्त टेक्स्ट (समीक्षा एवं सुधार करें):",
        "doc_confirm_btn": "पुष्टि करें और दस्तावेज़ जोड़ें 📋",
        "doc_discard_btn": "दस्तावेज़ हटाएं ❌",
        "review_title": "अपनी इनटेक जानकारी की समीक्षा करें",
        "review_subtitle": "कृपया अपने दिए गए उत्तरों की समीक्षा करें। डॉक्टर/समीक्षक के पास जमा करने से पहले आप सुधार कर सकते हैं।",
        "submit_btn": "स्वास्थ्य विशेषज्ञ समीक्षा हेतु इनटेक जमा करें ✅",
        "return_btn": "← प्रश्नों पर वापस जाएँ",
        "success_title": "🎉 आपका क्लिनिकल इनटेक सफलतापूर्वक जमा हो गया है!",
        "confirm_card_title": "इनटेक पावती (Confirmation)",
        "confirm_status": "पंजीकृत स्वास्थ्य समीक्षक समीक्षा हेतु कतारबद्ध",
        "confirm_body": "आपके उत्तर और अपलोड किए गए मेडिकल दस्तावेज़ क्लिनिकल समीक्षक डैशबोर्ड पर सुरक्षित रूप से स्थानांतरित कर दिए गए हैं। एक पंजीकृत चिकित्सक आपके केस और सारांश की समीक्षा करेंगे।",
        "safety_reminder": "<strong>महत्वपूर्ण सुरक्षा चेतावनी:</strong> CareFlow एक स्वचालित क्लिनिकल सहायक प्रोटोटाइप है। यह कोई चिकित्सा निदान (डायग्नोसिस) नहीं करता है। यदि आपके लक्षण बिगड़ते हैं या आपको सीने में तेज़ दर्द अथवा सांस लेने में तकलीफ़ होती है, तो तुरंत आपातकालीन चिकित्सा सहायता लें।",
        "restart_btn": "नया इनटेक शुरू करें"
    }
}

# Initialize Session State
if "selected_lang" not in st.session_state:
    st.session_state.selected_lang = "English"
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
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []
if "pending_voice" not in st.session_state:
    st.session_state.pending_voice = None
if "pending_doc_preview" not in st.session_state:
    st.session_state.pending_doc_preview = None

# Language selector bar in sidebar / top
with st.sidebar:
    st.markdown("### 🌐 Language / भाषा")
    chosen_lang = st.radio(
        "Interface & Conversation Language:",
        options=["English", "हिन्दी (Hindi)"],
        index=0 if st.session_state.selected_lang == "English" else 1,
        key="lang_radio"
    )
    if chosen_lang != st.session_state.selected_lang:
        st.session_state.selected_lang = chosen_lang
        st.rerun()

t = UI_TEXT[st.session_state.selected_lang]

# Header
st.markdown(f"""
<div class="patient-header">
    <h2 style="margin:0; font-size: 24px;">{t['title']}</h2>
    <div style="font-size: 14px; opacity: 0.9; margin-top: 4px;">{t['subtitle']}</div>
</div>
""", unsafe_allow_html=True)

# Privacy Warning
st.markdown(f"""
<div class="privacy-notice">
    <strong>{t['privacy_title']}</strong><br/>
    {t['privacy_body']}
</div>
""", unsafe_allow_html=True)

# Emergency Warning Banner (Conditional)
if st.session_state.is_emergency:
    st.markdown(f"""
    <div class="emergency-banner">
        <strong>{t['emergency_title']}</strong><br/>
        {t['emergency_body']}
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# STEP 1: START INTAKE (Language, Chief Complaint, & Document OCR)
# =====================================================================
if st.session_state.intake_step == "start":
    st.subheader(t["start_title"])

    # Document upload accordion before/with chief complaint
    with st.expander(t["doc_upload_header"], expanded=False):
        st.caption(t["doc_upload_help"])
        doc_file_start = st.file_uploader(
            "Upload Prescription or Medical Document (PDF, PNG, JPG, JPEG)",
            type=["pdf", "png", "jpg", "jpeg"],
            key="doc_start"
        )
        if doc_file_start is not None:
            st.info(f"📁 Selected: **{doc_file_start.name}** ({doc_file_start.size} bytes). This will be scanned via OCR and attached to your intake.")

    chief_complaint = st.text_area(
        t["complaint_label"],
        placeholder=t["complaint_placeholder"],
        height=120
    )

    if st.button(t["start_btn"], type="primary", use_container_width=True):
        if not chief_complaint.strip() and not doc_file_start:
            st.warning("Please provide a description of what you are experiencing or upload a medical document.")
        else:
            with st.spinner("Initializing private intake session & analyzing medical input..."):
                try:
                    # Map selected UI language to backend format
                    backend_lang = "Hindi" if "Hindi" in st.session_state.selected_lang else "English"

                    # 1. Create case in backend
                    initial_complaint = chief_complaint.strip() if chief_complaint.strip() else "Medical document attached for intake assessment"
                    new_case = api_client.create_case(
                        preferred_language=backend_lang,
                        chief_complaint=initial_complaint
                    )
                    case_id = new_case["id"]
                    st.session_state.case_id = case_id

                    # 2. If a document was uploaded, process OCR via upload endpoint
                    if doc_file_start is not None:
                        file_bytes = doc_file_start.getvalue()
                        up_res = api_client.upload_document(case_id, file_bytes, doc_file_start.name)
                        st.session_state.uploaded_docs.append({
                            "name": doc_file_start.name,
                            "char_count": up_res.get("char_count", 0),
                            "ocr_preview": up_res.get("extracted_text_snippet", "")
                        })
                        if up_res.get("is_emergency"):
                            st.session_state.is_emergency = True

                    # 3. Add first message to backend if chief complaint was typed
                    if chief_complaint.strip():
                        msg_res = api_client.send_message(
                            case_id,
                            chief_complaint.strip(),
                            source_type="patient_typed",
                            source_language=backend_lang,
                            is_confirmed=True
                        )
                        st.session_state.messages.append({
                            "role": "patient",
                            "content": chief_complaint.strip(),
                            "source_type": "patient_typed",
                            "source_language": backend_lang
                        })
                        if msg_res.get("is_emergency"):
                            st.session_state.is_emergency = True
                    elif doc_file_start is not None:
                        st.session_state.messages.append({
                            "role": "patient",
                            "content": f"[Attached Medical Document: {doc_file_start.name}]",
                            "source_type": "ocr"
                        })

                    # 4. Request first follow-up question
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
# STEP 2: CONVERSATIONAL ADAPTIVE QUESTIONING & IN-CHAT OCR UPLOAD
# =====================================================================
elif st.session_state.intake_step == "chatting":
    q_str = t["question_count"].format(curr=st.session_state.turn_count, max=st.session_state.max_turns)
    st.caption(f"{t['case_id_label']}: **{st.session_state.case_id}** | {q_str}")
    
    # Progress bar
    progress_val = min(st.session_state.turn_count / st.session_state.max_turns, 1.0)
    st.progress(progress_val)

    # Conversation history
    st.write("---")
    for msg in st.session_state.messages[:-1]:  # Show history except latest assistant prompt
        if msg["role"] == "patient":
            src = msg.get("source_type", "patient_typed")
            badge = ""
            if src == "voice":
                badge = '<span style="background:#EEF2FF; color:#4338CA; border:1px solid #C7D2FE; font-size:11px; font-weight:700; padding:2px 6px; border-radius:10px; margin-right:6px;">🎤 Voice Transcript</span>'
            elif src == "ocr":
                badge = '<span style="background:#ECFDF5; color:#047857; border:1px solid #A7F3D0; font-size:11px; font-weight:700; padding:2px 6px; border-radius:10px; margin-right:6px;">📄 Document OCR</span>'
            st.markdown(f'<div class="chat-bubble-patient">{badge}<strong>{t["you_label"]}:</strong> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-assistant"><strong>{t["assistant_label"]}:</strong> {msg["content"]}</div>', unsafe_allow_html=True)

    # Current Question Card
    current_q = st.session_state.current_question or ("Could you clarify any other symptoms you are experiencing?" if "English" in st.session_state.selected_lang else "क्या आप अपने लक्षणों के बारे में कुछ और विस्तार से बता सकते हैं?")
    st.markdown(f'<div class="chat-bubble-assistant" style="margin-right: 0; background: #E0F2FE; border-color: #BAE6FD; color: #0369A1;">'
                f'<strong>{t["q_label"]}</strong><br/><span style="font-size: 16px;">{current_q}</span></div>', unsafe_allow_html=True)

    backend_lang = "Hindi" if "Hindi" in st.session_state.selected_lang else "English"

    # -------------------------------------------------------------
    # CASE A: PENDING VOICE TRANSCRIPT REVIEW (Human-in-the-loop)
    # -------------------------------------------------------------
    if st.session_state.pending_voice is not None:
        pv = st.session_state.pending_voice
        p_trans = pv.get("transcript", "")
        p_lang = pv.get("language", backend_lang)
        p_conf = pv.get("confidence", 0.0)
        conf_pct = int(p_conf * 100) if p_conf is not None else 0
        conf_status = "High confidence" if (p_conf or 0) >= 0.70 else "⚠️ Lower confidence — please verify accuracy"

        st.markdown(f"""
        <div style="background-color: #F5F3FF; border: 2px solid #C4B5FD; border-radius: 8px; padding: 14px; margin-bottom: 12px;">
            <h4 style="margin: 0 0 6px 0; color: #5B21B6; font-size: 15px;">{t['voice_transcript_title']}</h4>
            <div style="font-size: 13px; color: #6D28D9;">
                Detected: <strong>{p_lang.title()}</strong> | Acoustic Confidence: <strong>{conf_pct}% ({conf_status})</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        edited_voice_text = st.text_area(
            "Review or edit transcribed text before submitting:",
            value=p_trans,
            key="voice_transcript_edit",
            height=100
        )
        col_v1, col_v2 = st.columns([2, 1])
        with col_v1:
            if st.button(t["voice_confirm_btn"], type="primary", key="btn_confirm_voice", use_container_width=True):
                if edited_voice_text.strip():
                    with st.spinner("Submitting verified voice response..."):
                        try:
                            send_res = api_client.send_message(
                                case_id=st.session_state.case_id,
                                content=edited_voice_text.strip(),
                                source_type="voice",
                                source_language=p_lang,
                                transcription_confidence=p_conf,
                                is_confirmed=True
                            )
                            st.session_state.messages.append({
                                "role": "patient",
                                "content": edited_voice_text.strip(),
                                "source_type": "voice",
                                "source_language": p_lang,
                                "transcription_confidence": p_conf
                            })
                            if send_res.get("is_emergency"):
                                st.session_state.is_emergency = True
                            st.session_state.pending_voice = None

                            q_res = api_client.get_next_question(st.session_state.case_id)
                            st.session_state.turn_count += 1
                            if q_res.get("is_terminal") or not q_res.get("next_question") or st.session_state.turn_count > st.session_state.max_turns:
                                st.session_state.intake_step = "review_edit"
                            else:
                                st.session_state.current_question = q_res["next_question"]
                                st.session_state.messages.append({"role": "assistant", "content": q_res["next_question"]})
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error submitting voice response: {e}")
                else:
                    st.warning("Please enter text or discard the recording.")
        with col_v2:
            if st.button(t["voice_discard_btn"], key="btn_discard_voice", use_container_width=True):
                st.session_state.pending_voice = None
                st.rerun()

    # -------------------------------------------------------------
    # CASE B: PENDING DOCUMENT OCR REVIEW (Human-in-the-loop)
    # -------------------------------------------------------------
    elif st.session_state.pending_doc_preview is not None:
        pd = st.session_state.pending_doc_preview
        doc_text = pd.get("extracted_text", "")
        doc_fname = pd.get("filename", "document")
        doc_chars = pd.get("char_count", 0)
        doc_engine = pd.get("ocr_engine", "unknown")
        doc_conf = pd.get("confidence", 1.0)

        st.markdown(f"""
        <div style="background-color: #F0FDF4; border: 2px solid #86EFAC; border-radius: 8px; padding: 14px; margin-bottom: 12px;">
            <h4 style="margin: 0 0 6px 0; color: #166534; font-size: 15px;">{t['doc_preview_title']}</h4>
            <div style="font-size: 13px; color: #15803D;">
                File: <strong>{doc_fname}</strong> | Engine: <code>{doc_engine}</code> | Length: <strong>{doc_chars} chars</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        edited_doc_text = st.text_area(
            "Review or edit extracted medical report findings before ingesting:",
            value=doc_text,
            key="doc_ocr_edit",
            height=140
        )
        col_d1, col_d2 = st.columns([2, 1])
        with col_d1:
            if st.button(t["doc_confirm_btn"], type="primary", key="btn_confirm_doc", use_container_width=True):
                if edited_doc_text.strip():
                    with st.spinner("Ingesting verified document findings..."):
                        try:
                            content_str = f"[Verified Medical Report: {doc_fname}]\n{edited_doc_text.strip()}"
                            send_res = api_client.send_message(
                                case_id=st.session_state.case_id,
                                content=content_str,
                                source_type="ocr",
                                transcription_confidence=doc_conf,
                                is_confirmed=True
                            )
                            st.session_state.uploaded_docs.append({
                                "name": doc_fname,
                                "char_count": len(edited_doc_text),
                                "ocr_preview": edited_doc_text[:200]
                            })
                            st.session_state.messages.append({
                                "role": "patient",
                                "content": content_str,
                                "source_type": "ocr"
                            })
                            if send_res.get("is_emergency"):
                                st.session_state.is_emergency = True
                            st.session_state.pending_doc_preview = None

                            q_res = api_client.get_next_question(st.session_state.case_id)
                            st.session_state.turn_count += 1
                            if q_res.get("is_terminal") or not q_res.get("next_question") or st.session_state.turn_count > st.session_state.max_turns:
                                st.session_state.intake_step = "review_edit"
                            else:
                                st.session_state.current_question = q_res["next_question"]
                                st.session_state.messages.append({"role": "assistant", "content": q_res["next_question"]})
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error ingesting document text: {e}")
                else:
                    st.warning("Please retain some text or discard.")
        with col_d2:
            if st.button(t["doc_discard_btn"], key="btn_discard_doc", use_container_width=True):
                st.session_state.pending_doc_preview = None
                st.rerun()

    # -------------------------------------------------------------
    # CASE C: STANDARD PATIENT INPUT (Typed / Voice / Report OCR)
    # -------------------------------------------------------------
    else:
        # Patient Typed Answer Form
        with st.form("patient_answer_form", clear_on_submit=True):
            patient_answer = st.text_input(t["input_label"], placeholder=t["input_placeholder"], key="current_input")
            col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
            with col_btn1:
                submit_answer = st.form_submit_button(t["send_btn"], type="primary", use_container_width=True)
            with col_btn2:
                skip_question = st.form_submit_button(t["skip_btn"], use_container_width=True)
            with col_btn3:
                unknown_answer = st.form_submit_button(t["unknown_btn"], use_container_width=True)

        content_to_send = None
        if submit_answer:
            if patient_answer.strip():
                content_to_send = patient_answer.strip()
            else:
                st.warning("Please enter your answer or click 'Skip' / 'I Don't Know'.")
        elif skip_question:
            content_to_send = "I choose to skip this question." if "English" in st.session_state.selected_lang else "मैं इस प्रश्न को छोड़ना चाहता हूँ।"
        elif unknown_answer:
            content_to_send = "I do not know or am not sure of the answer to this question." if "English" in st.session_state.selected_lang else "मुझे इस प्रश्न का उत्तर निश्चित रूप से नहीं पता है।"

        if content_to_send:
            with st.spinner("Processing your response..."):
                try:
                    send_res = api_client.send_message(
                        st.session_state.case_id,
                        content_to_send,
                        source_type="patient_typed",
                        source_language=backend_lang,
                        is_confirmed=True
                    )
                    st.session_state.messages.append({
                        "role": "patient",
                        "content": content_to_send,
                        "source_type": "patient_typed",
                        "source_language": backend_lang
                    })

                    if send_res.get("is_emergency"):
                        st.session_state.is_emergency = True

                    q_res = api_client.get_next_question(st.session_state.case_id)
                    st.session_state.turn_count += 1

                    if q_res.get("is_terminal") or not q_res.get("next_question") or st.session_state.turn_count > st.session_state.max_turns:
                        st.session_state.intake_step = "review_edit"
                    else:
                        st.session_state.current_question = q_res["next_question"]
                        st.session_state.messages.append({"role": "assistant", "content": q_res["next_question"]})

                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing message: {e}")

        # Multimodal Input Expanders
        col_exp1, col_exp2 = st.columns(2)

        with col_exp1:
            with st.expander(t["voice_header"], expanded=False):
                st.caption(t["voice_help"])
                rec_audio = None
                if hasattr(st, "audio_input"):
                    rec_audio = st.audio_input("Record audio (Mic)", key="mic_audio_chat")
                upl_audio = st.file_uploader(
                    "Or upload voice file (WAV, MP3, M4A, OGG, WEBM, FLAC)",
                    type=["wav", "mp3", "m4a", "ogg", "webm", "flac"],
                    key="upl_audio_chat"
                )
                chosen_audio = rec_audio or upl_audio
                if chosen_audio is not None:
                    if st.button("Transcribe Audio 🎙️", type="primary", key="btn_run_transcribe", use_container_width=True):
                        with st.spinner("Transcribing voice audio via faster-whisper..."):
                            try:
                                audio_bytes = chosen_audio.getvalue()
                                fname = getattr(chosen_audio, "name", "recording.wav") or "recording.wav"
                                t_res = api_client.transcribe_audio(
                                    file_bytes=audio_bytes,
                                    filename=fname,
                                    language=backend_lang
                                )
                                st.session_state.pending_voice = t_res
                                st.rerun()
                            except Exception as e:
                                st.error(f"Transcription error: {e}")

        with col_exp2:
            with st.expander(t["upload_during_chat"], expanded=False):
                st.caption("Upload prescription or report image/PDF for OCR preview before adding to record.")
                doc_in_chat = st.file_uploader(
                    "Upload image/PDF (PNG, JPG, PDF)",
                    type=["pdf", "png", "jpg", "jpeg", "webp"],
                    key="doc_chat"
                )
                if doc_in_chat is not None:
                    if st.button("Extract & Preview Document Text 🔍", type="secondary", key="btn_run_preview_doc", use_container_width=True):
                        with st.spinner("Extracting text via OCR (PaddleOCR/winocr)..."):
                            try:
                                file_bytes = doc_in_chat.getvalue()
                                preview_res = api_client.preview_document_extraction(
                                    case_id=st.session_state.case_id,
                                    file_bytes=file_bytes,
                                    filename=doc_in_chat.name
                                )
                                st.session_state.pending_doc_preview = preview_res
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error previewing document: {e}")

    # Early submission option
    st.write("---")
    col_early, _ = st.columns([2, 1])
    with col_early:
        if st.button(t["early_submit_btn"], use_container_width=True):
            st.session_state.intake_step = "review_edit"
            st.rerun()

# =====================================================================
# STEP 3: PRE-SUBMISSION REVIEW & EDIT SCREEN
# =====================================================================
elif st.session_state.intake_step == "review_edit":
    st.subheader(t["review_title"])
    st.write(t["review_subtitle"])

    if st.session_state.uploaded_docs:
        with st.expander(f"📄 Uploaded Documents ({len(st.session_state.uploaded_docs)})", expanded=True):
            for d in st.session_state.uploaded_docs:
                st.markdown(f"- **{d['name']}** ({d['char_count']} chars extracted via OCR)")
                if d.get("ocr_preview"):
                    st.caption(f"Preview: {d['ocr_preview'][:200]}...")

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
        if st.button(t["submit_btn"], type="primary", use_container_width=True):
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
        if st.button(t["return_btn"], use_container_width=True):
            st.session_state.intake_step = "chatting"
            st.rerun()

# =====================================================================
# STEP 4: SUBMISSION CONFIRMATION
# =====================================================================
elif st.session_state.intake_step == "submitted":
    st.balloons()
    st.success(t["success_title"])

    st.markdown(f"""
    <div class="card-review">
        <h3>{t['confirm_card_title']}</h3>
        <p><strong>{t['case_id_label']}:</strong> <code>{st.session_state.case_id}</code></p>
        <p><strong>Status:</strong> {t['confirm_status']}</p>
        <p>{t['confirm_body']}</p>
        <div style="background: #F1F5F9; padding: 12px; border-radius: 6px; font-size: 13.5px; color: #475569;">
            {t['safety_reminder']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button(t["restart_btn"], use_container_width=True):
        st.session_state.case_id = None
        st.session_state.intake_step = "start"
        st.session_state.messages = []
        st.session_state.current_question = None
        st.session_state.is_emergency = False
        st.session_state.turn_count = 0
        st.session_state.uploaded_docs = []
        st.rerun()
