# CareFlow: MedGemma-Powered Clinical Intake, Multimodal Triage & Reviewer Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://careflow-ai-d.streamlit.app/)
[![Live Demo](https://img.shields.io/badge/Live_Demo-CareFlow_Streamlit_App-0F766E?style=for-the-badge&logo=streamlit)](https://careflow-ai-d.streamlit.app/)

> 🌐 **Live Web Application**: **[https://careflow-ai-d.streamlit.app/](https://careflow-ai-d.streamlit.app/)**  
> Experience the live Patient Symptom Intake, Clinician Reviewer Dashboard (Staff PIN: `CAREFLOW`), and AI Model Architecture Showcase directly in your browser.

**CareFlow** is an end-to-end clinical intake and triage decision-support prototype built for medical research and hackathons. It integrates **Google MedGemma** for conversational clinical fact extraction and grounded summarization, paired with an **auditable, deterministic rule-based triage engine**, **multilingual capability (English, Devanagari Hindi, and Hinglish)**, **local voice speech-to-text with patient confirmation**, **medical report OCR ingestion**, and a **clinician-in-the-loop review dashboard with complete provenance tracking**.

> **⚠️ RESEARCH PROTOTYPE DISCLAIMER**
> CareFlow is an exploratory clinical decision support and intake assistant prototype. It is **NOT** a certified medical diagnostic device or clinically validated emergency triage system. It does not diagnose medical conditions or provide treatment instructions. All clinical summaries and priority classifications must be verified, edited, and approved by a qualified healthcare professional. In a life-threatening medical emergency, call 911 (or 112 / local emergency services) or visit the nearest emergency department immediately.

---

## 1. System Architecture & Multimodal Workflow

CareFlow strictly isolates generative AI capabilities from safety-critical triage decisions and enforces human-in-the-loop confirmation before clinical data ingestion:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                   Streamlit Frontend                                    │
│   ├── Patient Interface (/patient): Trilingual text/voice intake, OCR upload & review   │
│   ├── Reviewer Dashboard (/reviewer): Priority queue, atomic evidence & provenance      │
│   └── Evaluation Suite (/evaluation): Live benchmark verification & policy trials       │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │ HTTP Requests
┌────────────────────────────────────────────▼────────────────────────────────────────────┐
│                                     FastAPI Backend                                     │
│   ├── /cases, /messages, /next-question, /submit, /review, /override, /reopen           │
│   ├── /transcribe-audio (faster-whisper), /cases/{id}/extract-document (OCR preview)   │
│   └── Database: SQLite (SQLAlchemy ORM with Foreign Keys & Automatic Schema Migrations) │
└──────────────────────┬──────────────────────────────────────────────┬───────────────────┘
                       │                                              │
┌──────────────────────▼──────┐                        ┌──────────────▼────────────────────────────────┐
│  Generative AI (MedGemma)   │                        │      Dual-Model Triage Safety System          │
│ ─────────────────────────── │                        │ ───────────────────────────────────────────── │
│ • Untrusted prompt defense  │                        │ 1. Deterministic Rule Triage Baseline (Fixed) │
│ • Constrained extraction    │                        │    • P0 (Emergency Escalation)                │
│ • Adaptive single questions │                        │    • P1 (Urgent Review) / P2 / P3 / ERR       │
│ • Grounded clinical summary │                        │    • Trilingual negation guard                │
│ • Verbatim quote citations  │                        │ 2. LightGBM Tabular Classifier (Secondary)    │
│ • Single-retry JSON repair  │                        │    • 41-feature schema (vitals + SLM enums)   │
│ • Cross-lingual reasoning   │                        │    • High-concordance probability scores      │
└─────────────────────────────┘                        └───────────────────────────────────────────────┘
```

### Component Responsibility Breakdown

| Component | Technology | Responsibility |
| :--- | :--- | :--- |
| **Generative AI (SLM)** | Google MedGemma 4B Instruction-Tuned (`google/medgemma-1.5-4b-it`) | • Structured clinical extraction across English, Hindi, and Hinglish<br/>• Constrained vocabulary extraction (`primary_system`, `acuity_red_flags`, `comorbidities_identified`)<br/>• Adaptive single follow-up question generation without repetitive loops<br/>• Factual, evidence-grounded clinical summaries with verbatim citations |
| **Deterministic Rule Triage** | Rule-based engine (`backend/services/triage_rules.py`) | • Transparent priority categorization (`P0`, `P1`, `P2`, `P3`, `ERR`)<br/>• Primary clinical safety baseline; evaluated first and escalation cannot be downgraded<br/>• Trilingual negation guard (`_is_negated`) prevents false alerts on denials<br/>• Fail-safe fallback: model errors trigger `ERR: Review required — AI unavailable` |
| **LightGBM ML Classifier** | LightGBM Tabular Model (`models/lightgbm_triage.pkl`) | • Secondary predictive signal trained on 41-column clinical dataset<br/>• Ingests continuous vitals, derived metrics (`shock_index`, `map_bp`), and SLM categorical enums<br/>• Outputs `ml_model_priority` and `ml_confidence` (0.0–1.0) without touching rule baseline |
| **Voice Transcription** | `faster-whisper` (`base` / `tiny` on CPU) & Mock ASR | • Speech-to-text with acoustic confidence and language detection<br/>• Patient-in-the-loop review card: editable transcript before submitting<br/>• Raw transcript provenance preserved for clinician inspection |
| **Medical Report OCR** | Windows Native `winocr`, `PaddleOCR`, & `pypdf` | • Ingestion of prescriptions, lab tests, and discharge summaries<br/>• Multi-tier fallback (native WinRT OCR -> PaddleOCR -> text extraction)<br/>• Non-destructive preview before committing to case record |
| **Human Reviewer** | Clinician Dashboard (`frontend/pages/reviewer.py`) | • Edit and approve final clinical summaries<br/>• Side-by-side Dual Triage Signal Comparison (`Rule-Based` vs `LightGBM ML`)<br/>• Priority overrides (requires mandatory recorded justification)<br/>• Atomic evidence inspection with source modality provenance (`🎤 Voice`, `📄 OCR`, `💬 Typed`)<br/>• Audit trail verification and case mark done/deletion |

---

## 2. Model Execution Modes & Hardware Guidance

CareFlow provides a configurable model adapter (`backend/services/medgemma.py`) with three execution modes:

### A. `mock` Mode (Default for Development & CI)
- **Zero external dependencies, zero GPU required**.
- Deterministic, high-fidelity clinical parser extracting symptoms, onset, severity, denials, and contradictions across English, Hindi, and Hinglish.
- Executes in `< 5 ms`, enabling instant testing on any standard laptop or automated CI pipeline without downloading multi-gigabyte weights.

### B. `hosted` Mode (Recommended for Cloud / Hackathons)
- Connects to an external endpoint serving `google/medgemma-1.5-4b-it` (e.g. Hugging Face Dedicated Inference Endpoint, vLLM, or Text Generation Inference).
- Reads `MEDGEMMA_ENDPOINT` and `MEDGEMMA_API_KEY` (or `HF_TOKEN`) from environment variables.
- Zero local RAM overhead.

### C. `local` Mode (Local Hugging Face Transformers)
- Loads `google/medgemma-1.5-4b-it` locally via `transformers` and `torch`.
- **Memory Requirements:**
  - 16-bit Float: **~8 GB to 10 GB VRAM / RAM**.
  - 4-bit Quantization (`bitsandbytes`): **~3.5 GB to 4.5 GB RAM**.
  - **16 GB RAM Laptop with No Dedicated GPU:** If loading locally on CPU, 4-bit quantization or CPU offloading can be used (`MEDGEMMA_LOCAL_QUANTIZATION=4bit`), but inference latency on CPU may range from 15 to 45 seconds per response. For real-time testing, `mock` or `hosted` mode is strongly recommended.

---

## 3. Operational Triage Protocol & Safety Safeguards

Triage categories are evaluated deterministically against explicit clinical evidence:

| Priority | Label | Clinical Meaning & Triggers | Operational Protocol |
| :---: | :--- | :--- | :--- |
| **P0** | 🚨 Emergency Escalation | Chest pain radiating to arm/jaw, acute severe dyspnea, acute stroke signs (FAST), severe active hematemesis, anaphylaxis. | Immediate emergency escalation / ED transfer. Checked first; cannot be downgraded. |
| **P1** | ⚠️ Urgent Review | High fever (>39°C) with stiff neck, severe uncontrolled pain (>= 8/10), acute central chest pain without radiation, acute abdomen. | Prompt clinician evaluation (within 1–2 hours). |
| **P2** | ❓ Incomplete / Uncertain | Contradictory statements detected across turns, clinical ambiguity, or multiple unknown/skipped dimensions. | Clinician callback or targeted re-interview to clarify discrepancies. |
| **P3** | 📋 Routine Review | Stable, mild-to-moderate symptoms with no red flags or escalation triggers. | Standard intake queue for scheduled outpatient review. |
| **ERR** | ⚠️ Review Required — AI Unavailable | Model failure, schema parsing failure after retry, or missing model credentials. | Manual clinician triage; system never silently assigns a routine priority. |

### Core Safety Safeguards:
1. **Trilingual Negation Guard (`_is_negated`):** Inspects prefix ("no", "without", "denies", "नहीं", "बिना", "nahi") and postfix ("nahi hai", "नहीं है", "absent") tokens across clause boundaries to prevent false positive emergency triggers when a patient explicitly denies a symptom.
2. **Escalation Preservation:** Once an interim P0 or P1 acuity is established, subsequent follow-up answers cannot silently erase or downgrade the emergency alert upon final case submission.
3. **Denial Integrity:** Explicit negative findings are strictly derived from confirmed patient statements; fabricated cardiac or medication denials are strictly prevented.

---

## 4. Quick Start & Setup Guide

### Prerequisites
- Python 3.10+
- Virtual Environment tool (`venv`)

### 1. Clone & Set Up Virtual Environment

```bash
cd "c:\Users\DELL\Documents\Inter IIT\GR"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Environment Variables

CareFlow includes a `.env.example` template:

```bash
cp .env.example .env
```

Default configuration (`mock` mode):
```ini
MEDGEMMA_MODE=mock
MEDGEMMA_MODEL_ID=google/medgemma-1.5-4b-it
MEDGEMMA_ENDPOINT=
MEDGEMMA_API_KEY=
ASR_MODEL_SIZE=base
ASR_DEVICE=cpu
OCR_ENGINE=auto
DATABASE_URL=sqlite:///./careflow.db
MAX_FOLLOWUP_QUESTIONS=8
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
BACKEND_URL=http://127.0.0.1:8000
STREAMLIT_SERVER_PORT=8501
```

### 3. Train the LightGBM Urgency Classifier (Optional / Artifact Verification)

CareFlow comes pre-trained with `models/lightgbm_triage.pkl`. To re-train or reproduce the model from `careflow_lightgbm_training_dataset.csv`:

```bash
python scripts/train_lightgbm.py
```
- Maps `gender` to binary (`M`: 0.0, `F`: 1.0).
- Fits `LGBMClassifier(objective='multiclass', num_class=4, n_estimators=100, learning_rate=0.05, max_depth=5, class_weight='balanced', random_state=42)` across 41 features.
- Saves the trained model to `models/lightgbm_triage.pkl`.

### 4. Launch the Backend API

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Health Check: `http://127.0.0.1:8000/health`
- Interactive OpenAPI / Swagger Docs: `http://127.0.0.1:8000/docs`

### 5. Launch the Streamlit Frontend

In a separate terminal (with `.venv` activated):

```bash
streamlit run frontend/app.py --server.port 8501
```
- Patient Intake Portal: `http://localhost:8501`
- Clinician Reviewer Dashboard: `http://localhost:8501/reviewer`
- Benchmark & Policy Evaluation: `http://localhost:8501/evaluation`

---

## 5. Running Automated Tests

CareFlow features a comprehensive automated `pytest` test suite covering **100+ verified test cases across 12 test modules** (passing in ~6 seconds with zero regressions):

```bash
pytest -v
```

### Verified Test Modules:
1. **LightGBM ML Classifier & Dual Triage Guardrail (`tests/test_ml_triage.py`):**
   - Artifact loading, graceful fallback to `P2` (`model_artifact_missing`) when missing.
   - Derived clinical metrics calculation (`shock_index`, `map_bp`, Celsius-to-Fahrenheit conversion, missing vitals defaults).
   - Robust vocabulary filtering guardrails (prevents unmapped SLM hallucinations in `primary_system`, `rf_*`, and `hx_*`).
   - Dual-model safety guardrail (deterministic rule triage baseline preserved with zero interference from ML).
   - FastAPI submit response schema validation (`rule_based_priority`, `ml_model_priority`, `ml_confidence`, `extracted_features`, `suggested_follow_up`).
2. **HTTP 500 Regressions, Voice Integration & SQLite Migrations (`tests/test_intake_500_and_voice.py`):**
   - Dynamic SQLite schema migration (`messages.source_type`, `raw_transcript`, `evidence.confidence`, `cases.ml_model_triage`, `cases.ml_confidence`).
   - Empty/whitespace message 400 validation and clean rollback on model extraction errors.
   - Multilingual message ingestion (English, Devanagari Hindi, Hinglish).
   - Audio transcription validation, language normalization, and patient confirmation workflow.
   - End-to-end multimodal patient & reviewer journey.
3. **Safety Defect Regressions (`tests/test_safety_fixes.py`):**
   - Fabricated cardiac denials strictly prevented.
   - Interim P1 urgency persistence (interim fever/severe pain cannot be erased by follow-up answers).
   - Document upload actor attribution (`ai_system`, never `reviewer`).
4. **Multilingual Processing (`tests/test_multilingual.py`):**
   - Devanagari Hindi and Hinglish symptom extraction.
   - Cross-lingual duration, progression, and severity understanding.
   - Cross-turn timeline contradiction detection across English and Hindi.
5. **Voice Audio Transcription (`tests/test_transcription.py`):**
   - `/transcribe-audio` endpoint validation, size limits (25 MB), empty file rejection.
   - Secure temporary file cleanup with zero leaked disk files.
6. **Medical Report OCR (`tests/test_ocr_service.py`):**
   - Multi-tier OCR extraction (`winocr`, `PaddleOCR`, `pypdf`, plain text).
   - Non-destructive document extraction preview (`/cases/{id}/extract-document`).
7. **Multi-Policy Questioning Benchmark (`tests/test_eval_policies.py`):**
   - Verification of Fixed, MedGemma-only, State-aware, and CareFlow-BQ policies under budgets 4, 6, 8.
8. **Clinical Voice Triage Regressions (`tests/test_clinical_voice_triage_regression.py`):**
   - Verifies raw transcript provenance and triage escalation preservation during voice input.
9. **Core Protocols & Persistence (`tests/test_api.py`, `tests/test_database.py`, `tests/test_evidence_state.py`, `tests/test_extraction.py`, `tests/test_triage_rules.py`):
   - Deterministic P0 emergency precedence, P1 urgent triage, P2 conflict handling, P3 routine review.
   - Fail-safe fallback to `ERR: Review required — AI unavailable` on model failure.

---

## 6. Running Research Benchmarks & Policy Evaluation

CareFlow provides two repeatable evaluation benchmarks:

### Suite A: Standardized Clinical Pipeline Fixtures (7 Tests)
```bash
python -m evaluation.run_eval
```
- Tests 7 standardized scenarios: Acute Coronary P0, Febrile Meningeal P1, Severe Flank Pain P1, Contradictory Duration P2, Unknown Dimension P3, Routine Cold P3, and AI Failure Fallback ERR.
- Logs traces to `evaluation_results/eval_<timestamp>.jsonl` and summary to `evaluation_results/eval_<timestamp>_summary.json`.

### Suite B: Multi-Policy Questioning Benchmark (Budgets 4, 6, 8)
```bash
python -m evaluation.eval_policies
```
- Empirically benchmarks 4 question-selection policies across question budgets (4, 6, 8):
  1. **Fixed:** Scripted clinical sequence.
  2. **MedGemma-only:** Zero-shot conversational generation from history.
  3. **State-aware:** Deterministic missing-slot fulfillment.
  4. **CareFlow-BQ:** Budgeted questioning prioritizing high-acuity life-threats.
- Evaluates: Extraction quality, critical question recall %, required info coverage %, redundancy rate %, missed escalation triggers, and turn latency.

Both benchmarks can also be executed and inspected visually via the Streamlit interface at **`http://localhost:8501/evaluation`**.

---

## 7. API Reference Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Reports backend status, database status, and MedGemma readiness |
| `POST` | `/cases` | Creates a new patient intake with a random non-identifying Case ID |
| `GET` | `/cases` | Lists cases sorted by priority (`P0` -> `P1` -> `P2` -> `P3` -> `ERR`) and wait time |
| `GET` | `/cases/{id}` | Fetches full case details, messages, evidence items, and audit logs |
| `POST` | `/cases/{id}/messages` | Appends patient message, updates evidence state, runs interim red-flag check (supports `source_type`, `raw_transcript`, `source_language`, `transcription_confidence`, `is_confirmed`) |
| `POST` | `/cases/{id}/next-question` | Generates single next clarifying question via MedGemma or symptom catalog (max 8 turns) |
| `POST` | `/transcribe-audio` | Transcribes audio without auto-submitting; returns transcript, language, confidence, and model size |
| `POST` | `/cases/{id}/extract-document` | Previews OCR text extraction with confidence score without committing to case transcript |
| `POST` | `/cases/{id}/upload-document` | Uploads and extracts text from medical image/PDF directly into case with audit logging |
| `POST` | `/cases/{id}/submit` | Finalizes intake, produces clinical summary, applies deterministic triage rules with escalation preservation |
| `POST` | `/cases/{id}/generate-summary` | Regenerates clinical summary draft |
| `POST` | `/cases/{id}/review` | Saves reviewer edits, clinical notes, and approval status |
| `POST` | `/cases/{id}/override-priority` | Overrides priority with mandatory reason (audit logged) |
| `POST` | `/cases/{id}/reopen` | Reopens an approved case for further clarification |

---

## 8. Database Schema Overview

CareFlow uses SQLite with SQLAlchemy models, foreign key enforcement, and automatic schema migrations:

- **`cases`**: `id` (PK, e.g. `CF-2026-XXXX`), `created_at`, `updated_at`, `preferred_language`, `chief_complaint`, `evidence_state_json`, `ai_summary_json`, `ai_status`, `priority_category`, `priority_reasons_json`, `missing_information_json`, `uncertainties_json`, `triage_rule_version`, `review_status`, `reviewer_summary`, `reviewer_priority`, `reviewer_notes`, `reviewed_at`.
- **`messages`**: `id` (PK), `case_id` (FK), `role` (`patient` / `assistant`), `content`, `raw_transcript` (unedited ASR transcript), `source_type` (`patient_typed`, `voice_transcript`, `ocr_extraction`), `source_language`, `transcription_confidence`, `is_confirmed`, `created_at`.
- **`evidence`**: `id` (PK), `case_id` (FK), `field`, `value_json`, `status` (`reported`, `unknown`, `conflicting`, `uncertain`), `source_message_id` (FK), `source_quote`, `source_type`, `confidence`, `created_at`.
- **`audit_events`**: `id` (PK), `case_id` (FK), `event_type`, `old_value_json`, `new_value_json`, `reason`, `actor_role` (`patient`, `ai_system`, `triage_engine`, `reviewer`), `created_at`.

---

## 9. Local Windows 11 Run Guide

For complete, copy-paste Windows 11 setup commands (including PowerShell execution policies, D: drive Hugging Face cache redirection, and local audio setup), see [LOCAL_WINDOWS_TEST.md](LOCAL_WINDOWS_TEST.md).
