# CareFlow: MedGemma-Powered Clinical Intake and Reviewer Dashboard

**CareFlow** is an end-to-end, text-only clinical intake and triage review assistant built for research and hackathons. It leverages **Google MedGemma** for conversational clinical extraction and grounded summarization, paired with a **deterministic, auditable rule-based triage engine** and a **clinician-in-the-loop review dashboard**.

> **⚠️ RESEARCH PROTOTYPE DISCLAIMER**
> CareFlow is an exploratory clinical decision support and intake assistant prototype. It is **NOT** a certified medical diagnostic device or clinically validated emergency triage system. It does not diagnose medical conditions or provide treatment instructions. All clinical summaries and priority classifications must be verified, edited, and approved by a qualified healthcare professional. In a life-threatening medical emergency, call 911 (or local emergency services) or visit the nearest emergency department immediately.

---

## 1. System Architecture & Component Separation

CareFlow strictly isolates generative AI capabilities from safety-critical triage decisions:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             Streamlit Frontend                              │
│   ├── Patient Interface (/patient): Conversational intake & review          │
│   ├── Reviewer Dashboard (/reviewer): Queue, citations & approval controls  │
│   └── Evaluation Suite (/evaluation): Live benchmark verification           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ HTTP Requests
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                              FastAPI Backend                                │
│   ├── /cases, /messages, /next-question, /submit, /review, /override        │
│   └── Database: SQLite (SQLAlchemy ORM with Foreign Key Enforcement)        │
└──────────────┬──────────────────────────────────────────────┬───────────────┘
               │                                              │
┌──────────────▼──────────────┐                ┌──────────────▼──────────────┐
│  Generative AI (MedGemma)   │                │ Deterministic Triage Engine │
│ ─────────────────────────── │                │ ─────────────────────────── │
│ • Untrusted prompt defense  │                │ • Separate from MedGemma    │
│ • Clinical fact extraction  │                │ • P0 (Emergency Escalation) │
│ • Adaptive single questions │                │ • P1 (Urgent Review)        │
│ • Grounded clinical summary │                │ • P2 (Uncertain/Conflict)   │
│ • Verbatim quote citations  │                │ • P3 (Routine Review)       │
│ • Single-retry JSON repair  │                │ • ERR (AI Failure Fallback) │
└─────────────────────────────┘                └─────────────────────────────┘
```

### Responsibility Breakdown

| Component | Technology | Responsibility |
| :--- | :--- | :--- |
| **Generative AI** | Google MedGemma 4B Instruction-Tuned (`google/medgemma-1.5-4b-it`) | • Structured clinical extraction<br/>• Adaptive single follow-up question generation<br/>• Factual, evidence-grounded clinical summaries<br/>• Verbatim patient quote citations |
| **Deterministic Triage** | Rule-based engine (`backend/services/triage_rules.py`) | • Transparent priority categorization (`P0`, `P1`, `P2`, `P3`, `ERR`)<br/>• Emergency criteria checked first; never downgraded<br/>• Fail-safe fallback: model errors trigger `ERR: Review required — AI unavailable`<br/>• Documented protocol version (`v1.0.0-prototype`) |
| **Human Reviewer** | Clinician Dashboard (`frontend/pages/reviewer.py`) | • Edit and approve final clinical summaries<br/>• Priority overrides (requires mandatory recorded justification)<br/>• Enter clinical referral notes & orders<br/>• Reopen cases if further patient clarification is required |

---

## 2. Model Execution Modes & Hardware Guidance

CareFlow provides a configurable model adapter (`backend/services/medgemma.py`) with three execution modes:

### A. `mock` Mode (Default for Development & CI)
- **Zero external dependencies, zero GPU required**.
- Deterministic, high-fidelity clinical parser extracting real symptoms, onset, severity, denials, and contradictions directly from patient text.
- Executes in `< 5 ms`, enabling instant testing on any standard laptop or automated CI pipeline without downloading multi-gigabyte weights.

### B. `hosted` Mode (Recommended for Hackathons / Cloud)
- Connects to an external endpoint serving `google/medgemma-1.5-4b-it` (e.g. Hugging Face Dedicated Inference Endpoint, vLLM, or Text Generation Inference).
- Reads `MEDGEMMA_ENDPOINT` and `MEDGEMMA_API_KEY` from environment variables.
- Zero local RAM overhead.

### C. `local` Mode (Local Hugging Face Transformers)
- Loads `google/medgemma-1.5-4b-it` locally via `transformers` and `torch`.
- **Memory Requirements:**
  - 16-bit Float: **~8 GB to 10 GB VRAM / RAM**.
  - 4-bit Quantization (`bitsandbytes`): **~3.5 GB to 4.5 GB RAM**.
  - **16 GB RAM Laptop with No Dedicated GPU:** If loading locally on CPU, 4-bit quantization or CPU offloading can be used (`MEDGEMMA_LOCAL_QUANTIZATION=4bit`), but inference latency on CPU may range from 15 to 45 seconds per response. For real-time testing, `mock` or `hosted` mode is strongly recommended.

---

## 3. Operational Triage Protocol

Triage categories are evaluated deterministically against explicit evidence:

| Priority | Label | Clinical Meaning & Triggers | Operational Protocol |
| :---: | :--- | :--- | :--- |
| **P0** | 🚨 Emergency Escalation | Chest pain radiating to arm/jaw, severe acute dyspnea, sudden stroke signs (FAST), severe hemorrhage, anaphylaxis. | Immediate emergency escalation / ED transfer. Checked first; cannot be downgraded. |
| **P1** | ⚠️ Urgent Review | High fever (>39°C) with stiff neck, severe uncontrolled pain (>= 8/10), acute abdomen with intractable vomiting. | Prompt clinician evaluation (within 1–2 hours). |
| **P2** | ❓ Incomplete / Uncertain | Contradictory statements detected across conversation turns, or high clinical ambiguity. | Clinician callback or targeted re-interview to clarify discrepancies. |
| **P3** | 📋 Routine Review | Stable, mild-to-moderate symptoms (e.g., uncomplicated upper respiratory symptoms) with no escalation triggers. | Standard intake queue for scheduled outpatient review. |
| **ERR** | ⚠️ Review Required — AI Unavailable | Model failure, schema parsing failure after retry, or missing model credentials. | Manual clinician triage; system never silently assigns a routine priority. |

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
DATABASE_URL=sqlite:///./careflow.db
MAX_FOLLOWUP_QUESTIONS=8
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
BACKEND_URL=http://127.0.0.1:8000
STREAMLIT_SERVER_PORT=8501
```

To switch to a hosted MedGemma endpoint:
```ini
MEDGEMMA_MODE=hosted
MEDGEMMA_ENDPOINT=https://your-endpoint.endpoints.huggingface.cloud
MEDGEMMA_API_KEY=your_hf_token_here
```

### 3. Launch the Backend API

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Health Check: `http://127.0.0.1:8000/health`
- Interactive OpenAPI / Swagger Docs: `http://127.0.0.1:8000/docs`

### 4. Launch the Streamlit Frontend

In a separate terminal (with `.venv` activated):

```bash
streamlit run frontend/app.py --server.port 8501
```
- Open in browser: `http://localhost:8501`

---

## 5. Running Automated Tests

CareFlow features an automated `pytest` test suite covering 18 test cases across all critical modules:

```bash
pytest -v
```

### Verified Test Areas
1. **Database & ORM Integrity:** Table relationships, cascade deletion, foreign key pragmas in SQLite.
2. **Evidence-State Tracking:** Atomic fact extraction, unknown answer preservation (`status="unknown"`, not converted to negative), and conflict detection retaining both clashing records.
3. **Deterministic Triage Rules:** Emergency precedence (P0 never downgraded), P1 urgent indicators, P2 conflict/uncertainty escalation, P3 routine classification.
4. **AI Failure Safety:** Ensuring that AI or schema failure automatically places cases into `ERR: Review required — AI unavailable` rather than assigning a routine priority.
5. **Adaptive Questioning Limits:** Enforcing the 8-turn question limit and stopping gracefully.
6. **API Endpoints & Workflows:** Full conversational intake lifecycle, early submissions, queue sorting and filtering, reviewer approvals, and priority overrides with mandatory reasons.

---

## 6. Running the Clinical AI Benchmark Evaluation

CareFlow includes an empirical evaluation pipeline measuring extraction accuracy, triage protocol consistency, conflict preservation, and latency across 7 standardized test fixtures:

### Run via Command Line

```bash
python -m evaluation.run_eval
```

Outputs:
- Live console summary table with latency and accuracy percentages.
- Benchmark reports saved to `evaluation_results/eval_<timestamp>.jsonl` and `eval_<timestamp>_summary.json`.

### Run via Streamlit UI
1. Navigate to `http://localhost:8501/evaluation`.
2. Click **▶️ Run Benchmark Evaluation Suite**.
3. Inspect live metrics cards, fixture-by-fixture comparison tables, and download JSONL reports.

---

## 7. API Reference Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Reports backend status, database status, and MedGemma readiness |
| `POST` | `/cases` | Creates a new patient intake with a random non-identifying Case ID |
| `GET` | `/cases` | Lists cases sorted by priority (`P0` -> `P1` -> `P2` -> `P3` -> `ERR`) and wait time |
| `GET` | `/cases/{id}` | Fetches full case details, messages, evidence items, and audit logs |
| `POST` | `/cases/{id}/messages` | Appends patient message, updates evidence state, runs interim red-flag check |
| `POST` | `/cases/{id}/next-question` | Generates single next clarifying question via MedGemma (max 8 turns) |
| `POST` | `/cases/{id}/submit` | Finalizes intake, produces clinical summary, applies deterministic triage rules |
| `POST` | `/cases/{id}/generate-summary` | Regenerates clinical summary draft |
| `POST` | `/cases/{id}/review` | Saves reviewer edits, clinical notes, and approval status |
| `POST` | `/cases/{id}/override-priority` | Overrides priority with mandatory reason (audit logged) |
| `POST` | `/cases/{id}/reopen` | Reopens an approved case for further clarification |

---

## 8. Database Schema Overview

CareFlow uses SQLite with SQLAlchemy models and foreign key enforcement:

- **`cases`**: `id` (PK, e.g. `CF-2026-XXXX`), `created_at`, `updated_at`, `preferred_language`, `chief_complaint`, `evidence_state_json`, `ai_summary_json`, `ai_status`, `priority_category`, `priority_reasons_json`, `missing_information_json`, `uncertainties_json`, `triage_rule_version`, `review_status`, `reviewer_summary`, `reviewer_priority`, `reviewer_notes`, `reviewed_at`.
- **`messages`**: `id` (PK), `case_id` (FK), `role` (`patient` / `assistant`), `content`, `created_at`.
- **`evidence`**: `id` (PK), `case_id` (FK), `field`, `value_json`, `status` (`reported`, `unknown`, `conflicting`, `uncertain`), `source_message_id` (FK), `source_quote`, `created_at`.
- **`audit_events`**: `id` (PK), `case_id` (FK), `event_type`, `old_value_json`, `new_value_json`, `reason`, `actor_role`, `created_at`.
