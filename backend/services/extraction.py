import logging
from typing import List, Dict, Any, Optional
from backend.services.medgemma import medgemma_service, ModelInferenceError
from backend.schemas.clinical import ClinicalExtractionOutput

logger = logging.getLogger("careflow.extraction")

EXTRACTION_SYSTEM_PROMPT = """You are an expert clinical intake parser assisting in a structured, non-diagnostic triage assistant.
TASK: CLINICAL_EXTRACTION

Your task is to analyze the patient utterances (in English, Hindi, or bilingual Hinglish) and extract clinical facts strictly grounded in the text.
CRITICAL SAFETY & INTEGRITY RULES:
1. Extract ONLY facts explicitly stated by the patient or medical documents. Do NOT fabricate, infer, or hallucinate findings, vitals, normal exams, or diagnoses.
2. MULTILINGUAL SUPPORT: Seamlessly understand English, Hindi (हिन्दी in Devanagari script), and Hinglish (Hindi in Roman script). Standardize clinical field names into English while preserving verbatim quotes in original language.
3. CLINICAL URGENCY REASONING: Think carefully about clinical acuity and physiological risk. Identify any red flags or urgency signals under 'urgency_signals', classifying each with severity_level ('emergency', 'urgent', 'moderate', or 'routine'), explicit clinical rationale, and verbatim quote.
4. If an answer is skipped, unknown, or the patient says 'I do not know' / 'पता नहीं', record it under 'unknown_information'. Do NOT assume it is a negative finding.
5. If the patient explicitly denies a symptom (e.g. 'no chest pain', 'सीने में दर्द नहीं है'), record it under 'explicit_negative_findings'.
6. If the patient makes clashing or conflicting claims across messages, record them under 'contradictions'.
7. If the patient uses vague language (e.g. 'maybe', 'शायद', 'not sure'), record it under 'uncertainties'.
8. Return your output strictly as a JSON object matching this schema:
{
  "symptoms": ["list of strings"],
  "onset_and_duration": "string or 'Not reported'",
  "progression": "string or 'Not reported'",
  "severity": "string or 'Not reported'",
  "associated_symptoms": ["list of strings"],
  "medical_history": ["list of strings"],
  "medications_and_allergies": ["list of strings"],
  "explicit_negative_findings": ["list of strings"],
  "unknown_information": ["list of strings"],
  "contradictions": ["list of strings"],
  "uncertainties": ["list of strings"],
  "evidence_items": [
    {
      "field": "canonical_field_name",
      "value": "extracted value",
      "status": "reported" | "unknown" | "conflicting" | "uncertain",
      "source_type": "patient_message",
      "source_quote": "exact verbatim quote from patient"
    }
  ],
  "urgency_signals": [
    {
      "signal": "name of red flag or urgency marker",
      "severity_level": "emergency" | "urgent" | "moderate" | "routine",
      "rationale": "clinical reasoning explaining why this is an emergency or urgent marker",
      "source_quote": "verbatim quote from patient text"
    }
  ],
  "is_sufficient_for_summary": false
}
"""

class ClinicalExtractionService:
    """Service orchestrating clinical extraction using MedGemma."""

    @staticmethod
    def extract_clinical_facts(patient_messages: List[str]) -> ClinicalExtractionOutput:
        """
        Runs clinical fact extraction across patient message utterances.
        """
        # Format patient utterances safely within XML tags to defend against prompt injection
        user_content_blocks = [
            f"<patient_utterance>{msg.strip()}</patient_utterance>"
            for msg in patient_messages if msg.strip()
        ]
        user_prompt = (
            "Analyze the following patient message history and extract all clinical findings according to instructions:\n\n"
            + "\n".join(user_content_blocks)
        )

        try:
            raw_json = medgemma_service.generate_json(
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_retries=1
            )
            # Validate through Pydantic schema
            return ClinicalExtractionOutput.model_validate(raw_json)
        except Exception as e:
            logger.error(f"Failed to extract clinical facts via MedGemma: {e}")
            raise ModelInferenceError(f"Clinical fact extraction failed: {e}")

clinical_extraction_service = ClinicalExtractionService()
