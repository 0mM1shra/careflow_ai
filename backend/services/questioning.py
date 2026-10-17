import logging
from typing import List, Dict, Any, Optional
from backend.services.medgemma import medgemma_service, ModelInferenceError
from backend.schemas.clinical import FollowUpQuestionOutput
from backend.config import settings

logger = logging.getLogger("careflow.questioning")

QUESTION_SYSTEM_PROMPT = """You are an empathetic, clinical triage intake assistant.
TASK: FOLLOW_UP_QUESTION

Your goal is to formulate ONE single clarifying follow-up question to collect clinically critical missing information.
RULES:
1. Ask ONLY ONE focused question at a time.
2. Prioritize screening for critical red-flag symptoms (e.g. pain radiation, shortness of breath, neck stiffness) and essential dimensions (onset, duration, severity scale, medications/allergies).
3. Do NOT tell the patient they have any disease, do NOT give diagnostic labels, and do NOT tell them they are safe.
4. Do NOT assume skipped or unknown answers are negative or absent.
5. ANTI-REPETITION MANDATE: NEVER repeat any previously asked question or re-ask a topic that has already been asked. If the patient already answered, skipped, or said 'I don't know', that topic is considered addressed. Always move to an unasked dimension.
6. If enough information has been gathered, or if further questioning is unlikely to add value, set "next_question": null and "is_terminal": true.
7. Return your output strictly as a JSON object matching this schema:
{
  "next_question": "the single question to ask or null",
  "reason_for_question": "clinical justification",
  "target_field": "clinical attribute name",
  "is_terminal": false
}
"""

class AdaptiveQuestioningService:
    """Service generating the single next clinically relevant question without repetition."""

    # Fallback rotation of clinical dimensions if model repeats
    CLINICAL_DIMENSION_QUESTIONS = [
        ("pain_radiation", "Does the discomfort radiate or spread to any other area, such as your neck, jaw, back, or arm?"),
        ("associated_symptoms", "Are you experiencing any other associated symptoms, such as shortness of breath, dizziness, sweating, or nausea?"),
        ("onset_and_duration", "When did your symptoms first begin, and have they been constant, worsening, or coming and going?"),
        ("severity", "On a scale from 0 to 10 (where 0 is no discomfort and 10 is unbearable), how severe is your symptom right now?"),
        ("aggravating_relieving", "Does anything in particular make your symptoms better or worse (such as rest, position changes, or medication)?"),
        ("medications_and_history", "Are you currently taking any prescription medications, or do you have any relevant medical conditions or allergies?")
    ]

    @classmethod
    def generate_next_question(
        cls,
        patient_messages: List[str],
        current_question_count: int,
        assistant_messages: Optional[List[str]] = None,
        evidence_state: Optional[Dict[str, Any]] = None,
        preferred_language: str = "English"
    ) -> FollowUpQuestionOutput:
        """
        Determines the next follow-up question, guaranteeing no repetitive questions.
        """
        assistant_messages = assistant_messages or []

        # Enforce maximum question limit or early completion
        if current_question_count >= settings.MAX_FOLLOWUP_QUESTIONS:
            logger.info(f"Reached maximum question limit ({settings.MAX_FOLLOWUP_QUESTIONS}). Ending questioning.")
            return FollowUpQuestionOutput(
                next_question=None,
                reason_for_question=f"Reached maximum limit of {settings.MAX_FOLLOWUP_QUESTIONS} follow-up questions. Intake ready for human clinician review.",
                target_field="limit_reached",
                is_terminal=True
            )

        # Build prompt with safe utterance enclosure
        user_content_blocks = [
            f"<patient_utterance>{msg.strip()}</patient_utterance>"
            for msg in patient_messages if msg.strip()
        ]
        
        # Include previously asked questions
        prev_q_blocks = [
            f"<previously_asked_question>{q.strip()}</previously_asked_question>"
            for q in assistant_messages if q.strip()
        ]

        user_prompt = (
            f"The patient has provided the following messages (turn count: {current_question_count}/{settings.MAX_FOLLOWUP_QUESTIONS}, preferred language: {preferred_language}):\n\n"
            + "\n".join(user_content_blocks)
            + "\n\n"
            + (("PREVIOUSLY ASKED QUESTIONS (DO NOT REPEAT ANY OF THESE):\n" + "\n".join(prev_q_blocks) + "\n\n") if prev_q_blocks else "")
            + "Formulate the single most clinically relevant follow-up question to clarify an unaddressed critical dimension. "
            + "If all key areas have been covered, set 'next_question': null and 'is_terminal': true."
        )

        try:
            raw_json = medgemma_service.generate_json(
                system_prompt=QUESTION_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_retries=1
            )
            result = FollowUpQuestionOutput.model_validate(raw_json)

            # Deduplication Verification: Check if generated question repeats any previous question
            if result.next_question and cls._is_duplicate_question(result.next_question, assistant_messages):
                logger.warning(f"Detected duplicate question '{result.next_question}'. Finding unaddressed alternative.")
                result = cls._get_unasked_fallback_question(assistant_messages, patient_messages)

            return result

        except Exception as e:
            logger.error(f"Failed to generate follow-up question: {e}")
            # Fallback gracefully to an unasked dimension or terminate
            return cls._get_unasked_fallback_question(assistant_messages, patient_messages)

    @classmethod
    def _is_duplicate_question(cls, question: str, previous_questions: List[str]) -> bool:
        """Checks if a question has essentially already been asked."""
        q_norm = question.lower().strip()
        for prev in previous_questions:
            p_norm = prev.lower().strip()
            # Exact or substring match
            if q_norm == p_norm or q_norm in p_norm or p_norm in q_norm:
                return True
            # Check key semantic word overlap
            q_keywords = set(w for w in q_norm.split() if len(w) > 4)
            p_keywords = set(w for w in p_norm.split() if len(w) > 4)
            if q_keywords and p_keywords:
                overlap = len(q_keywords.intersection(p_keywords)) / max(len(q_keywords), len(p_keywords))
                if overlap > 0.6:
                    return True
        return False

    @classmethod
    def _get_unasked_fallback_question(cls, previous_questions: List[str], patient_messages: List[str]) -> FollowUpQuestionOutput:
        """Selects the next unaddressed clinical dimension."""
        full_patient_text = " ".join(patient_messages).lower()

        for target_field, question_text in cls.CLINICAL_DIMENSION_QUESTIONS:
            # Check if this topic was already asked
            if any(cls._is_duplicate_question(question_text, previous_questions) for _ in [1]):
                continue

            # Check if already answered in patient messages
            if target_field == "onset_and_duration" and any(w in full_patient_text for w in ["yesterday", "ago", "started", "since", "days", "hours"]):
                continue
            if target_field == "severity" and any(w in full_patient_text for w in ["/10", "out of 10", "mild", "severe", "moderate"]):
                continue

            return FollowUpQuestionOutput(
                next_question=question_text,
                reason_for_question=f"Clarifying clinical dimension: {target_field}",
                target_field=target_field,
                is_terminal=False
            )

        # If all dimensions have been asked, conclude questioning
        return FollowUpQuestionOutput(
            next_question=None,
            reason_for_question="All primary clinical intake dimensions have been gathered.",
            target_field="intake_complete",
            is_terminal=True
        )

adaptive_questioning_service = AdaptiveQuestioningService()
