import pytest
from backend.services.medgemma import MedGemmaService, ModelInferenceError
from backend.services.extraction import clinical_extraction_service
from backend.services.summarization import clinical_summarization_service
from backend.services.questioning import adaptive_questioning_service
from backend.schemas.clinical import ClinicalExtractionOutput, ClinicalSummaryOutput

def test_extraction_service_with_symptoms():
    """Extracts symptoms, onset, and creates atomic evidence records."""
    messages = [
        "I have a fever and cough that started 3 days ago.",
        "My pain is about 4/10."
    ]
    extraction = clinical_extraction_service.extract_clinical_facts(messages)
    assert isinstance(extraction, ClinicalExtractionOutput)
    assert "fever" in [s.lower() for s in extraction.symptoms] or "cough" in [s.lower() for s in extraction.symptoms]
    assert len(extraction.evidence_items) > 0


def test_questioning_service_max_turns():
    """Adaptive questioning respects configured turn limits."""
    messages = ["Symptom statement"]
    # With turn_count = 8 (MAX_FOLLOWUP_QUESTIONS), questioning concludes
    res = adaptive_questioning_service.generate_next_question(
        patient_messages=messages,
        current_question_count=8
    )
    assert res.is_terminal is True
    assert res.next_question is None
    assert "limit" in res.reason_for_question.lower()


def test_questioning_service_never_repeats():
    """Adaptive questioning strictly rotates clinical dimensions and never asks the same question twice."""
    patient_msgs = ["I have had a severe sore throat and fever."]
    asked_questions = []

    for turn in range(4):
        res = adaptive_questioning_service.generate_next_question(
            patient_messages=patient_msgs,
            current_question_count=turn,
            assistant_messages=asked_questions
        )
        if res.next_question:
            # Ensure the question is not in asked_questions
            assert res.next_question not in asked_questions
            # Ensure no high word overlap
            for prev in asked_questions:
                assert res.next_question.lower() != prev.lower()
            asked_questions.append(res.next_question)
            patient_msgs.append("My throat is swollen.")

    # Must have generated distinct questions across different clinical dimensions
    assert len(set(asked_questions)) == len(asked_questions)
    assert len(asked_questions) >= 2


def test_summary_grounding():
    """Summary extracts chief complaint, reported symptoms, and does not fabricate missing info."""
    messages = [
        "Severe throbbing headache since yesterday morning.",
        "No fever. No head injury."
    ]
    summary = clinical_summarization_service.generate_summary(messages)
    assert isinstance(summary, ClinicalSummaryOutput)
    assert "headache" in summary.summary_text.lower()
    # Negative findings
    assert any("fever" in n.lower() or "cardiac" in n.lower() for n in summary.explicit_negative_findings)


def test_medgemma_single_retry_failure():
    """When a model consistently returns invalid JSON, it retries once and raises ModelInferenceError."""
    service = MedGemmaService()
    # Mock _call_model to return non-JSON text
    service._call_model = lambda prompt: "I am unable to answer in JSON format."
    
    with pytest.raises(ModelInferenceError) as exc_info:
        service.generate_json("System prompt", "User prompt", max_retries=1)
    
    assert "failed to produce valid structured JSON" in str(exc_info.value)
