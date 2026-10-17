from backend.services.triage_rules import DeterministicTriageEngine, triage_engine

def test_p0_emergency_cardiac_precedence():
    """Emergency trigger (chest pain radiating to arm) takes precedence over all else."""
    messages = [
        "I am having crushing chest pain radiating to my left arm."
    ]
    evidence_state = {"facts": {"chest_pain": {"value": "crushing"}}}

    result = triage_engine.evaluate(evidence_state, messages, ai_status="ready")
    assert result.priority_category == "P0"
    assert "P0" in result.priority_label
    assert len(result.triggered_rules) >= 1
    assert result.triggered_rules[0].rule_id == "RULE-P0-CARD-01"
    assert "Acute Coronary Syndrome" in result.triggered_rules[0].rule_name
    assert result.triage_rule_version == DeterministicTriageEngine.PROTOCOL_VERSION


def test_p0_never_downgraded_by_absent_symptoms():
    """Even if patient denies other symptoms (e.g. no fever, normal breathing), P0 is never downgraded."""
    messages = [
        "I have severe chest pain radiating to my jaw. No fever. Breathing feels normal. No headache."
    ]
    evidence_state = {
        "explicit_negative_findings": ["denies fever", "denies shortness of breath"]
    }

    result = triage_engine.evaluate(evidence_state, messages, ai_status="ready")
    assert result.priority_category == "P0"
    assert result.triggered_rules[0].rule_id == "RULE-P0-CARD-01"


def test_p1_urgent_meningeal_signs():
    """Fever combined with stiff neck triggers P1 urgent review within 1 hour."""
    messages = [
        "I have a high fever of 103 and a really stiff neck, can't touch chin to chest."
    ]
    evidence_state = {"facts": {}}

    result = triage_engine.evaluate(evidence_state, messages, ai_status="ready")
    assert result.priority_category == "P1"
    assert any(t.rule_id == "RULE-P1-INF-01" for t in result.triggered_rules)


def test_p1_severe_pain_score():
    """Pain score >= 8/10 triggers P1 urgent review."""
    messages = [
        "My toothache is excruciating, it is 9/10 pain."
    ]
    evidence_state = {"facts": {}}

    result = triage_engine.evaluate(evidence_state, messages, ai_status="ready")
    assert result.priority_category == "P1"
    assert any(t.rule_id == "RULE-P1-PAIN-02" for t in result.triggered_rules)


def test_p2_incomplete_or_conflicting_information():
    """Contradictory findings escalate to P2: Incomplete or uncertain."""
    messages = [
        "I have had a mild rash for two days."
    ]
    evidence_state = {
        "contradictions": ["Conflict in onset: previously '2 days', now '2 months'"],
        "uncertainties": []
    }

    result = triage_engine.evaluate(evidence_state, messages, ai_status="ready")
    assert result.priority_category == "P2"
    assert any(t.rule_id == "RULE-P2-CONFLICT-01" for t in result.triggered_rules)


def test_p3_routine_review():
    """Mild symptoms with no red flags assign P3 routine review."""
    messages = [
        "I have a mild runny nose and sneezing for 3 days. No fever, pain is 1/10."
    ]
    evidence_state = {"facts": {}, "contradictions": [], "uncertainties": []}

    result = triage_engine.evaluate(evidence_state, messages, ai_status="ready")
    assert result.priority_category == "P3"
    assert "Routine" in result.priority_label


def test_ai_failure_never_produces_routine_priority():
    """If AI fails or model status is 'failed', triage MUST assign ERR, never P3 routine."""
    messages = [
        "Patient message text here"
    ]
    evidence_state = {}

    result = triage_engine.evaluate(evidence_state, messages, ai_status="failed")
    assert result.priority_category == "ERR"
    assert "Review Required — AI Unavailable" in result.priority_label
    assert result.triggered_rules[0].rule_id == "RULE-SYS-01"


def test_p0_emergency_hindi_cardiac():
    """Bilingual Hindi symptom (सीने में दर्द radiating to बांह) triggers P0 emergency triage."""
    messages = [
        "सीने में बहुत तेज दबाव और दर्द है, दर्द बाएं हाथ तक जा रहा है और सांस फूल रही है।"
    ]
    evidence_state = {"facts": {}}

    result = triage_engine.evaluate(evidence_state, messages, ai_status="ready")
    assert result.priority_category == "P0"
    assert any("RULE-P0" in t.rule_id for t in result.triggered_rules)


def test_generalized_model_urgency_signal_p0():
    """MedGemma-reasoned emergency urgency signal elevates unscripted symptoms to P0."""
    messages = [
        "My throat is closing up after a bee sting and I am gasping for air."
    ]
    evidence_state = {
        "urgency_signals": [
            {
                "signal": "anaphylactic_airway_compromise",
                "severity_level": "emergency",
                "rationale": "Rapidly progressing acute airway obstruction following hymenoptera sting",
                "source_quote": "throat is closing up after a bee sting and I am gasping for air"
            }
        ]
    }

    result = triage_engine.evaluate(evidence_state, messages, ai_status="ready")
    assert result.priority_category == "P0"
    assert any(t.rule_id == "RULE-P0-MODEL-01" for t in result.triggered_rules)


def test_generalized_model_urgency_signal_p1():
    """MedGemma-reasoned urgent signal elevates unscripted condition to P1."""
    messages = [
        "Severe pain started around my belly button and has moved to my right lower side."
    ]
    evidence_state = {
        "urgency_signals": [
            {
                "signal": "suspected_acute_appendicitis",
                "severity_level": "urgent",
                "rationale": "Periumbilical pain migrating to right lower quadrant concerning for acute surgical abdomen",
                "source_quote": "pain started around my belly button and has moved to my right lower side"
            }
        ]
    }

    result = triage_engine.evaluate(evidence_state, messages, ai_status="ready")
    assert result.priority_category == "P1"
    assert any(t.rule_id == "RULE-P1-MODEL-01" for t in result.triggered_rules)
