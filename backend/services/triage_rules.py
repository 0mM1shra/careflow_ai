import re
import logging
from typing import Dict, Any, List, Optional
from backend.schemas.clinical import TriageResult, TriageTrigger
from backend.config import settings

logger = logging.getLogger("careflow.triage")

# Documented Protocol Source:
# Adapted for illustrative clinical research prototype from Emergency Severity Index (ESI)
# and Manchester Triage System (MTS) Red Flag Discriminators.
# PROTOCOL_VERSION = "v1.0.0-prototype"

class DeterministicTriageEngine:
    """
    Independent, rule-based clinical triage engine.
    
    Evaluates grounded evidence and facts extracted from patient intake.
    MedGemma DOES NOT determine the urgency category.
    
    Triage Hierarchy:
    1. System Health / Model Failure Check -> 'ERR' (Review required — AI unavailable)
    2. Emergency Red Flags (P0) -> Immediate Escalation
    3. Urgent Indicators (P1) -> Priority Clinician Assessment
    4. Unresolved Contradictions or Missing Critical Flags (P2) -> Incomplete / Uncertain
    5. Routine Review (P3) -> Standard Non-Urgent Queue
    """

    PROTOCOL_VERSION = settings.TRIAGE_RULE_VERSION

    # P0: Emergency Red Flag Rules
    P0_RULES = [
        {
            "id": "RULE-P0-CARD-01",
            "name": "Acute Coronary Syndrome Red Flag",
            "criterion": "Chest pain radiating to left arm, neck, or jaw",
            "pattern": r"(chest\s*pain|pressure\s*in\s*chest|chest\s*tightness).*(radiat|arm|jaw|neck)|(radiat|arm|jaw|neck).*(chest\s*pain|chest\s*pressure)",
            "action": "Immediate emergency department transfer; activate acute coronary protocol."
        },
        {
            "id": "RULE-P0-RESP-02",
            "name": "Acute Severe Respiratory Compromise",
            "criterion": "Severe shortness of breath, inability to speak full sentences, or stridor",
            "pattern": r"(can't\s*breathe|cannot\s*breathe|gasping|severe\s*shortness\s*of\s*breath|stridor|choking)",
            "action": "Immediate respiratory assessment and oxygenation support."
        },
        {
            "id": "RULE-P0-NEURO-03",
            "name": "Acute Stroke Symptom Complex (FAST)",
            "criterion": "Sudden facial drooping, arm weakness, or slurred speech",
            "pattern": r"(facial\s*droop|face\s*droop|slurred\s*speech|weakness\s*on\s*one\s*side|sudden\s*numbness|can't\s*move\s*arm)",
            "action": "Activate emergency stroke pathway; record time of last known normal."
        },
        {
            "id": "RULE-P0-HEM-04",
            "name": "Severe Hemorrhage or Vascular Collapse",
            "criterion": "Uncontrolled bleeding, vomiting blood, or sudden syncope with chest/back pain",
            "pattern": r"(vomiting\s*blood|coughing\s*up\s*blood|uncontrolled\s*bleeding|passed\s*out|syncope)",
            "action": "Immediate resuscitation and surgical/trauma evaluation."
        },
        {
            "id": "RULE-P0-ANAPH-05",
            "name": "Anaphylaxis Warning Signs",
            "criterion": "Swelling of lips/tongue/throat with respiratory distress or sudden generalized hives",
            "pattern": r"(swollen\s*tongue|swollen\s*throat|throat\s*closing|lips\s*swelling|anaphylaxis)",
            "action": "Immediate intramuscular epinephrine and airway management."
        }
    ]

    # P1: Urgent Review Rules
    P1_RULES = [
        {
            "id": "RULE-P1-INF-01",
            "name": "Meningeal Irritation in Febrile Illness",
            "criterion": "High fever accompanied by neck stiffness or photophobia",
            "pattern": r"(fever|high\s*temp).*(stiff.*neck|neck.*stiff|cannot\s*bend\s*neck|hurts\s*to\s*bend.*head)|(stiff.*neck|neck.*stiff).*(fever|temp)",
            "action": "Urgent in-person physician assessment within 1 hour."
        },
        {
            "id": "RULE-P1-PAIN-02",
            "name": "Severe Uncontrolled Pain",
            "criterion": "Reported pain severity >= 8/10",
            "pattern": r"(\b[89]|10)\s*(?:out\s*of\s*10|/10)|severe\s*unbearable\s*pain",
            "action": "Urgent clinical review for acute pain etiology and analgesia."
        },
        {
            "id": "RULE-P1-ABD-03",
            "name": "Acute Abdomen with Intolerance",
            "criterion": "Severe abdominal pain with intractable vomiting or fever",
            "pattern": r"(abdominal\s*pain|stomach\s*pain|belly\s*pain).*(vomit|throwing\s*up)|(vomit|throwing\s*up).*(abdominal\s*pain|stomach\s*pain)",
            "action": "Urgent surgical / abdominal workup within 2 hours."
        },
        {
            "id": "RULE-P1-RESP-04",
            "name": "Subacute Respiratory Distress",
            "criterion": "Shortness of breath with fever or cough, without immediate collapse",
            "pattern": r"(shortness\s*of\s*breath|trouble\s*breathing).*(fever|cough)|(fever|cough).*(shortness\s*of\s*breath|trouble\s*breathing)",
            "action": "Urgent chest evaluation and pulse oximetry."
        }
    ]

    def evaluate(
        self,
        evidence_state: Dict[str, Any],
        raw_messages: List[str],
        ai_status: str = "ready"
    ) -> TriageResult:
        """
        Evaluates clinical evidence and generates a transparent, auditable TriageResult.
        """
        # 1. Check for AI / System Failure
        if ai_status == "failed":
            logger.warning("AI status is 'failed'. Triage defaulting to ERR.")
            return TriageResult(
                priority_category="ERR",
                priority_label="⚠️ Review Required — AI Unavailable",
                reasons=["AI model service or schema validation failed. Manual clinician review mandatory."],
                warning_signs=["System failure: AI-extracted evidence unavailable."],
                triggered_rules=[
                    TriageTrigger(
                        rule_id="RULE-SYS-01",
                        rule_name="System Failure Safeguard",
                        priority="ERR",
                        criterion="AI extraction failure or invalid model output",
                        matched_evidence=f"ai_status: {ai_status}",
                        action_protocol="Manual triage by registered nurse or medical reviewer."
                    )
                ],
                missing_information=["Full automated clinical extraction"],
                uncertainties=["All fields require manual clinician assessment"],
                triage_rule_version=self.PROTOCOL_VERSION
            )

        # Aggregate all text for pattern verification
        all_patient_text = " ".join(raw_messages).lower()

        triggered_rules: List[TriageTrigger] = []
        warning_signs: List[str] = []
        reasons: List[str] = []

        # 2. Evaluate P0 (Emergency) Rules FIRST
        # Emergency triggers can NEVER be downgraded by absence of other symptoms.
        for rule in self.P0_RULES:
            match = re.search(rule["pattern"], all_patient_text)
            if match:
                matched_snippet = match.group(0)
                # Find which message contained it
                source_msg = next((msg for msg in raw_messages if matched_snippet in msg.lower()), matched_snippet)
                
                trigger = TriageTrigger(
                    rule_id=rule["id"],
                    rule_name=rule["name"],
                    priority="P0",
                    criterion=rule["criterion"],
                    matched_evidence=f"Patient stated: '{source_msg[:120]}'",
                    action_protocol=rule["action"]
                )
                triggered_rules.append(trigger)
                warning_signs.append(f"{rule['name']}: {rule['criterion']}")
                reasons.append(f"Triggered {rule['id']} ({rule['name']}): '{source_msg[:90]}'")

        if triggered_rules:
            return TriageResult(
                priority_category="P0",
                priority_label="🚨 P0: Emergency Escalation",
                reasons=reasons,
                warning_signs=warning_signs,
                triggered_rules=triggered_rules,
                missing_information=evidence_state.get("missing_information", []),
                uncertainties=evidence_state.get("uncertainties", []),
                triage_rule_version=self.PROTOCOL_VERSION
            )

        # 3. Evaluate P1 (Urgent) Rules
        for rule in self.P1_RULES:
            match = re.search(rule["pattern"], all_patient_text)
            if match:
                matched_snippet = match.group(0)
                source_msg = next((msg for msg in raw_messages if matched_snippet in msg.lower()), matched_snippet)
                
                trigger = TriageTrigger(
                    rule_id=rule["id"],
                    rule_name=rule["name"],
                    priority="P1",
                    criterion=rule["criterion"],
                    matched_evidence=f"Patient stated: '{source_msg[:120]}'",
                    action_protocol=rule["action"]
                )
                triggered_rules.append(trigger)
                warning_signs.append(f"{rule['name']}: {rule['criterion']}")
                reasons.append(f"Triggered {rule['id']} ({rule['name']}): '{source_msg[:90]}'")

        if triggered_rules:
            return TriageResult(
                priority_category="P1",
                priority_label="⚠️ P1: Urgent Review",
                reasons=reasons,
                warning_signs=warning_signs,
                triggered_rules=triggered_rules,
                missing_information=evidence_state.get("missing_information", []),
                uncertainties=evidence_state.get("uncertainties", []),
                triage_rule_version=self.PROTOCOL_VERSION
            )

        # 4. Evaluate P2 (Incomplete or Uncertain)
        # Check for contradictions, unresolved uncertainties, or severe symptoms without timeline
        contradictions = evidence_state.get("contradictions", [])
        uncertainties = evidence_state.get("uncertainties", [])
        missing_info = evidence_state.get("missing_information", [])

        p2_triggers = []
        if contradictions:
            p2_triggers.append(
                TriageTrigger(
                    rule_id="RULE-P2-CONFLICT-01",
                    rule_name="Unresolved Contradictory Evidence",
                    priority="P2",
                    criterion="Conflicting statements detected in patient conversation",
                    matched_evidence="; ".join(contradictions),
                    action_protocol="Clinician callback or targeted re-interview to resolve conflicting reports."
                )
            )
            reasons.append(f"Contradictory information present: {'; '.join(contradictions)}")

        if uncertainties and len(uncertainties) >= 2:
            p2_triggers.append(
                TriageTrigger(
                    rule_id="RULE-P2-UNCERTAIN-02",
                    rule_name="High Clinical Uncertainty",
                    priority="P2",
                    criterion="Multiple ambiguous or uncertain patient reports",
                    matched_evidence="; ".join(uncertainties),
                    action_protocol="Reviewer clarification required."
                )
            )
            reasons.append(f"High clinical uncertainty: {'; '.join(uncertainties)}")

        if p2_triggers:
            return TriageResult(
                priority_category="P2",
                priority_label="❓ P2: Incomplete or Uncertain",
                reasons=reasons,
                warning_signs=[],
                triggered_rules=p2_triggers,
                missing_information=missing_info,
                uncertainties=uncertainties,
                triage_rule_version=self.PROTOCOL_VERSION
            )

        # 5. Fallback to P3 (Routine Review)
        return TriageResult(
            priority_category="P3",
            priority_label="📋 P3: Routine Review",
            reasons=["No emergency (P0), urgent (P1), or contradictory/uncertain (P2) criteria detected. Patient meets criteria for routine scheduled assessment."],
            warning_signs=[],
            triggered_rules=[
                TriageTrigger(
                    rule_id="RULE-P3-STD-01",
                    rule_name="Standard Routine Criteria",
                    priority="P3",
                    criterion="Absence of acute red-flag or escalation markers; stable clinical picture",
                    matched_evidence="All reported symptoms are non-escalating and manageable in standard intake.",
                    action_protocol="Standard clinic queue; routine medical appointment."
                )
            ],
            missing_information=missing_info,
            uncertainties=uncertainties,
            triage_rule_version=self.PROTOCOL_VERSION
        )

# Global singleton
triage_engine = DeterministicTriageEngine()
