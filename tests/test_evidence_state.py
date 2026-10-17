import json
from datetime import datetime
from backend.database.models import Case, Message, Evidence, AuditEvent
from backend.services.evidence_state import EvidenceStateManager
from backend.schemas.clinical import ClinicalExtractionOutput, EvidenceItemSchema

def test_evidence_state_unknown_preservation(db_session):
    """Ensures that unknown or skipped answers are preserved as 'unknown' and never treated as negative/absent."""
    case = Case(
        id="CF-2026-UNK01",
        preferred_language="English",
        chief_complaint="Headache",
        ai_status="pending",
        priority_category="P3",
        review_status="pending"
    )
    db_session.add(case)
    db_session.commit()

    msg = Message(
        case_id=case.id,
        role="patient",
        content="I don't know when it started.",
        created_at=datetime.utcnow()
    )
    db_session.add(msg)
    db_session.commit()

    extraction = ClinicalExtractionOutput(
        symptoms=["headache"],
        onset_and_duration="Not reported",
        unknown_information=["Patient stated: 'I don't know when it started.'"],
        evidence_items=[
            EvidenceItemSchema(
                field="onset_and_duration",
                value="unknown",
                status="unknown",
                source_type="patient_message",
                source_quote="I don't know when it started."
            )
        ]
    )

    state = EvidenceStateManager.update_evidence_state(
        db=db_session,
        case=case,
        extraction=extraction,
        source_message=msg,
        actor_role="ai_system"
    )

    assert "onset_and_duration" in state["facts"]
    assert state["facts"]["onset_and_duration"]["status"] == "unknown"
    # Verify it was NOT converted to an explicit negative finding
    assert "no onset" not in state.get("explicit_negative_findings", [])
    assert len(state["unknown_information"]) > 0

    # Verify audit event logged
    audit = db_session.query(AuditEvent).filter(AuditEvent.case_id == case.id).first()
    assert audit is not None
    assert audit.event_type == "evidence_state_updated"


def test_evidence_state_conflict_handling(db_session):
    """Ensures that conflicting answers preserve both source records and are flagged as conflicting."""
    case = Case(
        id="CF-2026-CONF01",
        preferred_language="English",
        chief_complaint="Fever",
        ai_status="pending",
        priority_category="P3",
        review_status="pending"
    )
    db_session.add(case)
    db_session.commit()

    # First message: duration is 2 days
    msg1 = Message(
        case_id=case.id,
        role="patient",
        content="Fever started 2 days ago.",
        created_at=datetime.utcnow()
    )
    db_session.add(msg1)
    db_session.commit()

    extraction1 = ClinicalExtractionOutput(
        symptoms=["fever"],
        evidence_items=[
            EvidenceItemSchema(
                field="fever_duration",
                value="2 days",
                status="reported",
                source_type="patient_message",
                source_quote="Fever started 2 days ago."
            )
        ]
    )
    EvidenceStateManager.update_evidence_state(db_session, case, extraction1, msg1)

    # Second message: duration is now claimed to be 2 months
    msg2 = Message(
        case_id=case.id,
        role="patient",
        content="Actually, I have had this fever for 2 months.",
        created_at=datetime.utcnow()
    )
    db_session.add(msg2)
    db_session.commit()

    extraction2 = ClinicalExtractionOutput(
        symptoms=["fever"],
        evidence_items=[
            EvidenceItemSchema(
                field="fever_duration",
                value="2 months",
                status="reported",
                source_type="patient_message",
                source_quote="Actually, I have had this fever for 2 months."
            )
        ]
    )
    updated_state = EvidenceStateManager.update_evidence_state(db_session, case, extraction2, msg2)

    # Check that conflict was detected
    assert len(updated_state["contradictions"]) > 0
    assert updated_state["facts"]["fever_duration"]["status"] == "conflicting"

    # Verify both evidence records exist in the Evidence table
    evidence_rows = db_session.query(Evidence).filter(
        Evidence.case_id == case.id,
        Evidence.field == "fever_duration"
    ).all()
    assert len(evidence_rows) == 2
    values = [json.loads(row.value_json) for row in evidence_rows]
    assert "2 days" in values
    assert "2 months" in values
