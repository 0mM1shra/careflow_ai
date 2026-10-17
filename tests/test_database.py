import json
from datetime import datetime
from backend.database.models import Case, Message, Evidence, AuditEvent

def test_create_case_and_relationships(db_session):
    """Verifies creating a case, adding messages, evidence, and audit logs with relationships."""
    case = Case(
        id="CF-2026-TEST01",
        preferred_language="English",
        chief_complaint="Chest pressure",
        ai_status="pending",
        priority_category="P0",
        review_status="pending"
    )
    db_session.add(case)
    db_session.commit()

    # Add message
    msg = Message(
        case_id=case.id,
        role="patient",
        content="I have had chest pressure for 2 hours.",
        created_at=datetime.utcnow()
    )
    db_session.add(msg)
    db_session.commit()
    db_session.refresh(msg)

    # Add atomic evidence item linked to message
    ev = Evidence(
        case_id=case.id,
        field="chest_pressure",
        value_json=json.dumps("chest pressure"),
        status="reported",
        source_message_id=msg.id,
        source_quote="I have had chest pressure for 2 hours.",
        created_at=datetime.utcnow()
    )
    db_session.add(ev)

    # Add audit log
    audit = AuditEvent(
        case_id=case.id,
        event_type="case_created",
        reason="Test initiation",
        actor_role="patient",
        created_at=datetime.utcnow()
    )
    db_session.add(audit)
    db_session.commit()

    # Query back
    queried = db_session.query(Case).filter(Case.id == "CF-2026-TEST01").first()
    assert queried is not None
    assert queried.chief_complaint == "Chest pressure"
    assert len(queried.messages) == 1
    assert queried.messages[0].content == "I have had chest pressure for 2 hours."
    assert len(queried.evidence_items) == 1
    assert queried.evidence_items[0].source_message_id == msg.id
    assert len(queried.audit_events) == 1
    assert queried.audit_events[0].actor_role == "patient"
