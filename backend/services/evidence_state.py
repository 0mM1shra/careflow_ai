from datetime import datetime
import json
import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from backend.database.models import Case, Message, Evidence, AuditEvent
from backend.schemas.clinical import ClinicalExtractionOutput, EvidenceItemSchema

logger = logging.getLogger("careflow.evidence_state")

class EvidenceStateManager:
    """
    Manages persistent, auditable evidence records for clinical cases.
    
    Guarantees:
    - Never overwrites conflicting statements; retains both source records.
    - Preserves unknown answers with status='unknown' without converting them to negative/absent.
    - Maintains exact verbatim patient quotes.
    - Records immutable audit events on every change.
    """

    @staticmethod
    def update_evidence_state(
        db: Session,
        case: Case,
        extraction: ClinicalExtractionOutput,
        source_message: Optional[Message] = None,
        actor_role: str = "ai_system"
    ) -> Dict[str, Any]:
        """
        Merges new extracted facts into the case's evidence state.
        Detects conflicts, updates the evidence table, and creates an audit log.
        """
        old_state = json.loads(case.evidence_state_json or "{}")
        current_facts = old_state.get("facts", {})
        contradictions = list(old_state.get("contradictions", []))
        uncertainties = list(old_state.get("uncertainties", []))
        unknowns = list(old_state.get("unknown_information", []))
        negatives = list(old_state.get("explicit_negative_findings", []))

        # Merge extracted uncertainties & contradictions
        for c in extraction.contradictions:
            if c not in contradictions:
                contradictions.append(c)
        for u in extraction.uncertainties:
            if u not in uncertainties:
                uncertainties.append(u)
        for unk in extraction.unknown_information:
            if unk not in unknowns:
                unknowns.append(unk)
        for neg in extraction.explicit_negative_findings:
            if neg not in negatives:
                negatives.append(neg)

        # Process each evidence item from extraction
        for item in extraction.evidence_items:
            field = item.field
            new_value = item.value
            status = item.status
            quote = item.source_quote or (source_message.content if source_message else None)
            source_msg_id = source_message.id if source_message else None

            # Check if this field was already recorded
            if field in current_facts:
                existing_fact = current_facts[field]
                existing_val = existing_fact.get("value")
                existing_status = existing_fact.get("status")

                # Detect conflict if values are significantly different and not unknown
                if existing_status == "reported" and status == "reported" and str(existing_val).lower() != str(new_value).lower():
                    conflict_desc = f"Conflict in field '{field}': previously '{existing_val}', now reported as '{new_value}'"
                    if conflict_desc not in contradictions:
                        contradictions.append(conflict_desc)
                    status = "conflicting"
                    # Update status of existing fact
                    existing_fact["status"] = "conflicting"
                    existing_fact["conflict_note"] = conflict_desc

            # Create persistent database Evidence row
            evidence_row = Evidence(
                case_id=case.id,
                field=field,
                value_json=json.dumps(new_value),
                status=status,
                source_message_id=source_msg_id,
                source_quote=quote,
                created_at=datetime.utcnow()
            )
            db.add(evidence_row)

            # Update working dictionary
            current_facts[field] = {
                "field": field,
                "value": new_value,
                "status": status,
                "source_type": item.source_type,
                "source_message_id": source_msg_id,
                "source_quote": quote,
                "updated_at": datetime.utcnow().isoformat()
            }

        # Update high-level summary fields if reported
        if extraction.onset_and_duration and extraction.onset_and_duration != "Not reported":
            current_facts["onset_and_duration"] = {
                "field": "onset_and_duration",
                "value": extraction.onset_and_duration,
                "status": "reported",
                "source_type": "model_extraction",
                "source_message_id": source_message.id if source_message else None,
                "source_quote": source_message.content if source_message else None,
                "updated_at": datetime.utcnow().isoformat()
            }

        if extraction.severity and extraction.severity != "Not reported":
            current_facts["severity"] = {
                "field": "severity",
                "value": extraction.severity,
                "status": "reported",
                "source_type": "model_extraction",
                "source_message_id": source_message.id if source_message else None,
                "source_quote": source_message.content if source_message else None,
                "updated_at": datetime.utcnow().isoformat()
            }

        if extraction.progression and extraction.progression != "Not reported":
            current_facts["progression"] = {
                "field": "progression",
                "value": extraction.progression,
                "status": "reported",
                "source_type": "model_extraction",
                "source_message_id": source_message.id if source_message else None,
                "source_quote": source_message.content if source_message else None,
                "updated_at": datetime.utcnow().isoformat()
            }

        # Check for missing critical fields
        missing_critical = []
        if "onset_and_duration" not in current_facts or current_facts["onset_and_duration"]["value"] in ["Not reported", "Unknown"]:
            missing_critical.append("Onset and duration of symptoms")
        if "severity" not in current_facts or current_facts["severity"]["value"] in ["Not reported", "Unknown"]:
            missing_critical.append("Pain or severity level")

        new_state = {
            "facts": current_facts,
            "contradictions": contradictions,
            "uncertainties": uncertainties,
            "unknown_information": unknowns,
            "explicit_negative_findings": negatives,
            "missing_information": missing_critical,
            "last_updated": datetime.utcnow().isoformat()
        }

        # Update case entity
        case.evidence_state_json = json.dumps(new_state)
        case.missing_information_json = json.dumps(missing_critical)
        case.uncertainties_json = json.dumps(uncertainties)

        # Record Audit Event
        audit_event = AuditEvent(
            case_id=case.id,
            event_type="evidence_state_updated",
            old_value_json=json.dumps(old_state),
            new_value_json=json.dumps(new_state),
            reason=f"Extracted {len(extraction.evidence_items)} evidence items from message",
            actor_role=actor_role,
            created_at=datetime.utcnow()
        )
        db.add(audit_event)
        db.commit()

        return new_state
