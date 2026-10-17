"""
Permitted illustrative evaluation test fixtures for CareFlow.
Clearly labeled as research benchmark fixtures, NOT real clinical patient records.
"""

TEST_FIXTURES = [
    {
        "id": "EVAL-01-P0-CARDIAC",
        "title": "Acute Coronary Red Flag (Emergency P0)",
        "description": "Patient reports retrosternal crushing chest pressure radiating to left arm and jaw.",
        "messages": [
            "I have crushing chest pain that started 45 minutes ago.",
            "Yes, the pain radiates down my left arm and up to my jaw. I feel dizzy."
        ],
        "expected_priority": "P0",
        "expected_rule_id": "RULE-P0-CARD-01",
        "expected_symptoms": ["chest pain"],
        "expect_contradiction": False,
        "expect_unknown": False,
        "force_model_failure": False
    },
    {
        "id": "EVAL-02-P1-MENINGEAL",
        "title": "Febrile Meningeal Sign (Urgent P1)",
        "description": "High fever with acute neck stiffness and difficulty bending neck.",
        "messages": [
            "I have had a high fever for 2 days.",
            "My neck feels very stiff today and it hurts to bend my head forward."
        ],
        "expected_priority": "P1",
        "expected_rule_id": "RULE-P1-INF-01",
        "expected_symptoms": ["fever"],
        "expect_contradiction": False,
        "expect_unknown": False,
        "force_model_failure": False
    },
    {
        "id": "EVAL-03-P1-SEVERE-PAIN",
        "title": "Severe Pain Acuity (Urgent P1)",
        "description": "Acute severe flank/abdominal pain rated 9/10.",
        "messages": [
            "Severe sharp belly and flank pain started this morning.",
            "The pain is 9 out of 10 and I cannot get comfortable."
        ],
        "expected_priority": "P1",
        "expected_rule_id": "RULE-P1-PAIN-02",
        "expected_symptoms": ["abdominal pain"],
        "expect_contradiction": False,
        "expect_unknown": False,
        "force_model_failure": False
    },
    {
        "id": "EVAL-04-P2-CONTRADICTION",
        "title": "Conflicting Timeline Reports (Uncertain P2)",
        "description": "Patient first claims symptoms started yesterday, then claims 6 months.",
        "messages": [
            "I have had a cough that started yesterday.",
            "Actually, I have been coughing like this for 6 months."
        ],
        "expected_priority": "P2",
        "expected_rule_id": "RULE-P2-CONFLICT-01",
        "expected_symptoms": ["cough"],
        "expect_contradiction": True,
        "expect_unknown": False,
        "force_model_failure": False
    },
    {
        "id": "EVAL-05-P2-UNKNOWN-DIMENSION",
        "title": "Unknown Answer Preservation (Unknown P2/P3)",
        "description": "Patient explicitly skips or answers 'I do not know' to onset question.",
        "messages": [
            "I have a strange rash on my forearm.",
            "I don't know when it first appeared, I just noticed it."
        ],
        "expected_priority": "P3",
        "expected_rule_id": "RULE-P3-STD-01",
        "expected_symptoms": ["rash"],
        "expect_contradiction": False,
        "expect_unknown": True,
        "force_model_failure": False
    },
    {
        "id": "EVAL-06-P3-ROUTINE",
        "title": "Mild Upper Respiratory Symptoms (Routine P3)",
        "description": "Mild cold symptoms without fever, respiratory distress, or severe pain.",
        "messages": [
            "Mild runny nose, sneezing, and slight tickle in throat for 3 days.",
            "No fever, no difficulty breathing, pain is 1/10."
        ],
        "expected_priority": "P3",
        "expected_rule_id": "RULE-P3-STD-01",
        "expected_symptoms": ["runny nose", "throat"],
        "expect_contradiction": False,
        "expect_unknown": False,
        "force_model_failure": False
    },
    {
        "id": "EVAL-07-ERR-FAILURE",
        "title": "AI Service Failure Degradation (ERR)",
        "description": "Simulated model/parser failure verifying safe fallback to human review.",
        "messages": [
            "Vague joint pain in wrist."
        ],
        "expected_priority": "ERR",
        "expected_rule_id": "RULE-SYS-01",
        "expected_symptoms": [],
        "expect_contradiction": False,
        "expect_unknown": False,
        "force_model_failure": True
    }
]
