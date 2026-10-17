import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from evaluation.fixtures import TEST_FIXTURES
from backend.services.medgemma import medgemma_service, ModelInferenceError
from backend.services.extraction import clinical_extraction_service
from backend.services.questioning import adaptive_questioning_service
from backend.services.summarization import clinical_summarization_service
from backend.services.triage_rules import triage_engine

RESULTS_DIR = Path(__file__).resolve().parent.parent / "evaluation_results"
RESULTS_DIR.mkdir(exist_ok=True)

def run_evaluation_pipeline() -> Dict[str, Any]:
    """
    Executes all benchmark test fixtures through the CareFlow pipeline,
    records quantitative metrics, and writes JSONL output.
    """
    eval_run_id = f"eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    case_results = []

    total_fixtures = len(TEST_FIXTURES)
    triage_matches = 0
    schema_failures = 0
    extraction_matches = 0
    conflict_matches = 0
    unknown_matches = 0
    latencies = []

    print(f"\n========================================================")
    print(f"  CareFlow Clinical AI Benchmark: MedGemma Evaluation   ")
    print(f"  Run ID: {eval_run_id} | Mode: {medgemma_service.mode} ")
    print(f"========================================================\n")

    for idx, fixture in enumerate(TEST_FIXTURES, 1):
        print(f"[{idx}/{total_fixtures}] Testing '{fixture['id']}': {fixture['title']}...")
        start_t = time.perf_counter()

        extraction_out = None
        summary_out = None
        triage_out = None
        ai_status = "ready"
        is_schema_valid = True

        # Check for simulated model failure
        if fixture.get("force_model_failure", False):
            ai_status = "failed"
            # Deterministic triage handles AI failure
            triage_out = triage_engine.evaluate(
                evidence_state={},
                raw_messages=fixture["messages"],
                ai_status="failed"
            )
        else:
            try:
                # 1. Clinical Extraction
                extraction_out = clinical_extraction_service.extract_clinical_facts(fixture["messages"])
                
                # 2. Mock Evidence State
                evidence_state = {
                    "facts": {item.field: item.value for item in extraction_out.evidence_items},
                    "contradictions": extraction_out.contradictions,
                    "uncertainties": extraction_out.uncertainties,
                    "unknown_information": extraction_out.unknown_information,
                    "missing_information": []
                }

                # 3. Clinical Summarization
                summary_out = clinical_summarization_service.generate_summary(fixture["messages"])

                # 4. Deterministic Triage
                triage_out = triage_engine.evaluate(
                    evidence_state=evidence_state,
                    raw_messages=fixture["messages"],
                    ai_status="ready"
                )

            except Exception as e:
                schema_failures += 1
                is_schema_valid = False
                ai_status = "failed"
                triage_out = triage_engine.evaluate(
                    evidence_state={},
                    raw_messages=fixture["messages"],
                    ai_status="failed"
                )

        latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
        latencies.append(latency_ms)

        # Check Priority Match
        priority_matched = (triage_out.priority_category == fixture["expected_priority"])
        if priority_matched:
            triage_matches += 1

        # Check Extraction Match
        if fixture.get("force_model_failure", False):
            symptom_hit = True
            extraction_matches += 1
        else:
            extracted_symptoms = [s.lower() for s in (extraction_out.symptoms if extraction_out else [])]
            symptom_hit = any(
                any(exp in s for s in extracted_symptoms)
                for exp in fixture["expected_symptoms"]
            ) if fixture["expected_symptoms"] else True
            if symptom_hit:
                extraction_matches += 1

        # Check Contradiction Detection
        has_contradiction = bool(extraction_out and extraction_out.contradictions)
        if has_contradiction == fixture["expect_contradiction"]:
            conflict_matches += 1

        # Check Unknown Preservation
        has_unknown = bool(extraction_out and extraction_out.unknown_information)
        if has_unknown == fixture["expect_unknown"]:
            unknown_matches += 1

        case_record = {
            "fixture_id": fixture["id"],
            "title": fixture["title"],
            "expected_priority": fixture["expected_priority"],
            "assigned_priority": triage_out.priority_category,
            "priority_match": priority_matched,
            "triggered_rule_ids": [t.rule_id for t in triage_out.triggered_rules],
            "schema_valid": is_schema_valid,
            "ai_status": ai_status,
            "latency_ms": latency_ms,
            "symptoms_extracted": extraction_out.symptoms if extraction_out else [],
            "summary_text": summary_out.summary_text if summary_out else "AI summary unavailable",
            "reasons": triage_out.reasons
        }
        case_results.append(case_record)

        print(f"       -> Expected: {fixture['expected_priority']} | Assigned: {triage_out.priority_category} "
              f"({'MATCH' if priority_matched else 'MISMATCH'}) | Latency: {latency_ms}ms")

    # Aggregate Metrics
    triage_accuracy = round((triage_matches / total_fixtures) * 100, 1)
    extraction_accuracy = round((extraction_matches / total_fixtures) * 100, 1)
    conflict_accuracy = round((conflict_matches / total_fixtures) * 100, 1)
    unknown_accuracy = round((unknown_matches / total_fixtures) * 100, 1)
    avg_latency = round(sum(latencies) / len(latencies), 2)
    schema_error_rate = round((schema_failures / total_fixtures) * 100, 1)

    benchmark_summary = {
        "run_id": eval_run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "model_mode": medgemma_service.mode,
        "model_id": medgemma_service.model_id,
        "total_fixtures": total_fixtures,
        "triage_accuracy_pct": triage_accuracy,
        "extraction_accuracy_pct": extraction_accuracy,
        "conflict_accuracy_pct": conflict_accuracy,
        "unknown_accuracy_pct": unknown_accuracy,
        "schema_error_rate_pct": schema_error_rate,
        "mean_latency_ms": avg_latency,
        "cases": case_results
    }

    # Save to JSONL
    out_file = RESULTS_DIR / f"{eval_run_id}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for case in case_results:
            f.write(json.dumps(case) + "\n")

    summary_file = RESULTS_DIR / f"{eval_run_id}_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    print(f"\n--------------------------------------------------------")
    print(f"  Benchmark Summary Metrics:")
    print(f"  • Triage Protocol Accuracy:    {triage_accuracy}% ({triage_matches}/{total_fixtures})")
    print(f"  • Clinical Extraction Accuracy: {extraction_accuracy}% ({extraction_matches}/{total_fixtures})")
    print(f"  • Conflict Preservation Rate:  {conflict_accuracy}% ({conflict_matches}/{total_fixtures})")
    print(f"  • Unknown Preservation Rate:   {unknown_accuracy}% ({unknown_matches}/{total_fixtures})")
    print(f"  • Schema Error Rate:           {schema_error_rate}% ({schema_failures}/{total_fixtures})")
    print(f"  • Mean Pipeline Latency:       {avg_latency} ms")
    print(f"  • Results Logged To:           {out_file}")
    print(f"--------------------------------------------------------\n")

    return benchmark_summary

if __name__ == "__main__":
    run_evaluation_pipeline()
