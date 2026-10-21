"""
Compatibility bridge for backend.ml_triage_service.
Re-exports predict_urgency_lgbm, COLUMNS, MLTriageResult from backend.services.ml_triage_service.
"""

from backend.services.ml_triage_service import (
    predict_urgency_lgbm,
    construct_tabular_features,
    get_model,
    MLTriageResult,
    COLUMNS,
    CLASS_TO_TIER
)

__all__ = [
    "predict_urgency_lgbm",
    "construct_tabular_features",
    "get_model",
    "MLTriageResult",
    "COLUMNS",
    "CLASS_TO_TIER"
]
