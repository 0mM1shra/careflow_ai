import os
import re
import json
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import pandas as pd
import joblib

logger = logging.getLogger("careflow.ml_triage")

# Ordered features matching careflow_robust_ml.csv
FEATURE_COLUMNS = [
    "age",
    "heart_rate",
    "systolic_bp",
    "spo2_pct",
    "temp_f",
    "pain_severe",
    "onset_acute",
    "sys_cardiopulmonary",
    "sys_neurological",
    "sys_gastrointestinal",
    "sys_urogenital",
    "sys_musculoskeletal",
    "sys_dermatological",
    "sys_systemic",
    "sys_eent",
]

# Population median fallbacks derived from careflow_robust_ml.csv
DEFAULT_VITALS = {
    "age": 50,
    "heart_rate": 88,
    "systolic_bp": 125,
    "spo2_pct": 96,
    "temp_f": 98.6,
}

# Integer class mapping to clinical urgency categories
TARGET_PRIORITY_MAP = {
    0: "P0",
    1: "P1",
    2: "P2",
    3: "P3",
}

_MODEL_INSTANCE = None
_DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "lgbm_urgency_model.joblib"
_DEFAULT_DATASET_PATH = Path(__file__).resolve().parent.parent / "careflow_robust_dataset" / "careflow_robust_ml.csv"


def train_lgbm_model(dataset_path: Optional[Union[str, Path]] = None, save_path: Optional[Union[str, Path]] = None) -> Any:
    """
    Trains a LightGBM classifier on the tabular dataset and saves the artifact.
    """
    import lightgbm as lgb

    data_file = Path(dataset_path) if dataset_path else _DEFAULT_DATASET_PATH
    output_file = Path(save_path) if save_path else _DEFAULT_MODEL_PATH

    if not data_file.exists():
        raise FileNotFoundError(f"Training dataset not found at {data_file}")

    logger.info(f"Training LightGBM urgency model from {data_file}...")
    df = pd.read_csv(data_file)
    X = df[FEATURE_COLUMNS]
    y = df["target_priority"]

    clf = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
    )
    clf.fit(X, y)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_file)
    logger.info(f"LightGBM urgency model saved to {output_file}")
    return clf


def get_lgbm_model() -> Any:
    """
    Retrieves or loads the cached LightGBM model singleton.
    Auto-trains from careflow_robust_ml.csv if the artifact is missing.
    """
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is not None:
        return _MODEL_INSTANCE

    if _DEFAULT_MODEL_PATH.exists():
        try:
            logger.info(f"Loading LightGBM model from {_DEFAULT_MODEL_PATH}")
            _MODEL_INSTANCE = joblib.load(_DEFAULT_MODEL_PATH)
            return _MODEL_INSTANCE
        except Exception as e:
            logger.warning(f"Failed to load existing model ({e}); re-training...")

    if _DEFAULT_DATASET_PATH.exists():
        _MODEL_INSTANCE = train_lgbm_model(_DEFAULT_DATASET_PATH, _DEFAULT_MODEL_PATH)
        return _MODEL_INSTANCE

    raise RuntimeError(
        f"Unable to load or train LightGBM model: neither {_DEFAULT_MODEL_PATH} "
        f"nor {_DEFAULT_DATASET_PATH} was found."
    )


def extract_vitals_from_dict_or_text(
    vitals_dict: Optional[Dict[str, Any]] = None,
    text_content: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extracts continuous lab/vital parameters (age, heart_rate, systolic_bp, spo2_pct, temp_f)
    from a dictionary or parses them from clinical text/reports.
    """
    vitals = dict(DEFAULT_VITALS)

    if vitals_dict:
        # Age
        if "age" in vitals_dict and vitals_dict["age"] is not None:
            try:
                vitals["age"] = int(vitals_dict["age"])
            except (ValueError, TypeError):
                pass

        # Heart rate / Pulse
        for k in ("heart_rate", "hr", "pulse", "bpm"):
            if k in vitals_dict and vitals_dict[k] is not None:
                try:
                    vitals["heart_rate"] = int(vitals_dict[k])
                    break
                except (ValueError, TypeError):
                    pass

        # Systolic BP
        for k in ("systolic_bp", "systolic", "sbp", "bp_sys", "bp"):
            if k in vitals_dict and vitals_dict[k] is not None:
                val = str(vitals_dict[k])
                if "/" in val:
                    val = val.split("/")[0]
                try:
                    vitals["systolic_bp"] = int(float(re.sub(r"[^\d.]", "", val)))
                    break
                except (ValueError, TypeError):
                    pass

        # SpO2
        for k in ("spo2_pct", "spo2", "oximetry", "oxygen_saturation"):
            if k in vitals_dict and vitals_dict[k] is not None:
                try:
                    vitals["spo2_pct"] = int(float(re.sub(r"[^\d.]", "", str(vitals_dict[k]))))
                    break
                except (ValueError, TypeError):
                    pass

        # Temperature
        for k in ("temp_f", "temperature", "temp", "temp_deg_f"):
            if k in vitals_dict and vitals_dict[k] is not None:
                try:
                    t_val = float(re.sub(r"[^\d.]", "", str(vitals_dict[k])))
                    # Convert Celsius to Fahrenheit if needed
                    if t_val < 45.0:
                        t_val = (t_val * 9.0 / 5.0) + 32.0
                    vitals["temp_f"] = round(t_val, 1)
                    break
                except (ValueError, TypeError):
                    pass

    # If text is provided, extract missing vitals using regex
    if text_content:
        # HR regex (e.g. "HR 109 bpm" or "pulse 95")
        hr_match = re.search(r"\b(?:HR|pulse|heart rate)\s*[:=]?\s*(\d{2,3})\s*(?:bpm)?\b", text_content, re.IGNORECASE)
        if hr_match and vitals["heart_rate"] == DEFAULT_VITALS["heart_rate"]:
            vitals["heart_rate"] = int(hr_match.group(1))

        # BP regex (e.g. "BP 104/70 mmHg" or "BP 120")
        bp_match = re.search(r"\b(?:BP|blood pressure)\s*[:=]?\s*(\d{2,3})(?:/\d{2,3})?\s*(?:mmHg)?\b", text_content, re.IGNORECASE)
        if bp_match and vitals["systolic_bp"] == DEFAULT_VITALS["systolic_bp"]:
            vitals["systolic_bp"] = int(bp_match.group(1))

        # SpO2 regex (e.g. "SpO2 92%" or "oxygen 94%")
        spo2_match = re.search(r"\b(?:SpO2|O2 sat|oxygen)\s*[:=]?\s*(\d{2,3})\s*%?\b", text_content, re.IGNORECASE)
        if spo2_match and vitals["spo2_pct"] == DEFAULT_VITALS["spo2_pct"]:
            vitals["spo2_pct"] = int(spo2_match.group(1))

        # Temp regex (e.g. "Temp 97.0 F" or "temperature: 101.8 F")
        temp_match = re.search(r"\b(?:Temp|temperature)\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:F|deg F)?\b", text_content, re.IGNORECASE)
        if temp_match and vitals["temp_f"] == DEFAULT_VITALS["temp_f"]:
            vitals["temp_f"] = float(temp_match.group(1))

    return vitals


def _map_slm_and_vitals_to_dataframe(
    slm_json_output: Optional[Union[Dict[str, Any], str]],
    lab_vitals_dict: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Maps MedGemma / SLM extracted features and patient vitals into a single-row
    DataFrame matching the exact 15-column schema required by LightGBM.
    """
    # Parse slm_json_output if passed as string
    slm_data: Dict[str, Any] = {}
    if isinstance(slm_json_output, str):
        try:
            slm_data = json.loads(slm_json_output)
        except Exception:
            slm_data = {"raw_text": slm_json_output}
    elif isinstance(slm_json_output, dict):
        slm_data = slm_json_output
    elif slm_json_output is not None:
        slm_data = getattr(slm_json_output, "model_dump", lambda: {})() or {}

    # Extract continuous vitals
    vitals = extract_vitals_from_dict_or_text(lab_vitals_dict)

    # Initialize feature row
    row: Dict[str, Any] = {
        "age": vitals["age"],
        "heart_rate": vitals["heart_rate"],
        "systolic_bp": vitals["systolic_bp"],
        "spo2_pct": vitals["spo2_pct"],
        "temp_f": vitals["temp_f"],
        "pain_severe": 0,
        "onset_acute": 0,
        "sys_cardiopulmonary": 0,
        "sys_neurological": 0,
        "sys_gastrointestinal": 0,
        "sys_urogenital": 0,
        "sys_musculoskeletal": 0,
        "sys_dermatological": 0,
        "sys_systemic": 0,
        "sys_eent": 0,
    }

    # 1. Map Pain Severity (pain_severe)
    if "red_flag_severe_pain" in slm_data:
        row["pain_severe"] = 1 if slm_data["red_flag_severe_pain"] else 0
    elif "pain_severe" in slm_data:
        row["pain_severe"] = 1 if slm_data["pain_severe"] else 0
    else:
        # Check MedGemma standard fields
        sev_str = str(slm_data.get("severity", "")).lower()
        if any(w in sev_str for w in ["severe", "unbearable", "excruciating", "8/10", "9/10", "10/10", "बहुत तेज"]):
            row["pain_severe"] = 1
        elif any(sig.get("severity_level") in ("emergency", "urgent") for sig in slm_data.get("urgency_signals", []) if isinstance(sig, dict)):
            row["pain_severe"] = 1

    # 2. Map Acute Onset (onset_acute)
    if "red_flag_acute_onset" in slm_data:
        row["onset_acute"] = 1 if slm_data["red_flag_acute_onset"] else 0
    elif "onset_acute" in slm_data:
        row["onset_acute"] = 1 if slm_data["onset_acute"] else 0
    else:
        onset_str = str(slm_data.get("onset_and_duration", "")).lower()
        if any(w in onset_str for w in ["sudden", "acute", "today", "hour", "minute", "kal se", "कल से", "achanak"]):
            row["onset_acute"] = 1
        elif not onset_str or onset_str == "not reported":
            row["onset_acute"] = 1  # Default to conservative acute onset assumption

    # 3. Map System Flags (sys_*)
    # Check direct primary_system_flag from SLM
    primary_sys = str(slm_data.get("primary_system_flag", "")).strip().lower()
    if primary_sys:
        if not primary_sys.startswith("sys_"):
            primary_sys = f"sys_{primary_sys}"
        if primary_sys in row:
            row[primary_sys] = 1

    # Also check direct sys_* keys if already binary in slm_data
    for col in FEATURE_COLUMNS:
        if col.startswith("sys_") and col in slm_data:
            if slm_data[col]:
                row[col] = 1

    # If no system flag was matched, infer from symptoms and complaint text
    if sum(row[c] for c in FEATURE_COLUMNS if c.startswith("sys_")) == 0:
        symptoms_text = ""
        if "symptoms" in slm_data and isinstance(slm_data["symptoms"], list):
            symptoms_text += " ".join(str(s) for s in slm_data["symptoms"])
        if "chief_complaint" in slm_data:
            symptoms_text += " " + str(slm_data["chief_complaint"])
        if "raw_text" in slm_data:
            symptoms_text += " " + str(slm_data["raw_text"])
        s_low = symptoms_text.lower()

        # Keyword matching heuristics
        if any(w in s_low for w in ["chest", "heart", "breath", "dyspnea", "cough", "palpitation", "सीने", "छाती", "सांस"]):
            row["sys_cardiopulmonary"] = 1
        elif any(w in s_low for w in ["headache", "dizzy", "head", "stroke", "seizure", "numbness", "सिरदर्द", "चक्कर"]):
            row["sys_neurological"] = 1
        elif any(w in s_low for w in ["stomach", "vomit", "abdomen", "nausea", "belly", "diarrhea", "pet", "पेट", "उल्टी"]):
            row["sys_gastrointestinal"] = 1
        elif any(w in s_low for w in ["urine", "urinary", "bladder", "kidney", "dysuria", "flank"]):
            row["sys_urogenital"] = 1
        elif any(w in s_low for w in ["joint", "bone", "back pain", "muscle", "fracture", "knee", "कमर", "हड्डी"]):
            row["sys_musculoskeletal"] = 1
        elif any(w in s_low for w in ["rash", "skin", "itch", "hives", "allergy", "चकत्ते", "खुजली"]):
            row["sys_dermatological"] = 1
        elif any(w in s_low for w in ["throat", "ear", "eye", "vision", "nose", "gala", "गला", "आंख", "कान"]):
            row["sys_eent"] = 1
        elif any(w in s_low for w in ["fever", "chills", "fatigue", "weakness", "infection", "बुखार"]):
            row["sys_systemic"] = 1
        else:
            # Default to sys_cardiopulmonary for safety if completely undetermined
            row["sys_systemic"] = 1

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_urgency_lgbm(
    slm_json_output: Optional[Union[Dict[str, Any], str]] = None,
    lab_vitals_dict: Optional[Dict[str, Any]] = None
) -> str:
    """
    Main LightGBM Triage Predictor.
    
    1. Loads the trained LightGBM urgency model.
    2. Maps MedGemma / SLM JSON features and patient continuous vitals into the
       15-feature Pandas DataFrame schema matching careflow_robust_ml.csv.
    3. Returns the predicted P0-P3 urgency tier (e.g. 'P0', 'P1', 'P2', 'P3').
    """
    try:
        model = get_lgbm_model()
        df_features = _map_slm_and_vitals_to_dataframe(slm_json_output, lab_vitals_dict)
        pred_class = int(model.predict(df_features)[0])
        return TARGET_PRIORITY_MAP.get(pred_class, "P3")
    except Exception as e:
        logger.error(f"Error predicting urgency via LightGBM: {e}")
        # Safe fallback
        return "P2"


def predict_urgency_lgbm_with_details(
    slm_json_output: Optional[Union[Dict[str, Any], str]] = None,
    lab_vitals_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Predicts urgency and returns detailed class probabilities and feature vector.
    """
    try:
        model = get_lgbm_model()
        df_features = _map_slm_and_vitals_to_dataframe(slm_json_output, lab_vitals_dict)
        pred_class = int(model.predict(df_features)[0])
        probabilities = {}
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df_features)[0]
            for idx, prob in enumerate(probs):
                tier = TARGET_PRIORITY_MAP.get(idx, f"Class_{idx}")
                probabilities[tier] = round(float(prob), 4)

        tier_str = TARGET_PRIORITY_MAP.get(pred_class, "P3")
        return {
            "ml_model_triage": tier_str,
            "predicted_class": pred_class,
            "probabilities": probabilities,
            "feature_vector": df_features.iloc[0].to_dict(),
        }
    except Exception as e:
        logger.error(f"Error in predict_urgency_lgbm_with_details: {e}")
        return {
            "ml_model_triage": "P2",
            "predicted_class": 2,
            "probabilities": {},
            "feature_vector": {},
            "error": str(e),
        }
