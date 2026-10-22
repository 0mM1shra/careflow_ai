import io
import os
import re
import logging
from typing import Optional, Dict, Any, Tuple, List
from PIL import Image

from backend.config import settings
from backend.schemas.multilingual import DocumentExtractionResult, StructuredMedicalFact
from backend.services.safety import safety_service

logger = logging.getLogger("careflow.ocr")


class MedicalDocumentOCRService:
    """
    Extracts text from uploaded medical documents (prescription photos, lab images, PDFs).
    Performs medical document classification, structured clinical fact extraction,
    and automated PII redaction.
    """

    def __init__(self):
        self._paddle_ocr = None

    @classmethod
    def detect_medical_document(cls, text: str) -> Tuple[bool, float, str]:
        """
        Classifies whether extracted text represents a clinical or laboratory document.
        Returns: (is_medical, confidence_score, detected_category)
        """
        if not text or len(text.strip()) < 10:
            return False, 0.0, "non_medical_or_empty"

        t_low = text.lower()
        score = 0
        doc_type = "general_medical_report"

        lab_markers = [
            "hemoglobin", "hb", "wbc", "rbc", "platelet", "glucose", "creatinine",
            "urea", "bilirubin", "serum", "lipid", "cholesterol", "pathology", "cbc",
            "investigation", "reference range", "normal range", "specimen", "fasting"
        ]
        vital_markers = ["heart rate", "pulse", "bp", "blood pressure", "spo2", "temp", "temperature", "systolic", "diastolic", "bpm", "mmhg"]
        rx_markers = ["rx", "prescription", "capsule", "tablet", "syrup", "dosage", "sig:", "mg", "dr.", "doctor", "clinic", "hospital", "dispense"]

        lab_matches = sum(1 for m in lab_markers if m in t_low)
        vital_matches = sum(1 for m in vital_markers if m in t_low)
        rx_matches = sum(1 for m in rx_markers if m in t_low)

        score = lab_matches + vital_matches + rx_matches

        if lab_matches >= 2:
            doc_type = "laboratory_report"
        elif rx_matches >= 2:
            doc_type = "prescription_or_clinical_notes"
        elif vital_matches >= 2:
            doc_type = "vitals_monitoring_sheet"

        is_medical = score >= 2
        confidence = min(round(0.4 + (score * 0.1), 2), 0.98) if is_medical else 0.2
        return is_medical, confidence, doc_type

    @classmethod
    def extract_structured_medical_facts(cls, text: str) -> Tuple[List[StructuredMedicalFact], Dict[str, Any]]:
        """
        Extracts validated clinical parameters, units, reference ranges, report dates,
        and continuous vitals for the LightGBM urgency classifier.
        Flags uncertain or implausible values for human review.
        """
        facts: List[StructuredMedicalFact] = []
        vitals: Dict[str, Any] = {}

        if not text:
            return facts, vitals

        # 1. Report Date Extraction
        report_date = None
        date_match = re.search(r"\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}|\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})\b", text)
        if date_match:
            report_date = date_match.group(1)

        # 2. Blood Pressure (Systolic / Diastolic)
        bp_match = re.search(r"\b(?:blood\s*pressure|bp|systolic/diastolic)\s*[:=]?\s*(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mmHg)?\b", text, re.IGNORECASE)
        if not bp_match:
            bp_match = re.search(r"\b(?:systolic|sbp)\s*[:=]?\s*(\d{2,3})\s*(?:mmHg)?\b", text, re.IGNORECASE)
        if bp_match:
            try:
                sys_val = float(bp_match.group(1))
                is_unreliable = sys_val < 50 or sys_val > 260
                facts.append(StructuredMedicalFact(
                    measurement_name="Systolic Blood Pressure",
                    numeric_value=sys_val,
                    raw_value=bp_match.group(0),
                    unit="mmHg",
                    reference_range="90 - 120 mmHg",
                    report_date=report_date,
                    confidence=0.95,
                    is_uncertain=is_unreliable,
                    uncertainty_reason="Physiologically implausible systolic BP" if is_unreliable else None
                ))
                vitals["systolic_bp"] = int(sys_val)
            except Exception:
                pass

        # 3. Heart Rate / Pulse
        hr_match = re.search(r"\b(?:heart\s*rate|pulse|hr)\s*[:=]?\s*(\d{2,3})\s*(?:bpm|beats/min)?\b", text, re.IGNORECASE)
        if hr_match:
            try:
                hr_val = float(hr_match.group(1))
                is_unreliable = hr_val < 35 or hr_val > 230
                facts.append(StructuredMedicalFact(
                    measurement_name="Heart Rate",
                    numeric_value=hr_val,
                    raw_value=hr_match.group(0),
                    unit="bpm",
                    reference_range="60 - 100 bpm",
                    report_date=report_date,
                    confidence=0.95,
                    is_uncertain=is_unreliable,
                    uncertainty_reason="Physiologically implausible heart rate" if is_unreliable else None
                ))
                vitals["heart_rate"] = int(hr_val)
            except Exception:
                pass

        # 4. Oxygen Saturation (SpO2)
        spo2_match = re.search(r"\b(?:spo2|oxygen\s*sat|o2\s*sat)\s*[:=]?\s*(\d{2,3})\s*%?\b", text, re.IGNORECASE)
        if spo2_match:
            try:
                spo2_val = float(spo2_match.group(1))
                is_unreliable = spo2_val < 40 or spo2_val > 100
                facts.append(StructuredMedicalFact(
                    measurement_name="Oxygen Saturation (SpO2)",
                    numeric_value=spo2_val,
                    raw_value=spo2_match.group(0),
                    unit="%",
                    reference_range="95 - 100%",
                    report_date=report_date,
                    confidence=0.95,
                    is_uncertain=is_unreliable,
                    uncertainty_reason="Implausible SpO2 percentage" if is_unreliable else None
                ))
                vitals["spo2_pct"] = int(spo2_val)
            except Exception:
                pass

        # 5. Temperature
        temp_match = re.search(r"\b(?:temp|temperature)\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:F|deg F|C)?\b", text, re.IGNORECASE)
        if temp_match:
            try:
                t_val = float(temp_match.group(1))
                # Convert Celsius if in 35-43 range
                if 34.0 <= t_val <= 44.0:
                    t_val = round((t_val * 9.0 / 5.0) + 32.0, 1)
                is_unreliable = t_val < 85.0 or t_val > 108.0
                facts.append(StructuredMedicalFact(
                    measurement_name="Body Temperature",
                    numeric_value=t_val,
                    raw_value=temp_match.group(0),
                    unit="°F",
                    reference_range="97.0 - 99.0 °F",
                    report_date=report_date,
                    confidence=0.92,
                    is_uncertain=is_unreliable,
                    uncertainty_reason="Implausible body temperature" if is_unreliable else None
                ))
                vitals["temp_f"] = t_val
            except Exception:
                pass

        # 6. Hemoglobin (Hb)
        hb_match = re.search(r"\b(?:hemoglobin|hb)\s*[:=]?\s*(\d{1,2}(?:\.\d+)?)\s*(?:g/dL|gm/dl)?\b", text, re.IGNORECASE)
        if hb_match:
            try:
                hb_val = float(hb_match.group(1))
                facts.append(StructuredMedicalFact(
                    measurement_name="Hemoglobin",
                    numeric_value=hb_val,
                    raw_value=hb_match.group(0),
                    unit="g/dL",
                    reference_range="12.0 - 16.0 g/dL",
                    report_date=report_date,
                    confidence=0.94,
                    is_uncertain=hb_val < 3.0 or hb_val > 25.0
                ))
            except Exception:
                pass

        # 7. Blood Glucose / Sugar
        glu_match = re.search(r"\b(?:glucose|blood\s*sugar|rbs|fbs)\s*[:=]?\s*(\d{2,3})\s*(?:mg/dL)?\b", text, re.IGNORECASE)
        if glu_match:
            try:
                glu_val = float(glu_match.group(1))
                facts.append(StructuredMedicalFact(
                    measurement_name="Blood Glucose",
                    numeric_value=glu_val,
                    raw_value=glu_match.group(0),
                    unit="mg/dL",
                    reference_range="70 - 110 mg/dL",
                    report_date=report_date,
                    confidence=0.93,
                    is_uncertain=glu_val < 20 or glu_val > 800
                ))
            except Exception:
                pass

        # 8. Respiratory Rate (RR)
        rr_match = re.search(r"\b(?:respiratory\s*rate|resp\s*rate|rr|respiration)\s*[:=]?\s*(\d{1,2})\s*(?:breaths/min|bpm|/min)?\b", text, re.IGNORECASE)
        if rr_match:
            try:
                rr_val = float(rr_match.group(1))
                is_unreliable = rr_val < 6 or rr_val > 65
                facts.append(StructuredMedicalFact(
                    measurement_name="Respiratory Rate",
                    numeric_value=rr_val,
                    raw_value=rr_match.group(0),
                    unit="breaths/min",
                    reference_range="12 - 20 breaths/min",
                    report_date=report_date,
                    confidence=0.92,
                    is_uncertain=is_unreliable,
                    uncertainty_reason="Implausible respiratory rate" if is_unreliable else None
                ))
                vitals["respiratory_rate"] = int(rr_val)
            except Exception:
                pass

        # 9. White Blood Cell Count (WBC / TLC)
        wbc_match = re.search(r"\b(?:wbc|tlc|total\s*leukocyte\s*count|white\s*blood\s*cells?)\s*[:=]?\s*(\d{1,2}(?:,\d{3})*(?:\.\d+)?|\d{3,6})\s*(?:/\s*(?:cumm|mcL|uL|mm3))?\b", text, re.IGNORECASE)
        if wbc_match:
            try:
                raw_wbc_str = wbc_match.group(1).replace(",", "")
                wbc_val = float(raw_wbc_str)
                if wbc_val < 100:
                    wbc_val = wbc_val * 1000
                facts.append(StructuredMedicalFact(
                    measurement_name="White Blood Cell Count (WBC)",
                    numeric_value=wbc_val,
                    raw_value=wbc_match.group(0),
                    unit="cells/mcL",
                    reference_range="4,000 - 11,000 /mcL",
                    report_date=report_date,
                    confidence=0.94,
                    is_uncertain=wbc_val < 500 or wbc_val > 150000
                ))
            except Exception:
                pass

        # 10. Platelet Count (PLT)
        plt_match = re.search(r"\b(?:platelets?|platelet\s*count|plt)\s*[:=]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d{2,6})\s*(?:/\s*(?:cumm|mcL|uL|mm3)|lakhs?|k)?\b", text, re.IGNORECASE)
        if plt_match:
            try:
                raw_plt_str = plt_match.group(1).replace(",", "")
                plt_val = float(raw_plt_str)
                if plt_val < 10:
                    plt_val = plt_val * 100000
                elif plt_val < 1000:
                    plt_val = plt_val * 1000
                facts.append(StructuredMedicalFact(
                    measurement_name="Platelet Count",
                    numeric_value=plt_val,
                    raw_value=plt_match.group(0),
                    unit="/mcL",
                    reference_range="150,000 - 450,000 /mcL",
                    report_date=report_date,
                    confidence=0.94,
                    is_uncertain=plt_val < 5000 or plt_val > 2000000
                ))
            except Exception:
                pass

        # 11. Serum Creatinine
        creat_match = re.search(r"\b(?:creatinine|serum\s*creatinine)\s*[:=]?\s*(\d{1,2}(?:\.\d+)?)\s*(?:mg/dL)?\b", text, re.IGNORECASE)
        if creat_match:
            try:
                creat_val = float(creat_match.group(1))
                facts.append(StructuredMedicalFact(
                    measurement_name="Serum Creatinine",
                    numeric_value=creat_val,
                    raw_value=creat_match.group(0),
                    unit="mg/dL",
                    reference_range="0.6 - 1.2 mg/dL",
                    report_date=report_date,
                    confidence=0.93,
                    is_uncertain=creat_val < 0.1 or creat_val > 25.0
                ))
            except Exception:
                pass

        # 12. Clinical Symptoms / Findings in Report
        symptom_patterns = [
            ("Chest Pain / Angina", r"\b(?:chest\s*pain|angina|cardiac\s*ischemia)\b"),
            ("Shortness of Breath / Dyspnea", r"\b(?:shortness\s*of\s*breath|dyspnea|breathlessness|difficulty\s*breathing)\b"),
            ("High Fever / Chills", r"\b(?:high\s*(?:grade\s*)?fever|chills|rigors|pyrexia)\b"),
            ("Severe Headache", r"\b(?:severe\s*headache|migraine|thunderclap\s*headache)\b"),
            ("Dizziness / Syncope", r"\b(?:dizziness|syncope|fainting|vertigo)\b"),
            ("Tachycardia / Palpitations", r"\b(?:tachycardia|palpitations|arrhythmia)\b"),
            ("Abdominal Pain", r"\b(?:abdominal\s*pain|acute\s*abdomen)\b")
        ]
        for sym_name, sym_pat in symptom_patterns:
            m = re.search(sym_pat, text, re.IGNORECASE)
            if m:
                facts.append(StructuredMedicalFact(
                    measurement_name=f"Reported Finding: {sym_name}",
                    raw_value=m.group(0),
                    report_date=report_date,
                    confidence=0.90,
                    is_uncertain=False
                ))

        return facts, vitals

    def _get_paddle_ocr(self, lang: str = "en"):
        """Lazy loader for PaddleOCR to avoid startup delays or unnecessary memory usage."""
        if self._paddle_ocr is not None:
            return self._paddle_ocr
        try:
            from paddleocr import PaddleOCR
            logger.info("Initializing PaddleOCR engine...")
            self._paddle_ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
            return self._paddle_ocr
        except Exception as e:
            logger.warning(f"PaddleOCR is unavailable or failed to initialize: {e}")
            return None

    def _extract_with_paddle(self, image: Image.Image) -> Tuple[Optional[str], float]:
        """Runs PaddleOCR on PIL image, returning concatenated text and average confidence."""
        engine = self._get_paddle_ocr()
        if engine is None:
            return None, 0.0

        try:
            import numpy as np
            img_arr = np.array(image)
            result = engine.ocr(img_arr, cls=True)

            if not result or not result[0]:
                return None, 0.0

            lines = []
            scores = []
            for line_info in result[0]:
                if len(line_info) >= 2 and len(line_info[1]) >= 2:
                    text_str, score = line_info[1][0], line_info[1][1]
                    if text_str and text_str.strip():
                        lines.append(text_str.strip())
                        scores.append(float(score))

            if not lines:
                return None, 0.0

            avg_score = round(sum(scores) / len(scores), 3) if scores else 0.8
            return "\n".join(lines), avg_score
        except Exception as e:
            logger.warning(f"PaddleOCR execution failed: {e}")
            return None, 0.0

    def _extract_pdf_via_ocr(self, file_bytes: bytes) -> Tuple[str, str, float]:
        """
        Rasterizes PDF pages to high-resolution images via pypdfium2
        and runs multi-tier native OCR on each page.
        """
        pages_ocr_text = []
        confidences = []
        last_engine = "winocr"

        try:
            import pypdfium2 as pdfium
            pdf = pdfium.PdfDocument(file_bytes)
            max_pages = min(len(pdf), 8)
            for i in range(max_pages):
                page = pdf[i]
                pil_image = page.render(scale=2.0).to_pil()

                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                img_bytes = buf.getvalue()

                page_text, eng, conf = self.extract_from_image(img_bytes)
                if page_text and not page_text.startswith("[Image uploaded:"):
                    pages_ocr_text.append(f"--- Page {i+1} ---\n{page_text.strip()}")
                    confidences.append(conf)
                    last_engine = eng

            if pages_ocr_text:
                avg_conf = round(sum(confidences) / len(confidences), 2) if confidences else 0.90
                return "\n\n".join(pages_ocr_text), f"pdf_{last_engine}", avg_conf
        except Exception as e:
            logger.warning(f"pypdfium2 PDF page OCR failed: {e}")

        return (
            "[PDF document uploaded: text could not be extracted automatically. Please verify or manually type key findings.]",
            "pdf_fallback_failed",
            0.0
        )

    def extract_from_pdf(self, file_bytes: bytes) -> Tuple[str, str, float]:
        """
        Extracts text content from a PDF document.
        Tries pypdf direct embedded text extraction first.
        If the PDF contains replacement glyphs (\ufffd), low alphanumeric density (< 0.60),
        or scanned pages, seamlessly falls back to pypdfium2 high-res rasterization + native OCR.
        """
        needs_ocr = False
        raw_pages_text = []

        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            for idx, page in enumerate(reader.pages):
                txt = page.extract_text() or ""
                if txt.strip():
                    raw_pages_text.append((idx + 1, txt.strip()))

            full_combined = " ".join(t[1] for t in raw_pages_text)

            if not raw_pages_text or len(full_combined.strip()) < 15:
                needs_ocr = True
            else:
                ufffd_count = full_combined.count("\ufffd")
                if ufffd_count >= 3:
                    logger.info(f"PDF contains {ufffd_count} replacement characters (\ufffd). Routing to OCR rendering...")
                    needs_ocr = True
                else:
                    non_ws = [c for c in full_combined if not c.isspace()]
                    if non_ws:
                        alnum_count = sum(1 for c in non_ws if c.isalnum())
                        ratio = alnum_count / len(non_ws)
                        if ratio < 0.60:
                            logger.info(f"PDF text has low alphanumeric ratio ({ratio:.2f}). Routing to OCR rendering...")
                            needs_ocr = True

            if not needs_ocr and raw_pages_text:
                formatted = "\n\n".join(f"--- Page {pg} ---\n{txt}" for pg, txt in raw_pages_text)
                return formatted, "pypdf", 0.95

        except Exception as e:
            logger.warning(f"Direct pypdf extraction error: {e}. Falling back to OCR rendering...")
            needs_ocr = True

        return self._extract_pdf_via_ocr(file_bytes)

    def extract_from_image(self, file_bytes: bytes) -> Tuple[str, str, float]:
        """
        Extracts text from an image with multi-tier engine support:
        1. winocr (Windows Native)
        2. PaddleOCR (Multilingual fallback)
        3. pytesseract (Legacy fallback)
        """
        try:
            image = Image.open(io.BytesIO(file_bytes))
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")

            engine_choice = settings.OCR_ENGINE.lower()

            # Path 1: If PaddleOCR is explicitly requested, try it first
            if engine_choice == "paddleocr":
                paddle_text, paddle_conf = self._extract_with_paddle(image)
                if paddle_text:
                    return paddle_text, "paddleocr", paddle_conf

            # Path 2: Try native Windows OCR (winocr)
            try:
                import winocr
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    ocr_res = executor.submit(winocr.recognize_pil_sync, image).result(timeout=15)

                if isinstance(ocr_res, dict) and "text" in ocr_res and ocr_res["text"].strip():
                    return ocr_res["text"].strip(), "winocr", 0.92
                elif hasattr(ocr_res, "text") and ocr_res.text and ocr_res.text.strip():
                    return ocr_res.text.strip(), "winocr", 0.92
            except Exception as ocr_err:
                logger.warning(f"Native winocr recognition error: {ocr_err}")

            # Path 3: Fallback to PaddleOCR if winocr failed or was not selected
            if engine_choice != "winocr":
                paddle_text, paddle_conf = self._extract_with_paddle(image)
                if paddle_text:
                    return paddle_text, "paddleocr", paddle_conf

            # Path 4: Fallback to pytesseract if available
            try:
                import pytesseract
                text = pytesseract.image_to_string(image).strip()
                if text:
                    return text, "pytesseract", 0.85
            except Exception:
                pass

            return (
                "[Image uploaded: text could not be extracted automatically. Please verify or manually type key findings.]",
                "none",
                0.0
            )

        except Exception as e:
            logger.error(f"Error processing image for OCR: {e}")
            return f"[Error reading image file: {str(e)}]", "error", 0.0

    def extract_with_metadata(self, file_bytes: bytes, filename: str) -> DocumentExtractionResult:
        """
        Validates document, extracts text, performs PII masking, classifies document type,
        and extracts structured medical facts.
        """
        # 0. Empty input check
        if not file_bytes or len(file_bytes) == 0:
            return DocumentExtractionResult(
                filename=filename,
                extracted_text="",
                char_count=0,
                engine_used="none",
                confidence=0.0,
                is_medical_document=False,
                detected_document_type="empty"
            )

        # 1. Validation check
        is_safe, error_msg = safety_service.validate_file_upload(filename, file_bytes)
        if not is_safe:
            return DocumentExtractionResult(
                filename=filename,
                extracted_text=f"[Upload Error]: {error_msg}",
                char_count=0,
                engine_used="validation_rejected",
                confidence=0.0,
                source_type="ocr_extraction",
                is_medical_document=False,
                medical_document_confidence=0.0,
                detected_document_type="invalid_upload"
            )

        ext = filename.lower().split(".")[-1] if "." in filename else ""

        if ext == "pdf":
            raw_text, engine, conf = self.extract_from_pdf(file_bytes)
        elif ext in ["png", "jpg", "jpeg", "webp", "bmp", "tiff"]:
            raw_text, engine, conf = self.extract_from_image(file_bytes)
        elif ext in ["txt", "csv", "json"]:
            try:
                raw_text = file_bytes.decode("utf-8", errors="replace").strip()
            except Exception:
                raw_text = file_bytes.decode("latin-1", errors="replace").strip()
            engine = "plain_text"
            conf = 1.0
        else:
            raw_text, engine, conf = self.extract_from_image(file_bytes)

        # 2. Medical Document Classification
        is_med, med_conf, doc_type = self.detect_medical_document(raw_text)

        # 3. Structured Fact Extraction (Vitals, Lab tests, Reference ranges, Dates)
        structured_facts, extracted_vitals = self.extract_structured_medical_facts(raw_text)

        # 4. PII Masking on Extracted Clinical Text
        mask_res = safety_service.mask_pii(raw_text)
        safe_text = mask_res.masked_text

        return DocumentExtractionResult(
            filename=filename,
            extracted_text=safe_text,
            char_count=len(safe_text),
            engine_used=engine,
            confidence=conf,
            source_type="ocr_extraction",
            is_mock=(settings.MEDGEMMA_MODE == "mock"),
            is_medical_document=is_med,
            medical_document_confidence=med_conf,
            detected_document_type=doc_type,
            structured_facts=structured_facts,
            extracted_vitals=extracted_vitals,
            pii_redaction_count=mask_res.redaction_count
        )

    def extract_text(self, file_bytes: bytes, filename: str) -> str:
        """
        Backward-compatible method returning plain extracted text string.
        """
        res = self.extract_with_metadata(file_bytes, filename)
        return res.extracted_text


ocr_service = MedicalDocumentOCRService()
detect_medical_document = MedicalDocumentOCRService.detect_medical_document
extract_structured_medical_facts = MedicalDocumentOCRService.extract_structured_medical_facts


extract_from_pdf = ocr_service.extract_from_pdf
extract_from_image = ocr_service.extract_from_image
