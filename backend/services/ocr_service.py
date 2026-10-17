import io
import logging
from typing import Optional
from PIL import Image
import pypdf

logger = logging.getLogger("careflow.ocr")

class MedicalDocumentOCRService:
    """
    Extracts text from uploaded medical documents (PDFs, prescription photos, lab images)
    before feeding them into MedGemma clinical extraction.
    """

    @staticmethod
    def extract_from_pdf(file_bytes: bytes) -> str:
        """Extracts text content from a PDF document using pypdf."""
        try:
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pages_text = []
            for idx, page in enumerate(reader.pages):
                txt = page.extract_text()
                if txt and txt.strip():
                    pages_text.append(f"--- Page {idx+1} ---\n{txt.strip()}")
            
            if not pages_text:
                return "[Note: PDF contains scanned pages or images without embedded text layer.]"
            return "\n\n".join(pages_text)
        except Exception as e:
            logger.error(f"Error parsing PDF document: {e}")
            return f"[Error parsing PDF: {str(e)}]"

    @staticmethod
    def extract_from_image(file_bytes: bytes) -> str:
        """Extracts text from an image using Windows native OCR (winocr) with fallback."""
        try:
            image = Image.open(io.BytesIO(file_bytes))
            # Convert to RGB if needed (e.g. RGBA or palette)
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            
            try:
                import winocr
                import concurrent.futures

                # Run winocr in a separate thread so asyncio.run() does not conflict with active event loop
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    ocr_res = executor.submit(winocr.recognize_pil_sync, image).result(timeout=15)

                if isinstance(ocr_res, dict) and "text" in ocr_res:
                    extracted = ocr_res["text"].strip()
                    if extracted:
                        return extracted
                elif hasattr(ocr_res, "text") and ocr_res.text:
                    return ocr_res.text.strip()
            except Exception as ocr_err:
                logger.warning(f"Native winocr recognition error: {ocr_err}")

            # Fallback: check pytesseract if available
            try:
                import pytesseract
                return pytesseract.image_to_string(image).strip()
            except Exception:
                pass

            return "[Image uploaded: text could not be extracted automatically. Please verify or manually type key findings.]"
        except Exception as e:
            logger.error(f"Error processing image for OCR: {e}")
            return f"[Error reading image file: {str(e)}]"

    @classmethod
    def extract_text(cls, file_bytes: bytes, filename: str) -> str:
        """
        Detects document type from filename/extension and performs appropriate text extraction/OCR.
        """
        ext = filename.lower().split(".")[-1] if "." in filename else ""
        
        if ext == "pdf":
            return cls.extract_from_pdf(file_bytes)
        elif ext in ["png", "jpg", "jpeg", "webp", "bmp", "tiff"]:
            return cls.extract_from_image(file_bytes)
        elif ext in ["txt", "csv", "json"]:
            try:
                return file_bytes.decode("utf-8", errors="replace").strip()
            except Exception:
                return file_bytes.decode("latin-1", errors="replace").strip()
        else:
            # Try image first, then text
            try:
                return cls.extract_from_image(file_bytes)
            except Exception:
                return file_bytes.decode("utf-8", errors="replace").strip()

ocr_service = MedicalDocumentOCRService()
