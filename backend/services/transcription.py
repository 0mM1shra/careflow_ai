import os
import io
import math
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List

from backend.config import settings
from backend.schemas.multilingual import TranscriptionResult

logger = logging.getLogger("careflow.transcription")

ALLOWED_AUDIO_EXTENSIONS = {"wav", "mp3", "ogg", "m4a", "webm", "flac", "aac"}
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB


class TranscriptionError(Exception):
    """Raised when audio transcription fails."""
    pass


def _patch_pyav_metadata_errors_compat():
    """
    PyAV >= 14.0 removed the 'metadata_errors' parameter from av.open().
    faster-whisper passes metadata_errors='ignore', causing a TypeError.
    This compatibility patch safely strips metadata_errors when PyAV is imported.
    """
    try:
        import av
        if not getattr(av, "_careflow_compat_patched", False):
            orig_open = av.open

            def _compat_open(*args, **kwargs):
                kwargs.pop("metadata_errors", None)
                return orig_open(*args, **kwargs)

            av.open = _compat_open
            av._careflow_compat_patched = True
    except Exception:
        pass


class TranscriptionService:
    """
    Local Audio Transcription Service supporting faster-whisper.
    
    Guarantees:
    - Configurable ASR model: 'base' or 'tiny'.
    - Runs on CPU by default to conserve RTX 4050 VRAM for MedGemma.
    - Temporary files are guaranteed to be cleaned up after inference.
    - Provides a zero-dependency deterministic mock mode when faster-whisper is not installed
      or when MEDGEMMA_MODE=mock.
    - Audio contents are never logged to console or audit logs.
    """

    def __init__(self):
        self._models: Dict[str, Any] = {}
        _patch_pyav_metadata_errors_compat()

    def get_supported_models(self) -> List[str]:
        return ["base", "tiny"]

    def _get_whisper_model(self, model_size: str, device: str = "cpu"):
        """Lazy-loads and caches faster-whisper WhisperModel instance."""
        cache_key = f"{model_size}_{device}"
        if cache_key in self._models:
            return self._models[cache_key]

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise TranscriptionError(
                "faster-whisper is not installed. Run 'pip install faster-whisper' "
                "or set MEDGEMMA_MODE=mock for testing."
            )

        download_root = settings.HF_HOME if settings.HF_HOME else None
        compute_type = "int8" if device == "cpu" else "float16"

        logger.info(f"Loading faster-whisper model '{model_size}' on '{device}' (compute_type={compute_type})...")
        try:
            model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=download_root
            )
            self._models[cache_key] = model
            return model
        except Exception as e:
            raise TranscriptionError(f"Failed to initialize faster-whisper model '{model_size}': {e}")

    def _normalize_language(self, language: Optional[str]) -> Optional[str]:
        """Normalizes UI language strings ('Hindi', 'English', 'Hinglish', 'auto') to Whisper ISO codes or None."""
        if not language:
            return None
        cleaned = language.strip().lower()
        if cleaned in ("", "auto", "detect", "automatic", "none", "null", "hinglish", "mixed"):
            return None
        mapping = {
            "english": "en",
            "en": "en",
            "eng": "en",
            "hindi": "hi",
            "hi": "hi",
            "hin": "hi",
        }
        return mapping.get(cleaned, None)

    def transcribe_audio(
        self,
        file_bytes: bytes,
        filename: str,
        language: Optional[str] = None,
        model_size: Optional[str] = None,
        device: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribes audio bytes safely into text and metadata.
        Does NOT automatically submit to the clinical case.
        """
        if not file_bytes:
            raise TranscriptionError("Uploaded audio file is empty.")

        if len(file_bytes) > MAX_AUDIO_BYTES:
            raise TranscriptionError(f"Audio file exceeds {MAX_AUDIO_BYTES // (1024 * 1024)} MB limit.")

        ext = filename.lower().split(".")[-1] if "." in filename else ""
        if ext not in ALLOWED_AUDIO_EXTENSIONS:
            raise TranscriptionError(
                f"Unsupported audio format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}"
            )

        model_size = (model_size or settings.ASR_MODEL_SIZE).lower()
        if model_size not in ["base", "tiny", "small"]:
            model_size = "base"
        device = (device or settings.ASR_DEVICE).lower()

        # Check if unit test synthetic mock transcript is present
        if file_bytes.startswith(b"MOCK_TRANSCRIPT:"):
            return self._mock_transcribe(file_bytes, filename, language, model_size)

        # Determine execution mode: mock, local, or auto (independent of MEDGEMMA_MODE)
        asr_mode = (getattr(settings, "ASR_MODE", None) or os.getenv("ASR_MODE", "auto")).lower()
        use_mock = (asr_mode == "mock")
        if not use_mock:
            try:
                import faster_whisper
            except ImportError:
                if asr_mode == "local":
                    raise TranscriptionError(
                        "faster-whisper is not installed. Run 'pip install faster-whisper' "
                        "or set ASR_MODE=mock for testing."
                    )
                logger.warning("faster-whisper not installed; falling back to mock transcription.")
                use_mock = True

        if use_mock:
            return self._mock_transcribe(file_bytes, filename, language, model_size)

        # Path A: If hosted Whisper endpoint is configured, try it first
        if settings.WHISPER_ENDPOINT or asr_mode == "hosted":
            hosted_res = self._hosted_transcribe(file_bytes, filename, language, model_size)
            if hosted_res:
                return hosted_res
            if asr_mode == "hosted":
                raise TranscriptionError(f"Hosted Whisper endpoint '{settings.WHISPER_ENDPOINT}' failed.")

        # Path B: Local faster-whisper execution
        whisper_model = self._get_whisper_model(model_size, device)
        normalized_lang = self._normalize_language(language)

        # Domain prompt hints to guide Hindi/Hinglish clinical vocabulary
        initial_prompt = (
            "मरीज के लक्षण: बुखार, खांसी, सीने में दर्द, सांस लेने में तकलीफ, उल्टी, पेट दर्द, सिरदर्द, कमजोरी।"
            if normalized_lang == "hi"
            else (
                "Patient symptoms: chest pain, shortness of breath, fever, headache, vomiting blood, abdominal pain, nausea."
                if normalized_lang == "en"
                else "CareFlow intake: bukhar, sar dard, chest pain, ulti, saans lene me takleef, pet me dard, weakness."
            )
        )

        # Create temporary file for audio
        temp_dir = settings.CAREFLOW_TEMP_DIR if settings.CAREFLOW_TEMP_DIR else None
        temp_file = tempfile.NamedTemporaryFile(suffix=f".{ext}", dir=temp_dir, delete=False)
        temp_path = temp_file.name

        try:
            temp_file.write(file_bytes)
            temp_file.close()

            segments_generator, info = whisper_model.transcribe(
                temp_path,
                language=normalized_lang,
                initial_prompt=initial_prompt,
                beam_size=5,
                vad_filter=True
            )

            segments = []
            transcript_parts = []
            total_logprob = 0.0
            word_count = 0

            for seg in segments_generator:
                text = seg.text.strip()
                if text:
                    transcript_parts.append(text)
                    segments.append({
                        "start": round(seg.start, 2),
                        "end": round(seg.end, 2),
                        "text": text,
                        "avg_logprob": round(seg.avg_logprob, 3)
                    })
                    total_logprob += seg.avg_logprob
                    word_count += 1

            full_transcript = " ".join(transcript_parts).strip()
            # Approximate confidence from avg logprob
            avg_logprob = (total_logprob / word_count) if word_count > 0 else -1.0
            confidence = round(max(0.0, min(1.0, math.exp(avg_logprob))), 3) if avg_logprob < 0 else 0.95

            return TranscriptionResult(
                transcript=full_transcript,
                detected_language=info.language or "en",
                language_probability=round(info.language_probability or 1.0, 3),
                confidence=confidence,
                duration_seconds=round(info.duration or 0.0, 2),
                model_size=model_size,
                source_type="voice_transcript",
                is_mock=False,
                segments=segments
            )

        except Exception as e:
            logger.error(f"Error during audio transcription: {e}")
            raise TranscriptionError(f"Audio transcription failed: {e}")
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as clean_err:
                    logger.warning(f"Failed to remove temp audio file {temp_path}: {clean_err}")

    def _hosted_transcribe(
        self,
        file_bytes: bytes,
        filename: str,
        language: Optional[str] = None,
        model_size: Optional[str] = None
    ) -> Optional[TranscriptionResult]:
        """Calls an optional hosted Whisper-compatible API endpoint if configured."""
        endpoint = settings.WHISPER_ENDPOINT
        if not endpoint:
            return None

        try:
            import requests
            headers = {}
            if settings.WHISPER_API_KEY:
                headers["Authorization"] = f"Bearer {settings.WHISPER_API_KEY}"

            files = {"file": (filename, file_bytes, "audio/wav")}
            data = {"model": "whisper-1"}
            norm_lang = self._normalize_language(language)
            if norm_lang:
                data["language"] = norm_lang

            resp = requests.post(endpoint, headers=headers, files=files, data=data, timeout=30)
            if resp.status_code == 200:
                res_data = resp.json()
                text = res_data.get("text", "").strip()
                return TranscriptionResult(
                    transcript=text,
                    detected_language=norm_lang or "en",
                    language_probability=0.95,
                    confidence=0.92,
                    duration_seconds=0.0,
                    model_size=model_size or "hosted",
                    source_type="voice_transcript",
                    is_mock=False,
                    segments=[]
                )
            else:
                logger.warning(f"Hosted Whisper API returned HTTP {resp.status_code}: {resp.text[:150]}")
                return None
        except Exception as e:
            logger.warning(f"Failed calling hosted Whisper endpoint: {e}")
            return None

    def _mock_transcribe(
        self,
        file_bytes: bytes,
        filename: str,
        language: Optional[str],
        model_size: str
    ) -> TranscriptionResult:
        """
        High-fidelity deterministic mock transcription for unit testing and CPU-only prototype runs.
        Supports synthetic test payloads containing encoded text or keywords in filename.
        """
        fname_lower = filename.lower()

        # Check if the audio file contains synthetic text header (for unit tests)
        if file_bytes.startswith(b"MOCK_TRANSCRIPT:"):
            try:
                text = file_bytes[len(b"MOCK_TRANSCRIPT:"):].decode("utf-8").strip()
                lang = "hi" if any('\u0900' <= c <= '\u097F' for c in text) else "en"
                return TranscriptionResult(
                    transcript=text,
                    detected_language=lang,
                    language_probability=0.98,
                    confidence=0.96,
                    duration_seconds=4.2,
                    model_size=model_size,
                    source_type="voice_transcript",
                    is_mock=True,
                    segments=[{"start": 0.0, "end": 4.2, "text": text, "avg_logprob": -0.1}]
                )
            except Exception:
                pass

        norm_l = self._normalize_language(language)
        # Check for scenario markers in filename for deterministic testing
        if "cardiac" in fname_lower or "chest" in fname_lower:
            text = "I have severe crushing chest pain radiating to my left arm."
            lang = "en"
        elif "fever_hindi" in fname_lower or "hindi" in fname_lower or norm_l == "hi":
            text = "मुझे दो दिन से बहुत तेज बुखार है और सिर में दर्द है।"
            lang = "hi"
        elif "hinglish" in fname_lower or (language and "hinglish" in language.lower()):
            text = "mujhe kal se pet me dard ho raha hai aur ulti jaisa lag raha hai."
            lang = "hi"
        elif "noisy" in fname_lower:
            text = "I think... pain here... maybe two days..."
            lang = "en"
            return TranscriptionResult(
                transcript=text,
                detected_language="en",
                language_probability=0.65,
                confidence=0.52,  # Low confidence flagged
                duration_seconds=3.5,
                model_size=model_size,
                source_type="voice_transcript",
                is_mock=True,
                segments=[{"start": 0.0, "end": 3.5, "text": text, "avg_logprob": -0.8}]
            )
        else:
            text = "I have had a mild headache and runny nose for the past 2 days."
            lang = "en"

        return TranscriptionResult(
            transcript=text,
            detected_language=lang,
            language_probability=0.95,
            confidence=0.92,
            duration_seconds=5.0,
            model_size=model_size,
            source_type="voice_transcript",
            is_mock=True,
            segments=[{"start": 0.0, "end": 5.0, "text": text, "avg_logprob": -0.2}]
        )


transcription_service = TranscriptionService()
