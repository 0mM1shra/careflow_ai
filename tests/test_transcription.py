import os
import io
import pytest
from pathlib import Path
from backend.services.transcription import TranscriptionService, TranscriptionError
from backend.config import settings


def test_transcribe_audio_endpoint_success(client):
    """Test POST /transcribe-audio returns a valid transcript and metadata without auto-committing."""
    fake_audio_bytes = b"RIFF....WAVEfmt ...." + b"\x00" * 1024
    files = {"file": ("symptoms.wav", fake_audio_bytes, "audio/wav")}
    params = {"language": "English", "model_size": "base"}

    response = client.post("/transcribe-audio", files=files, params=params)
    assert response.status_code == 200
    res_data = response.json()

    assert "transcript" in res_data
    assert len(res_data["transcript"]) > 0
    assert "detected_language" in res_data
    assert "confidence" in res_data
    assert res_data["confidence"] > 0.0
    assert res_data["model_size"] == "base"
    assert res_data["is_mock"] is True


def test_transcribe_audio_endpoint_hindi(client):
    """Test POST /transcribe-audio handles Hindi language parameter."""
    fake_audio_bytes = b"RIFF....WAVEfmt ...." + b"\x00" * 512
    files = {"file": ("hindi_speech.mp3", fake_audio_bytes, "audio/mpeg")}
    params = {"language": "Hindi", "model_size": "tiny"}

    response = client.post("/transcribe-audio", files=files, params=params)
    assert response.status_code == 200
    res_data = response.json()

    assert res_data["detected_language"] in ["hi", "Hindi"]
    assert res_data["model_size"] == "tiny"


def test_transcribe_audio_endpoint_empty_file(client):
    """Test POST /transcribe-audio rejects empty audio files with 400."""
    files = {"file": ("empty.wav", b"", "audio/wav")}
    response = client.post("/transcribe-audio", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_transcribe_audio_endpoint_invalid_format(client):
    """Test POST /transcribe-audio rejects unsupported audio formats."""
    files = {"file": ("malicious.exe", b"MZ......", "application/octet-stream")}
    response = client.post("/transcribe-audio", files=files)
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()


def test_transcribe_service_cleanup_and_tempfile():
    """Verify transcription service cleans up temporary audio files safely."""
    service = TranscriptionService()
    fake_audio = b"RIFFFAKEAUDIOBYTES" * 50
    temp_dir = Path(settings.CAREFLOW_TEMP_DIR)

    # Count files in temp_dir before
    before_files = set(temp_dir.glob("careflow_audio_*"))

    result = service.transcribe_audio(
        file_bytes=fake_audio,
        filename="test_recording.wav",
        language="English",
        model_size="base"
    )

    # Count files in temp_dir after
    after_files = set(temp_dir.glob("careflow_audio_*"))

    # Assert no leaked files
    assert after_files == before_files
    assert result.confidence is not None
    assert result.confidence > 0.5


def test_transcribe_service_size_limit_rejection():
    """Verify transcription service enforces maximum file size limit."""
    service = TranscriptionService()
    huge_bytes = b"0" * (26 * 1024 * 1024)  # 26 MB

    with pytest.raises(TranscriptionError) as exc_info:
        service.transcribe_audio(
            file_bytes=huge_bytes,
            filename="large_file.wav"
        )
    assert "exceeds" in str(exc_info.value).lower()


def test_hosted_transcription_groq_and_hf_mocked():
    """Verify hosted transcription parses Groq / OpenAI compatible and Hugging Face responses."""
    from unittest.mock import patch, MagicMock

    service = TranscriptionService()
    fake_audio = b"RIFFFAKEAUDIOBYTES"

    # Test Groq / OpenAI compatible response
    with patch.object(settings, "WHISPER_ENDPOINT", "https://api.groq.com/openai/v1/audio/transcriptions"):
        with patch.object(settings, "WHISPER_API_KEY", "gsk_test_api_key"):
            with patch("requests.post") as mock_post:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"text": "Mujhe do din se bukhar aur ulti hai"}
                mock_post.return_value = mock_resp

                res = service._hosted_transcribe(
                    file_bytes=fake_audio,
                    filename="voice.wav",
                    language="Hindi"
                )
                assert res is not None
                assert "bukhar" in res.transcript
                assert res.is_mock is False
                assert res.detected_language == "hi"

    # Test Hugging Face response
    with patch.object(settings, "WHISPER_ENDPOINT", "https://api-inference.huggingface.co/models/openai/whisper-large-v3"):
        with patch.object(settings, "WHISPER_API_KEY", "hf_test_token"):
            with patch("requests.post") as mock_post:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.json.return_value = {"text": "I have severe chest pressure"}
                mock_post.return_value = mock_resp

                res = service._hosted_transcribe(
                    file_bytes=fake_audio,
                    filename="voice.wav",
                    language="English"
                )
                assert res is not None
                assert "chest pressure" in res.transcript
                assert res.model_size == "hosted-huggingface"
