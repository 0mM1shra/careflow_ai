import logging
import re
from functools import lru_cache
from typing import Optional

logger = logging.getLogger("careflow.translation")

class ClinicalTranslationService:
    """
    Zero-VRAM Pure Neural Translation Pipeline:
    1. Inbound (Hindi/Hinglish -> English):
       Attempts neural translation of patient messages into English.
       If Romanized Hinglish or if translation equals raw, returns raw text directly
       so MedGemma's native multilingual context encoder decodes it accurately.
    2. Outbound (English -> Devanagari Hindi):
       Dynamically translates the SLM-generated English question using neural translation
       (GoogleTranslator / MyMemoryTranslator) in ~100ms with 0 MB GPU VRAM.
       Never overwrites with canned dictionaries.
    """

    def __init__(self):
        self._google_to_en = None
        self._google_to_hi = None
        self._mymemory_to_en = None
        self._mymemory_to_hi = None
        self._init_translators()

    def _init_translators(self):
        try:
            from deep_translator import GoogleTranslator
            self._google_to_en = GoogleTranslator(source="auto", target="en")
            self._google_to_hi = GoogleTranslator(source="en", target="hi")
        except Exception as e:
            logger.debug(f"GoogleTranslator init: {e}")

        try:
            from deep_translator import MyMemoryTranslator
            self._mymemory_to_en = MyMemoryTranslator(source="hindi", target="english")
            self._mymemory_to_hi = MyMemoryTranslator(source="english", target="hindi")
        except Exception as e:
            logger.debug(f"MyMemoryTranslator init: {e}")

    @lru_cache(maxsize=512)
    def translate_to_english(self, text: str) -> str:
        """
        Translates Hindi or Romanized WhatsApp Hinglish into clean clinical English.
        Zero VRAM, no external API keys required.
        1. If Latin script with Hindi/Hinglish terms, transliterates to Devanagari via Google Input Tools.
        2. Translates Devanagari to English via GoogleTranslator (falling back to MyMemoryTranslator).
        3. If transliteration/translation fails or times out, returns original text so MedGemma's native multilingual encoder parses it.
        """
        if not text or not text.strip():
            return ""

        raw = text.strip()
        is_devanagari = bool(re.search(r'[\u0900-\u097F]', raw))
        devanagari_text = raw if is_devanagari else None

        # 1. If Latin script, attempt transliteration to Devanagari for Hinglish phrases
        if not is_devanagari:
            try:
                import urllib.parse
                import requests
                enc = urllib.parse.quote(raw)
                url = f"https://inputtools.google.com/request?text={enc}&itc=hi-t-i0-und&num=1"
                resp = requests.get(url, timeout=2.5)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, list) and len(data) > 1 and data[1]:
                        candidate = data[1][0][1][0]
                        if candidate and bool(re.search(r'[\u0900-\u097F]', candidate)):
                            devanagari_text = candidate
            except Exception as e:
                logger.debug(f"Input tools transliteration fallback: {e}")

        # 2. If we have Devanagari text, translate to English
        target_to_translate = devanagari_text or raw
        if devanagari_text:
            # Try GoogleTranslator first
            if self._google_to_en:
                try:
                    res = self._google_to_en.translate(target_to_translate)
                    if res and res.strip() and res.strip().lower() != raw.lower():
                        return res.strip()
                except Exception as e:
                    logger.debug(f"Google to EN error: {e}")

            # Fallback to MyMemoryTranslator
            if self._mymemory_to_en:
                try:
                    res = self._mymemory_to_en.translate(target_to_translate)
                    if res and res.strip() and res.strip().lower() != raw.lower():
                        return res.strip()
                except Exception as e:
                    logger.debug(f"MyMemory to EN error: {e}")

        # 3. Fallback to raw text
        return raw

    @lru_cache(maxsize=512)
    def translate_to_hindi(self, english_text: str, default_hindi: Optional[str] = None) -> str:
        """
        Translates an SLM-generated English follow-up question to natural Devanagari Hindi.
        Pure neural translation: ZERO hardcoded if/else dictionaries.
        If offline or translation error, preserves default_hindi from SLM verbatim.
        """
        if not english_text or not english_text.strip():
            return default_hindi or ""

        raw = english_text.strip()

        # 1. Primary: GoogleTranslator
        if self._google_to_hi:
            try:
                res = self._google_to_hi.translate(raw)
                if res and res.strip():
                    return res.strip()
            except Exception as e:
                logger.debug(f"Google EN->HI translation notice: {e}")

        # 2. Secondary: MyMemoryTranslator
        if self._mymemory_to_hi:
            try:
                res = self._mymemory_to_hi.translate(raw)
                if res and res.strip():
                    # Strip any leading numbered bullets if returned by translation memory
                    clean_res = re.sub(r'^\d+[\.\)]\s*', '', res.strip())
                    return clean_res
            except Exception as e:
                logger.debug(f"MyMemory EN->HI translation notice: {e}")

        # 3. Fallback: preserve original model output if provided, or return raw
        return default_hindi or raw

# Global singleton
translation_service = ClinicalTranslationService()
