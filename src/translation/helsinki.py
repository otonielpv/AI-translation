"""Spanish → German translation using Helsinki-NLP/opus-mt-es-de.

Provides a TranslationBackend interface so other backends (Argos, etc.)
can be swapped in by changing config.
"""

from __future__ import annotations

import logging
import re
import time

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Church-specific vocabulary correction hook.
# Add tuples of (pattern, replacement) here to post-correct common STT errors
# or to normalise church terminology before translation.
# ---------------------------------------------------------------------------
CHURCH_CORRECTIONS: list[tuple[str, str]] = [
    # Example: (r"\bespíritu santo\b", "Espíritu Santo"),
]

# Glossary entries that should be preserved or translated a specific way.
# Applied AFTER translation as a final pass.
GLOSSARY_DE: dict[str, str] = {
    # "Espíritu Santo": "Heiliger Geist",
}


class TranslationBackend:
    """Abstract interface for translation backends."""

    def translate(self, text: str) -> tuple[str, float]:
        """Return (translated_text, duration_seconds)."""
        raise NotImplementedError


class HelsinkiTranslator(TranslationBackend):
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-es-de"):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None

    def load(self) -> None:
        from transformers import MarianMTModel, MarianTokenizer

        log.info("Loading translation model '%s' …", self.model_name)
        t0 = time.monotonic()
        self._tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self._model = MarianMTModel.from_pretrained(self.model_name)
        log.info("Translation model loaded in %.1fs.", time.monotonic() - t0)

    def _normalize(self, text: str) -> str:
        text = text.strip()
        # Collapse internal whitespace runs
        text = re.sub(r"\s+", " ", text)
        # Apply church corrections
        for pattern, replacement in CHURCH_CORRECTIONS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _apply_glossary(self, text: str) -> str:
        for src, tgt in GLOSSARY_DE.items():
            text = text.replace(src, tgt)
        return text

    def translate(self, text: str) -> tuple[str, float]:
        if self._model is None:
            raise RuntimeError("HelsinkiTranslator not loaded. Call load() first.")

        text = self._normalize(text)
        if not text:
            return "", 0.0

        t0 = time.monotonic()
        inputs = self._tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self._model.generate(**inputs)
        translated = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        translated = self._apply_glossary(translated)
        duration = time.monotonic() - t0
        log.info("[TR] %r → %r  (%.2fs)", text, translated, duration)
        return translated, duration


def build_translator(cfg: dict) -> TranslationBackend:
    backend = cfg.get("translation", {}).get("backend", "helsinki")
    if backend == "helsinki":
        model = cfg.get("translation", {}).get("helsinki_model", "Helsinki-NLP/opus-mt-es-de")
        t = HelsinkiTranslator(model_name=model)
        t.load()
        return t
    raise ValueError(f"Unknown translation backend: {backend!r}")
