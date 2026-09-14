import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.tts.piper_backend import PiperTTS


class PiperConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = Path(self.temp.name) / "de_DE-thorsten-medium.onnx"
        self.model.touch()

    def load_voice(self, configured=""):
        backend = PiperTTS(str(self.model), str(configured))
        piper = Mock()
        with patch.dict("sys.modules", {"piper": piper}):
            backend.load()
        self.assertEqual(piper.PiperVoice.load.call_args.kwargs["config_path"], str(backend.config_path))
        return backend

    def test_prefixed_download_replaces_missing_configured_path(self):
        config = self.model.with_name("de_de_DE_thorsten_medium_" + self.model.name + ".json")
        config.touch()
        backend = self.load_voice(str(self.model) + ".json")
        self.assertEqual(backend.config_path, config)

    def test_standard_name_wins_over_prefixed_download(self):
        standard = Path(str(self.model) + ".json")
        standard.touch()
        self.model.with_name("prefix_" + standard.name).touch()
        self.assertEqual(self.load_voice().config_path, standard)

    def test_explicit_existing_path_is_respected(self):
        custom = self.model.with_name("custom.json")
        custom.touch()
        self.assertEqual(self.load_voice(custom).config_path, custom)

    def test_unrelated_json_is_not_used(self):
        self.model.with_name("another-voice.onnx.json").touch()
        with self.assertRaisesRegex(FileNotFoundError, "tts.piper_config"):
            self.load_voice()

    def test_ambiguous_downloads_require_explicit_path(self):
        for prefix in ("first_", "second_"):
            self.model.with_name(prefix + self.model.name + ".json").touch()
        with self.assertRaisesRegex(ValueError, "Multiple Piper configurations"):
            self.load_voice()
