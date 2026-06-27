import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gemini_srt_translator.main import GeminiSRTTranslator


class InterruptedTranslationTests(unittest.TestCase):
    def test_translation_interruption_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_file = root / "sample.en.srt"
            output_file = root / "sample.zh.srt"
            input_file.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n", encoding="utf-8")

            translator = GeminiSRTTranslator(
                gemini_api_key="test-key",
                target_language="Simplified Chinese",
                input_file=str(input_file),
                output_file=str(output_file),
                model_name="gemini-flash-latest",
                batch_size=1,
                use_colors=False,
                resume=False,
            )

            with (
                patch.object(translator, "getmodels", return_value=["gemini-flash-latest"]),
                patch.object(translator, "_get_token_limit", return_value=None),
                patch.object(translator, "_validate_token_size", return_value=True),
                patch.object(translator, "_process_batch", side_effect=RuntimeError("api failed")),
                patch("gemini_srt_translator.main.time.sleep", return_value=None),
                patch("gemini_srt_translator.main.progress_bar", return_value=None),
                patch("gemini_srt_translator.main.info_with_progress", return_value=None),
                patch("gemini_srt_translator.main.warning_with_progress", return_value=None),
                patch("gemini_srt_translator.main.error_with_progress", return_value=None),
                patch("gemini_srt_translator.main.success_with_progress", return_value=None),
            ):
                with self.assertRaises(SystemExit) as raised:
                    translator.translate()

            self.assertNotEqual(raised.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
