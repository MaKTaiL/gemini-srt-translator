import json
import io
import os
import stat
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from gemini_srt_translator.logger import set_quiet_mode
from gemini_srt_translator.main import GeminiSRTTranslator


class InterruptedTranslationTests(unittest.TestCase):
    def test_resume_keeps_existing_output_on_disk_until_batch_finishes(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            input_file = root / "sample.en.srt"
            output_file = root / "sample.zh.srt"
            progress_file = root / "sample.en.progress"
            input_file.write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n"
                "2\n00:00:03,000 --> 00:00:04,000\nWorld\n",
                encoding="utf-8",
            )
            existing_output = (
                "1\n00:00:01,000 --> 00:00:02,000\n你好\n\n"
                "2\n00:00:03,000 --> 00:00:04,000\nWorld\n"
            )
            output_file.write_text(existing_output, encoding="utf-8")
            progress_file.write_text(json.dumps({"line": 2, "input_file": str(input_file)}), encoding="utf-8")

            translator = GeminiSRTTranslator(
                gemini_api_key="test-key",
                target_language="Simplified Chinese",
                input_file=str(input_file),
                output_file=str(output_file),
                model_name="gemini-flash-latest",
                batch_size=1,
                use_colors=False,
                resume=True,
            )

            def stop_after_checking_output(*args):
                if output_file.read_text(encoding="utf-8") != existing_output:
                    raise SystemExit("output was truncated before batch finished")
                raise SystemExit("output remained intact")

            with (
                patch.object(translator, "getmodels", return_value=["gemini-flash-latest"]),
                patch.object(translator, "_get_token_limit", return_value=None),
                patch.object(translator, "_validate_token_size", return_value=True),
                patch.object(translator, "_process_batch", side_effect=stop_after_checking_output),
                patch("gemini_srt_translator.main.progress_bar", return_value=None),
                patch("gemini_srt_translator.main.info_with_progress", return_value=None),
                patch("gemini_srt_translator.main.warning_with_progress", return_value=None),
                patch("gemini_srt_translator.main.error_with_progress", return_value=None),
                patch("gemini_srt_translator.main.success_with_progress", return_value=None),
                patch("gemini_srt_translator.main.highlight", return_value=None),
            ):
                with self.assertRaises(SystemExit) as raised:
                    translator.translate()

            self.assertEqual(raised.exception.code, "output remained intact")

    def test_resume_context_size_limits_initial_resume_context(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            input_file = root / "sample.en.srt"
            output_file = root / "sample.zh.srt"
            progress_file = root / "sample.en.progress"
            input_file.write_text(
                "1\n00:00:01,000 --> 00:00:02,000\nLine one\n\n"
                "2\n00:00:03,000 --> 00:00:04,000\nLine two\n\n"
                "3\n00:00:05,000 --> 00:00:06,000\nLine three\n\n"
                "4\n00:00:07,000 --> 00:00:08,000\nLine four\n\n"
                "5\n00:00:09,000 --> 00:00:10,000\nLine five\n",
                encoding="utf-8",
            )
            output_file.write_text(
                "1\n00:00:01,000 --> 00:00:02,000\n第一行\n\n"
                "2\n00:00:03,000 --> 00:00:04,000\n第二行\n\n"
                "3\n00:00:05,000 --> 00:00:06,000\n第三行\n\n"
                "4\n00:00:07,000 --> 00:00:08,000\n第四行\n\n"
                "5\n00:00:09,000 --> 00:00:10,000\nLine five\n",
                encoding="utf-8",
            )
            progress_file.write_text(json.dumps({"line": 5, "input_file": str(input_file)}), encoding="utf-8")

            translator = GeminiSRTTranslator(
                gemini_api_key="test-key",
                target_language="Simplified Chinese",
                input_file=str(input_file),
                output_file=str(output_file),
                model_name="gemini-flash-latest",
                batch_size=100,
                resume=True,
                resume_context_size=2,
                use_colors=False,
            )

            def capture_context(batch, previous_message, translated_subtitle):
                source_context = json.loads(previous_message[0].parts[0].text)
                translated_context = json.loads(previous_message[1].parts[0].text)
                self.assertEqual([line["index"] for line in source_context], ["2", "3"])
                self.assertEqual([line["index"] for line in translated_context], ["2", "3"])
                raise SystemExit("context captured")

            with (
                patch.object(translator, "getmodels", return_value=["gemini-flash-latest"]),
                patch.object(translator, "_get_token_limit", return_value=None),
                patch.object(translator, "_validate_token_size", return_value=True),
                patch.object(translator, "_process_batch", side_effect=capture_context),
                patch("gemini_srt_translator.main.progress_bar", return_value=None),
                patch("gemini_srt_translator.main.info_with_progress", return_value=None),
                patch("gemini_srt_translator.main.warning_with_progress", return_value=None),
                patch("gemini_srt_translator.main.error_with_progress", return_value=None),
                patch("gemini_srt_translator.main.success_with_progress", return_value=None),
                patch("gemini_srt_translator.main.highlight", return_value=None),
            ):
                with self.assertRaises(SystemExit) as raised:
                    translator.translate()

            self.assertEqual(raised.exception.code, "context captured")

    @unittest.skipIf(os.name == "nt", "POSIX permission bits are not meaningful on Windows")
    def test_atomic_write_uses_normal_file_permissions_for_new_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_file = Path(tmp) / "sample.zh.srt"
            translator = GeminiSRTTranslator(output_file=str(output_file))
            old_umask = os.umask(0o022)
            try:
                translator._write_text_atomically(str(output_file), "translated")
            finally:
                os.umask(old_umask)

            self.assertEqual(stat.S_IMODE(output_file.stat().st_mode), 0o644)

    @unittest.skipIf(os.name == "nt", "POSIX permission bits are not meaningful on Windows")
    def test_atomic_write_preserves_existing_output_permissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_file = Path(tmp) / "sample.zh.srt"
            output_file.write_text("previous", encoding="utf-8")
            output_file.chmod(0o664)
            translator = GeminiSRTTranslator(output_file=str(output_file))

            translator._write_text_atomically(str(output_file), "translated")

            self.assertEqual(output_file.read_text(encoding="utf-8"), "translated")
            self.assertEqual(stat.S_IMODE(output_file.stat().st_mode), 0o664)

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

    def test_quiet_overload_abort_reports_last_error(self):
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
            captured = io.StringIO()

            set_quiet_mode(True)
            try:
                with (
                    patch.object(translator, "getmodels", return_value=["gemini-flash-latest"]),
                    patch.object(translator, "_get_token_limit", return_value=None),
                    patch.object(translator, "_validate_token_size", return_value=True),
                    patch.object(translator, "_process_batch", side_effect=RuntimeError("503 model overloaded")),
                    patch("gemini_srt_translator.main.time.sleep", return_value=None),
                    redirect_stdout(captured),
                ):
                    with self.assertRaises(SystemExit) as raised:
                        translator.translate()
            finally:
                set_quiet_mode(False)

            self.assertEqual(raised.exception.code, 130)
            self.assertIn("503 model overloaded", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
