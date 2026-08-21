import json
import os
import tempfile
import unittest

from pydub import AudioSegment

from gemini_srt_translator.agent_cli import (
    cmd_agent_commit,
    cmd_agent_next,
    cmd_agent_reset,
    cmd_agent_start,
    cmd_agent_status,
)
from gemini_srt_translator.session import SubtitleSession, TranscriptionSession

SAMPLE_SRT = """1
00:00:01,000 --> 00:00:03,000
Hello, world!

2
00:00:04,000 --> 00:00:06,000
How are you today?

3
00:00:07,000 --> 00:00:09,000
This is a test subtitle.

4
00:00:10,000 --> 00:00:12,000
Goodbye!
"""


class TestSubtitleSession(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.srt_path = os.path.join(self.temp_dir.name, "test.srt")
        self.out_path = os.path.join(self.temp_dir.name, "test_translated.srt")
        with open(self.srt_path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_SRT)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_session_initialization(self):
        session = SubtitleSession(
            input_file=self.srt_path,
            target_language="French",
            output_file=self.out_path,
            batch_size=2,
        )
        self.assertEqual(session.total_lines, 4)
        self.assertEqual(session.current_line, 1)
        self.assertFalse(session.is_complete())

    def test_get_next_batch_and_commit(self):
        session = SubtitleSession(
            input_file=self.srt_path,
            target_language="French",
            output_file=self.out_path,
            batch_size=2,
            resume_context_size=2,
        )

        # Batch 1
        batch1 = session.get_next_batch()
        self.assertIsNotNone(batch1)
        self.assertEqual(batch1["batch_number"], 1)
        self.assertEqual(len(batch1["batch"]), 2)
        self.assertEqual(batch1["batch"][0]["text"], "Hello, world!")
        self.assertEqual(batch1["context"], [])

        # Commit Batch 1
        trans1 = [
            {"index": "0", "text": "Bonjour le monde !"},
            {"index": "1", "text": "Comment allez-vous aujourd'hui ?"},
        ]
        res1 = session.commit_batch(trans1)
        self.assertTrue(res1["success"])
        self.assertFalse(res1["is_complete"])
        self.assertEqual(session.current_line, 3)

        # Batch 2
        batch2 = session.get_next_batch()
        self.assertIsNotNone(batch2)
        self.assertEqual(batch2["batch_number"], 2)
        self.assertEqual(len(batch2["batch"]), 2)
        self.assertEqual(batch2["batch"][0]["text"], "This is a test subtitle.")
        self.assertEqual(len(batch2["context"]), 2)
        self.assertEqual(batch2["context"][0]["text"], "Bonjour le monde !")

        # Commit Batch 2
        trans2 = json.dumps(
            [
                {"index": "2", "text": "Ceci est un sous-titre de test."},
                {"index": "3", "text": "Au revoir !"},
            ]
        )
        res2 = session.commit_batch(trans2)
        self.assertTrue(res2["success"])
        self.assertTrue(res2["is_complete"])
        self.assertTrue(session.is_complete())
        self.assertTrue(os.path.exists(self.out_path))

    def test_commit_item_count_validation(self):
        session = SubtitleSession(
            input_file=self.srt_path,
            target_language="French",
            output_file=self.out_path,
            batch_size=2,
        )
        # Pass 1 item when 2 are expected
        bad_trans = [{"index": "0", "text": "Only one item"}]
        res = session.commit_batch(bad_trans)
        self.assertFalse(res["success"])
        self.assertIn("Item count mismatch", res["error"])

    def test_resume_progress(self):
        session1 = SubtitleSession(
            input_file=self.srt_path,
            target_language="French",
            output_file=self.out_path,
            batch_size=2,
        )
        session1.commit_batch(
            [
                {"index": "0", "text": "Bonjour"},
                {"index": "1", "text": "Comment allez-vous"},
            ]
        )

        # Create new session loading the same file
        session2 = SubtitleSession(
            input_file=self.srt_path,
            target_language="French",
            output_file=self.out_path,
            batch_size=2,
            resume=True,
        )
        self.assertEqual(session2.current_line, 3)
        self.assertEqual(session2.get_status()["completed_lines"], 2)


class TestTranscriptionSession(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mp3_path = os.path.join(self.temp_dir.name, "test_audio.mp3")
        self.out_path = os.path.join(self.temp_dir.name, "test_transcribed.srt")

        # Generate a 20-second silent mp3 for testing
        seg = AudioSegment.silent(duration=20000)
        seg.export(self.mp3_path, format="mp3")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_transcription_initialization(self):
        session = TranscriptionSession(
            audio_file=self.mp3_path,
            output_file=self.out_path,
            audio_chunk_size=10,
        )
        self.assertEqual(session.total_seconds, 20)
        self.assertEqual(session.current_seconds, 0)
        self.assertFalse(session.is_complete())

    def test_transcription_chunk_and_commit(self):
        session = TranscriptionSession(
            audio_file=self.mp3_path,
            output_file=self.out_path,
            audio_chunk_size=10,
        )

        # Chunk 1 (0 to 10s)
        chunk1 = session.get_next_chunk()
        self.assertIsNotNone(chunk1)
        self.assertEqual(chunk1["chunk_number"], 1)
        self.assertEqual(chunk1["start_seconds"], 0)
        self.assertEqual(chunk1["end_seconds"], 10)
        self.assertIsNotNone(chunk1["audio_bytes"])

        # Commit chunk 1
        items1 = [
            {"text": "Hello world from 0 to 4s", "time_start": "00:00", "time_end": "00:04"},
            {"text": "Next sentence from 5 to 9s", "time_start": "00:05", "time_end": "00:09"},
        ]
        res1 = session.commit_chunk(items1)
        self.assertTrue(res1["success"])
        self.assertEqual(res1["added_subtitles"], 2)
        self.assertEqual(session.current_seconds, 10)
        self.assertFalse(session.is_complete())
        self.assertTrue(os.path.exists(self.out_path))

        # Chunk 2 (10 to 20s)
        chunk2 = session.get_next_chunk()
        self.assertIsNotNone(chunk2)
        self.assertEqual(chunk2["chunk_number"], 2)
        self.assertEqual(chunk2["start_seconds"], 10)
        self.assertEqual(chunk2["end_seconds"], 20)

        # Commit chunk 2
        items2 = [
            {"text": "Ending part from 11 to 15s", "time_start": "00:01", "time_end": "00:05"},
        ]
        res2 = session.commit_chunk(items2)
        self.assertTrue(res2["success"])
        self.assertTrue(res2["is_complete"])
        self.assertTrue(session.is_complete())
        self.assertEqual(len(session.transcribed_subtitles), 3)

    def test_transcription_default_output_name(self):
        session = TranscriptionSession(audio_file=self.mp3_path)
        expected_name = os.path.join(self.temp_dir.name, "test_audio.srt")
        self.assertEqual(session.output_file, expected_name)

    def test_transcription_cleanup_extracted_mp3_and_chunks(self):
        # Create a mock extracted mp3
        extracted_mp3 = os.path.join(self.temp_dir.name, "video_extracted.mp3")
        seg = AudioSegment.silent(duration=10000)
        seg.export(extracted_mp3, format="mp3")

        session = TranscriptionSession(audio_file=extracted_mp3, video_file="video.mp4")
        self.assertTrue(session.audio_extracted)
        chunk = session.get_next_chunk()
        self.assertIsNotNone(chunk)
        chunk_file = chunk["audio_chunk_path"]
        self.assertTrue(os.path.exists(chunk_file))
        self.assertTrue(os.path.exists(extracted_mp3))

        session.cleanup()
        self.assertFalse(os.path.exists(extracted_mp3))
        self.assertFalse(os.path.exists(chunk_file))


class TestSkillManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_get_skill_content(self):
        from gemini_srt_translator.skill import get_skill_content

        content = get_skill_content()
        self.assertIn("subtitle-translator", content)
        self.assertIn("gst agent translate", content)

    def test_install_skill_local(self):
        from gemini_srt_translator.skill import install_skill

        installed = install_skill(target="antigravity", is_global=False, cwd=self.temp_dir.name)
        self.assertEqual(len(installed), 1)
        expected_file = os.path.join(self.temp_dir.name, ".gemini", "skills", "subtitle-translator", "SKILL.md")
        self.assertTrue(os.path.exists(expected_file))
        self.assertEqual(installed[0], expected_file)

    def test_install_skill_all_targets(self):
        from gemini_srt_translator.skill import install_skill

        installed = install_skill(target="all", is_global=False, cwd=self.temp_dir.name)
        self.assertGreaterEqual(len(installed), 3)
        for p in installed:
            self.assertTrue(os.path.exists(p))


if __name__ == "__main__":
    unittest.main()
