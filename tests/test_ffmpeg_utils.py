import unittest
from unittest.mock import patch

from gemini_srt_translator.ffmpeg_utils import check_ffmpeg_installation


class FfmpegUtilsTests(unittest.TestCase):
    def test_check_ffmpeg_installation_returns_false_when_binary_is_missing(self):
        with patch(
            "gemini_srt_translator.ffmpeg_utils.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            self.assertFalse(check_ffmpeg_installation())


if __name__ == "__main__":
    unittest.main()
