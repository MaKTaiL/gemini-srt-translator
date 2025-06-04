import os

import gemini_srt_translator as gst

gst.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
gst.target_language = "French"
gst.input_file = "subtitle.srt"

gst.translate()
