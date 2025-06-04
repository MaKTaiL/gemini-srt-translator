import os

import gemini_srt_translator as gst

gst.gemini_api_key = os.getenv("GEMINI_API_KEY", "")

gst.listmodels()
