import os

import gemini_srt_translator as gst

gst.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
gst.gemini_api_key2 = os.getenv("GEMINI_API_KEY2", "")
gst.target_language = "French"
gst.input_file = "subtitle.srt"
gst.output_file = "translated_subtitle.srt"
gst.description = "This is a medical TV Show"
gst.model_name = "gemini-2.0-flash"
gst.start_line = 1
gst.batch_size = 30
gst.free_quota = True
gst.skip_upgrade = True
gst.use_colors = True

gst.translate()
