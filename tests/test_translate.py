import gemini_srt_translator as gst

gst.gemini_api_key = ""
gst.target_language = "French"
gst.input_file = "subtitle.srt"

gst.translate()