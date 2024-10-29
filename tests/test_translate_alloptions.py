import gemini_srt_translator as gst

gst.gemini_api_key = ""
gst.gemini_api_key2 = ""
gst.target_language = "French"
gst.input_file = "subtitle.srt"
gst.output_file = "translated_subtitle.srt"
gst.description = "This is a medical TV Show"
gst.model_name = "gemini-1.5-flash"
gst.batch_size = 30
gst.free_quota = True

gst.translate()
