import os
import gemini_srt_translator as gst

gst.gemini_api_key = "your_gemini_api_key_here"
gst.target_language = "French"

input_dir = r"input folder"
output_dir = r"output folder"

for filename in os.listdir(input_dir):
    if filename.endswith(".srt"):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename)
        gst.input_file = input_file
        gst.output_file = output_file
        gst.description = f"Translation of {filename}"
        gst.translate()
