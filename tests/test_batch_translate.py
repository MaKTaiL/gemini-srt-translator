import gemini_srt_translator as gst
import concurrent.futures
import os
import glob

def translate_file(input_file, output_file=None):
    gst.gemini_api_key = ""
    gst.gemini_api_key2 = ""
    gst.target_language = "Persian"
    gst.input_file = input_file
    gst.output_file = output_file or f"{os.path.splitext(input_file)[0]}_translated.srt"
    gst.description = "Note that these sentences are part of a React JS tutorial, so please tailor your translations accordingly."
    gst.model_name = "gemini-1.5-pro-latest"
    gst.batch_size = 60
    gst.free_quota = True

    try:
        # Skip if translated file already exists
        if os.path.exists(gst.output_file):
            print(f"File {gst.output_file} has already been translated. Skipping...")
            return
            
        gst.translate()
        print(f"Successfully translated file {input_file}")
    except Exception as e:
        print(f"Error translating file {input_file}: {str(e)}")

def translate_multiple_files(max_workers=3):
    """
    Translate multiple subtitle files concurrently in the current directory
    
    Args:
        max_workers (int): Maximum number of concurrent processes
    """
    # Find all .srt files in current directory
    srt_files = glob.glob("*.srt")
    
    if not srt_files:
        print("No .srt files found in current directory!")
        return
        
    print(f"Found {len(srt_files)} subtitle files:")
    for file in srt_files:
        print(f"- {file}")
    
    print("\nStarting translation...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(translate_file, input_file) for input_file in srt_files]
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    # Start concurrent translation of all .srt files
    translate_multiple_files()

