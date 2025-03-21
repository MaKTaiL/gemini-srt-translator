import gemini_srt_translator as gst
import os
import getpass

# Set the API key for the Gemini service

if not gst.gemini_api_key and not gst.gemini_api_key2:
    input_gemini_api_key = getpass.getpass("\033[96mEnter your Gemini API key: \033[0m").strip()
    if input_gemini_api_key:
        gst.gemini_api_key = input_gemini_api_key

# Allow the user to drag and drop the SRT file
input_file = input("\033[96mDrag and drop the .srt file here and press Enter: \033[0m").strip().strip('"')

# Check if the file exists
if not os.path.isfile(input_file) or not input_file.lower().endswith(".srt"):
    print("\033[31m[Error] The selected file is not valid or is not a .srt file.\033[0m")
    exit(1)

# Set the target language for translation
target_language = input("\033[96mEnter the target language (e.g., French, Spanish): \033[0m").strip()
if target_language:
    gst.target_language = target_language

# List of available models
available_models = gst.getmodels()

# Display options and select the model
def select_model():
    print()
    print("\033[95mSelect the translation model:\033[0m")
    for i, model in enumerate(available_models, 1):
        print(f"\033[92m{i}. \033[93m{model}\033[0m")
    print()
    
    while True:
        try:
            choice = int(input("\033[96mEnter the desired model number: \033[0m"))
            print()
            if 1 <= choice <= len(available_models):
                print(f"\033[93mYou selected the model: \033[92m{available_models[choice - 1]}\033[0m")
                return available_models[choice - 1]
            else:
                print("\033[31m[ERROR] Invalid choice. Please try again.\033[0m")
        except ValueError:
            print("\033[31m[ERROR] Please enter a valid number.\033[0m")

# Select and store the chosen model
selected_model = select_model()
gst.model_name = selected_model

# Extract the folder and generate the path for the translated file


file_name = os.path.basename(input_file)
output_file = os.path.join(os.path.dirname(input_file), file_name.replace(".srt", f"_translated_{selected_model}.srt"))

# Set input and output files
gst.input_file = input_file
gst.output_file = output_file

# Execute translation
print()
print(f"\033[91mTranslating the file. Model used: ”\033[92m{selected_model}\033[91m”. Please wait...\033[0m")
print()
gst.translate()

# Text colored in yellow
print()
print(f"\033[96mThe translation has been saved to: \033[92m{output_file}\033[0m")

print()
input("\033[31mPress any key to close the window...\033[0m")
