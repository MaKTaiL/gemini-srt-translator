# gemini_srt_translator.py

import typing
import json
import json_repair
import time
import unicodedata as ud
import signal
import os

import srt
from srt import Subtitle
from collections import Counter

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, ContentDict
from google.generativeai import GenerativeModel

from gemini_srt_translator.logger import (
    info, warning, error, success, highlight, 
    set_color_mode, input_prompt, progress_bar,
    warning_with_progress, error_with_progress, info_with_progress, 
    success_with_progress, highlight_with_progress
)

class SubtitleObject(typing.TypedDict):
    """
    TypedDict for subtitle objects used in translation
    """
    index: str
    content: str

class GeminiSRTTranslator:
    """
    A translator class that uses Gemini API to translate subtitles.
    """
    def __init__(self, gemini_api_key: str = None, gemini_api_key2: str = None, target_language: str = None, 
                 input_file: str = None, output_file: str = None, start_line: int = 1, description: str = None, 
                 model_name: str = "gemini-2.0-flash", batch_size: int = 30, free_quota: bool = True,
                 use_colors: bool = True):
        """
        Initialize the translator with necessary parameters.

        Args:
            gemini_api_key (str): Primary Gemini API key
            gemini_api_key2 (str): Secondary Gemini API key for additional quota
            target_language (str): Target language for translation
            input_file (str): Path to input subtitle file
            output_file (str): Path to output translated subtitle file
            start_line (int): Line number to start translation from
            description (str): Additional instructions for translation
            model_name (str): Gemini model to use
            batch_size (int): Number of subtitles to process in each batch
            free_quota (bool): Whether to use free quota (affects rate limiting)
            use_colors (bool): Whether to use colored output (default: True)
        """

        if not output_file:
            try:
                self.output_file = ".".join(self.input_file.split(".")[:-1]) + "_translated.srt"
            except:
                pass
        else:
            self.output_file = output_file
            
        self.gemini_api_key = gemini_api_key
        self.gemini_api_key2 = gemini_api_key2
        self.current_api_key = gemini_api_key
        self.current_api_number = 1
        self.backup_api_number = 2
        self.target_language = target_language
        self.input_file = input_file
        self.start_line = start_line
        self.description = description
        self.model_name = model_name
        self.batch_size = batch_size
        self.free_quota = free_quota
        
        # Set color mode based on user preference
        set_color_mode(use_colors)

        # Initialize progress tracking file path
        self.progress_file = None
        if input_file:
            self.progress_file = os.path.join(os.path.dirname(input_file), 
                                           f".{os.path.basename(input_file)}.progress")
            
        # Check for saved progress
        self._check_saved_progress()

    def _check_saved_progress(self):
        """Check if there's a saved progress file and load it if exists"""
        if not self.progress_file or not os.path.exists(self.progress_file):
            return
        
        if self.start_line != 1:
            return
        
        try:
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
                saved_line = data.get('line', 1)
                input_file = data.get('input_file')
                
                # Verify the progress file matches our current input file
                if input_file != self.input_file:
                    warning(f"Found progress file for different subtitle: {input_file}")
                    warning("Ignoring saved progress.")
                    return
                
                if saved_line > 1 and self.start_line == 1:
                    resume = input_prompt(f"Found saved progress. Resume? (y/n): ").lower().strip()
                    if resume == 'y' or resume == 'yes':
                        info(f"Resuming from line {saved_line}")
                        self.start_line = saved_line
                    else:
                        info("Starting from the beginning")
                        # Remove the progress file
                        try:
                            os.remove(self.output_file)
                        except Exception as e:
                            print(e)
                            pass
        except Exception as e:
            warning(f"Error reading progress file: {e}")
            
    def _save_progress(self, line):
        """Save current progress to temporary file"""
        if not self.progress_file:
            return
            
        try:
            with open(self.progress_file, 'w') as f:
                json.dump({
                    'line': line, 
                    'input_file': self.input_file
                }, f)
        except Exception as e:
            warning_with_progress(f"Failed to save progress: {e}")
            
    def _clear_progress(self):
        """Clear the progress file on successful completion"""
        if self.progress_file and os.path.exists(self.progress_file):
            try:
                os.remove(self.progress_file)
            except Exception as e:
                warning(f"Failed to remove progress file: {e}")

    def getmodels(self):
        """Get available Gemini models that support content generation."""
        if not self.current_api_key:
            error("Please provide a valid Gemini API key.")
            exit(0)

        genai.configure(api_key=self.current_api_key)
        models = genai.list_models()
        list_models = []
        for model in models:
            if "generateContent" in model.supported_generation_methods:
                list_models.append(model.name.replace("models/", ""))
        return list_models

    def translate(self):
        """
        Main translation method. Reads the input subtitle file, translates it in batches,
        and writes the translated subtitles to the output file.
        """
        if not self.current_api_key:
            error("Please provide a valid Gemini API key.")
            exit(0)
        
        if not self.target_language:
            error("Please provide a target language.")
            exit(0)
        
        if not self.input_file:
            error("Please provide a subtitle file.")
            exit(0)
        
        if not self.output_file:
            self.output_file = ".".join(self.input_file.split(".")[:-1]) + "_translated.srt"

        instruction = f"""You are an assistant that translates subtitles to {self.target_language}.
You will receive the following JSON type:

class SubtitleObject(typing.TypedDict):
    index: str
    content: str

Request: list[SubtitleObject]

The 'index' key is the index of the subtitle dialog.
The 'content' key is the dialog to be translated.

The indices must remain the same in the response as in the request.
Dialogs must be translated as they are without any changes.
"""

        if self.description:
            instruction += "\nAdditional user instruction: '" + self.description + "'"

        models = self.getmodels()
        model = self._get_model(instruction)

        if self.model_name not in models:
            error(f"Model {self.model_name} is not available. Please choose a different model.")
            exit(0)

        with open(self.input_file, "r", encoding="utf-8") as original_file:
            original_text = original_file.read()
            original_subtitle = list(srt.parse(original_text))
            try:
                translated_file_exists = open(self.output_file, "r", encoding="utf-8")
                translated_subtitle = list(srt.parse(translated_file_exists.read()))
                if self.start_line == 1:
                    info(f"Translated file {self.output_file} already exists. Loading existing translation...\n")
                    while True:
                        try:
                            self.start_line = int(input_prompt(f"Enter the line number to start from (1 to {len(original_subtitle)}): ").strip())
                            if self.start_line < 1 or self.start_line > len(original_subtitle):
                                warning(f"Line number must be between 1 and {len(original_subtitle)}. Please try again.")
                                continue
                            break
                        except ValueError:
                            warning("Invalid input. Please enter a valid number.")

            except FileNotFoundError:
                translated_subtitle = original_subtitle.copy()
            
            if len(original_subtitle) != len(translated_subtitle):
                error(f"Number of line of existing translated file does not match the number of lines in the original file.")
                exit(0)

            translated_file = open(self.output_file, "w", encoding="utf-8")

            if self.start_line > len(original_subtitle) or self.start_line < 1:
                error(f"Start line must be between 1 and {len(original_subtitle)}. Please check the input file.")
                exit(0)
            
            i = self.start_line - 1
            total = len(original_subtitle)
            batch = []
            previous_message = None
            if self.start_line > 1:
                previous_message = {"role": "model", "parts": [{"text": json.dumps(
                    [SubtitleObject(index=str(j), content=translated_subtitle[j].content) for j in range(max(0, i - self.batch_size), i)],
                    ensure_ascii=False
                )}]}
            reverted = 0
            delay = False
            delay_time = 30

            if 'pro' in self.model_name and self.free_quota:
                delay = True
                if not self.gemini_api_key2:
                    info("Pro model and free user quota detected, enabling 30s delay between requests...\n")
                else:
                    delay_time = 15
                    info("Pro model and free user quota detected, using secondary API key for additional quota...\n")
                    print()

            batch.append(SubtitleObject(index=str(i), content=original_subtitle[i].content))
            i += 1

            highlight(f"Starting translation of {total - self.start_line + 1} lines...\n")
            progress_bar(i, total, prefix="Translating:", suffix=f"{self.model_name}")

            if self.gemini_api_key2:
                info(f"Starting with API Key {self.current_api_number}\n")

            def handle_interrupt(signal_received, frame):
                warning_with_progress(f"Translation interrupted. Saving partial results to file. Progress saved.")
                if translated_file:
                    translated_file.write(srt.compose(translated_subtitle))
                self._save_progress(i - len(batch) + 1)
                exit(0)

            signal.signal(signal.SIGINT, handle_interrupt)
            
            # Save initial progress
            self._save_progress(i + 1)

            while len(batch) > 0:
                if i < total and len(batch) < self.batch_size:
                    batch.append(SubtitleObject(index=str(i), content=original_subtitle[i].content))
                    i += 1
                    continue
                try:
                    start_time = time.time()
                    previous_message = self._process_batch(model, batch, previous_message, translated_subtitle)
                    end_time = time.time()
                    
                    # Update progress bar
                    progress_bar(i, total, prefix="Translating:", suffix=f"{self.model_name}")
                    
                    # Save progress after each batch
                    self._save_progress(i)
                    
                    if delay and (end_time - start_time < delay_time) and i < total:
                        time.sleep(30 - (end_time - start_time))
                    if reverted > 0:
                        self.batch_size += reverted
                        reverted = 0
                        info_with_progress(f"Increasing batch size back to {self.batch_size}...")
                    if i < total and len(batch) < self.batch_size:
                        batch.append(SubtitleObject(index=str(i), content=original_subtitle[i].content))
                        i += 1
                except Exception as e:
                    e_str = str(e)
                    
                    if "quota" in e_str:
                        if self._switch_api():
                            highlight_with_progress(f"🔄 API {self.backup_api_number} quota exceeded! Switching to API {self.current_api_number}...")
                            model = self._get_model(instruction)
                        else:
                            warning_with_progress("All API quotas exceeded, waiting 1 minute...")
                            time.sleep(60)
                    else:
                        if self.batch_size == 1:
                            translated_file.write(srt.compose(translated_subtitle))
                            # Save progress before exiting
                            self._save_progress(i + 1)
                            error_with_progress(f"Translation failed. Saving partial results to file. Progress saved.")
                            exit(0)
                        if self.batch_size > 1:
                            decrement = min(10, self.batch_size - 1)
                            reverted += decrement
                            for _ in range(decrement):
                                i -= 1
                                batch.pop()
                            self.batch_size -= decrement
                        if "Gemini" in e_str:
                            error_with_progress(f"{e_str}")
                        else:
                            error_with_progress(f"An unexpected error has occurred: {e_str}")
                        warning_with_progress(f"Decreasing batch size to {self.batch_size} and trying again...")
                
            success_with_progress("Translation completed successfully!")
            translated_file.write(srt.compose(translated_subtitle))
            
            # Clear progress file on successful completion
            self._clear_progress()

    def _switch_api(self) -> bool:
        """
        Switch to the secondary API key if available.

        Returns:
            bool: True if switched successfully, False if no alternative API available
        """
        if self.current_api_number == 1 and self.gemini_api_key2:
            self.current_api_key = self.gemini_api_key2
            self.current_api_number = 2
            self.backup_api_number = 1
            return True
        if self.current_api_number == 2 and self.gemini_api_key:
            self.current_api_key = self.gemini_api_key
            self.current_api_number = 1
            self.backup_api_number = 2
            return True
        return False

    def _get_model(self, instruction: str) -> GenerativeModel:
        """
        Configure and return a Gemini model instance with current API key.

        Args:
            instruction (str): System instruction for the model

        Returns:
            GenerativeModel: Configured Gemini model instance
        """
        genai.configure(api_key=self.current_api_key)
        return genai.GenerativeModel(
            model_name=self.model_name,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
            system_instruction=instruction,
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )

    def _process_batch(self, model: GenerativeModel, batch: list[SubtitleObject], previous_message: ContentDict, translated_subtitle: list[Subtitle]) -> ContentDict:
        """
        Process a batch of subtitles for translation.

        Args:
            model (GenerativeModel): The Gemini model instance
            batch (list[SubtitleObject]): Batch of subtitles to translate
            previous_message (ContentDict): Previous message for context
            translated_subtitle (list[Subtitle]): List to store translated subtitles

        Returns:
            ContentDict: The model's response for context in next batch
        """
        if previous_message:
            messages = [previous_message] + [{"role": "user", "parts": [{"text": json.dumps(batch, ensure_ascii=False)}]}]
        else:
            messages = [{"role": "user", "parts": [{"text": json.dumps(batch, ensure_ascii=False)}]}]
        response = model.generate_content(messages)
        translated_lines: list[SubtitleObject] = json_repair.loads(response.text)
        
        if len(translated_lines) != len(batch):
            raise Exception("Gemini has returned the wrong number of lines.")
            
        for line in translated_lines:
            if line["index"] not in [x["index"] for x in batch]:
                raise Exception("Gemini has returned different indices.")
            if self._dominant_strong_direction(line["content"]) == "rtl":
                translated_subtitle[int(line["index"])].content = f"\u202B{line['content']}\u202C"
            else:
                translated_subtitle[int(line["index"])].content = line["content"]
        
        batch.clear()
        return response.candidates[0].content

    def _dominant_strong_direction(self, s: str) -> str:
        """
        Determine the dominant text direction (RTL or LTR) of a string.

        Args:
            s (str): Input string to analyze

        Returns:
            str: 'rtl' if right-to-left is dominant, 'ltr' otherwise
        """
        count = Counter([ud.bidirectional(c) for c in list(s)])
        rtl_count = count['R'] + count['AL'] + count['RLE'] + count["RLI"]
        ltr_count = count['L'] + count['LRE'] + count["LRI"]
        return "rtl" if rtl_count > ltr_count else "ltr"
