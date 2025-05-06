# gemini_srt_translator.py

import json
import os
import signal
import time
import typing
import unicodedata as ud
from collections import Counter

import json_repair
import srt
from google import genai
from google.genai import types
from google.genai.types import Content
from srt import Subtitle

from gemini_srt_translator.logger import (
    error,
    error_with_progress,
    highlight,
    highlight_with_progress,
    info,
    info_with_progress,
    input_prompt,
    input_prompt_with_progress,
    progress_bar,
    set_color_mode,
    success_with_progress,
    warning,
    warning_with_progress,
)

from .helpers import get_instruction, get_safety_settings


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

    def __init__(
        self,
        gemini_api_key: str = None,
        gemini_api_key2: str = None,
        target_language: str = None,
        input_file: str = None,
        output_file: str = None,
        start_line: int = 1,
        description: str = None,
        model_name: str = "gemini-2.0-flash",
        batch_size: int = 50,
        free_quota: bool = True,
        use_colors: bool = True,
    ):
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
                self.output_file = "translated.srt"
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
        self.token_limit = 0
        self.token_count = 0
        self.config = types.GenerateContentConfig(
            response_mime_type="application/json",
            safety_settings=get_safety_settings(),
            system_instruction=get_instruction(target_language, description),
        )

        # Set color mode based on user preference
        set_color_mode(use_colors)

        # Initialize progress tracking file path
        self.progress_file = None
        if input_file:
            self.progress_file = os.path.join(os.path.dirname(input_file), f".{os.path.basename(input_file)}.progress")

        # Check for saved progress
        self._check_saved_progress()

    def _check_saved_progress(self):
        """Check if there's a saved progress file and load it if exists"""
        if not self.progress_file or not os.path.exists(self.progress_file):
            return

        if self.start_line != 1:
            return

        try:
            with open(self.progress_file, "r") as f:
                data = json.load(f)
                saved_line = data.get("line", 1)
                input_file = data.get("input_file")

                # Verify the progress file matches our current input file
                if input_file != self.input_file:
                    warning(f"Found progress file for different subtitle: {input_file}")
                    warning("Ignoring saved progress.")
                    return

                if saved_line > 1 and self.start_line == 1:
                    resume = input_prompt(f"Found saved progress. Resume? (y/n): ").lower().strip()
                    if resume == "y" or resume == "yes":
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
            with open(self.progress_file, "w") as f:
                json.dump({"line": line, "input_file": self.input_file}, f)
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

        client = self._get_client()
        models = client.models.list()
        list_models = []
        for model in models:
            supported_actions = model.supported_actions
            if "generateContent" in supported_actions:
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

        models = self.getmodels()

        if self.model_name not in models:
            error(f"Model {self.model_name} is not available. Please choose a different model.")
            exit(0)

        self.token_limit = self._get_token_limit()

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
                            self.start_line = int(
                                input_prompt(
                                    f"Enter the line number to start from (1 to {len(original_subtitle)}): "
                                ).strip()
                            )
                            if self.start_line < 1 or self.start_line > len(original_subtitle):
                                warning(
                                    f"Line number must be between 1 and {len(original_subtitle)}. Please try again."
                                )
                                continue
                            break
                        except ValueError:
                            warning("Invalid input. Please enter a valid number.")

            except FileNotFoundError:
                translated_subtitle = original_subtitle.copy()

            if len(original_subtitle) != len(translated_subtitle):
                error(
                    f"Number of lines of existing translated file does not match the number of lines in the original file."
                )
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
                previous_message = types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            text=json.dumps(
                                [
                                    SubtitleObject(
                                        index=str(j),
                                        content=translated_subtitle[j].content,
                                    )
                                    for j in range(0, self.start_line - 1)
                                ],
                                ensure_ascii=False,
                            )
                        )
                    ],
                )
            reverted = 0
            delay = False
            delay_time = 30

            if "pro" in self.model_name and self.free_quota:
                delay = True
                if not self.gemini_api_key2:
                    info("Pro model and free user quota detected, enabling 30s delay between requests...\n")
                else:
                    delay_time = 15
                    info("Pro model and free user quota detected, using secondary API key for additional quota...\n")

            highlight(f"Starting translation of {total - self.start_line + 1} lines...\n")
            progress_bar(i, total, prefix="Translating:", suffix=f"{self.model_name}")

            batch.append(SubtitleObject(index=str(i), content=original_subtitle[i].content))
            i += 1

            if self.gemini_api_key2:
                info_with_progress(f"Starting with API Key {self.current_api_number}")

            def handle_interrupt(signal_received, frame):
                warning_with_progress(f"Translation interrupted. Saving partial results to file. Progress saved.")
                if translated_file:
                    translated_file.write(srt.compose(translated_subtitle))
                self._save_progress(i - len(batch) + 1)
                exit(0)

            signal.signal(signal.SIGINT, handle_interrupt)

            # Save initial progress
            self._save_progress(i + 1)

            last_time = 0
            while len(batch) > 0:
                if i < total and len(batch) < self.batch_size:
                    batch.append(SubtitleObject(index=str(i), content=original_subtitle[i].content))
                    i += 1
                    continue
                try:
                    if not self._validate_token_size(json.dumps(batch, ensure_ascii=False)):
                        error_with_progress(
                            f"Token size ({int(self.token_count/0.9)}) exceeds limit ({self.token_limit}) for {self.model_name}."
                        )
                        user_prompt = "0"
                        while not user_prompt.isdigit() or int(user_prompt) <= 0:
                            user_prompt = input_prompt_with_progress(
                                f"Please enter a new batch size (current: {self.batch_size}): "
                            )
                            if user_prompt.isdigit() and int(user_prompt) > 0:
                                new_batch_size = int(user_prompt)
                                decrement = self.batch_size - new_batch_size
                                if decrement > 0:
                                    for _ in range(decrement):
                                        i -= 1
                                        batch.pop()
                                self.batch_size = new_batch_size
                                info_with_progress(f"Batch size updated to {self.batch_size}.")
                            else:
                                warning_with_progress("Invalid input. Batch size must be a positive integer.")
                        continue

                    start_time = time.time()
                    previous_message = self._process_batch(batch, previous_message, translated_subtitle)
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
                        current_time = time.time()
                        if current_time - last_time > 60 and self._switch_api():
                            highlight_with_progress(
                                f"API {self.backup_api_number} quota exceeded! Switching to API {self.current_api_number}..."
                            )
                        else:
                            for j in range(60, 0, -1):
                                warning_with_progress(f"All API quotas exceeded, waiting {j} seconds...")
                                time.sleep(1)
                        last_time = current_time
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

    def _get_client(self) -> genai.Client:
        """
        Configure and return a Gemini client instance.

        Returns:
            genai.Client: Configured Gemini client instance
        """
        client = genai.Client(api_key=self.current_api_key)
        return client

    def _get_token_limit(self) -> int:
        """
        Get the token limit for the current model.

        Returns:
            int: Token limit for the current model
        """
        client = self._get_client()
        model = client.models.get(model=self.model_name)
        return model.output_token_limit

    def _validate_token_size(self, contents: str) -> bool:
        """
        Validate the token size of the input contents.

        Args:
            contents (str): Input contents to validate

        Returns:
            bool: True if token size is valid, False otherwise
        """
        client = self._get_client()
        token_count = client.models.count_tokens(model="gemini-2.0-flash-lite", contents=contents)
        self.token_count = token_count.total_tokens
        if token_count.total_tokens > self.token_limit * 0.9:
            return False
        return True

    def _process_batch(
        self,
        batch: list[SubtitleObject],
        previous_message: Content,
        translated_subtitle: list[Subtitle],
    ) -> Content:
        """
        Process a batch of subtitles for translation.

        Args:
            batch (list[SubtitleObject]): Batch of subtitles to translate
            previous_message (ContentDict): Previous message for context
            translated_subtitle (list[Subtitle]): List to store translated subtitles

        Returns:
            ContentDict: The model's response for context in next batch
        """
        client = self._get_client()
        current_message = types.Content(role="user", parts=[types.Part(text=json.dumps(batch, ensure_ascii=False))])
        if previous_message:
            contents = [previous_message, current_message]
        else:
            contents = [current_message]
        response = client.models.generate_content(model=self.model_name, contents=contents, config=self.config)
        translated_lines: list[SubtitleObject] = json_repair.loads(response.text)

        if len(translated_lines) != len(batch):
            raise Exception("Gemini has returned the wrong number of lines.")

        for line in translated_lines:
            if line["index"] not in [x["index"] for x in batch]:
                raise Exception("Gemini has returned different indices.")
            if self._dominant_strong_direction(line["content"]) == "rtl":
                translated_subtitle[int(line["index"])].content = f"\u202b{line['content']}\u202c"
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
        rtl_count = count["R"] + count["AL"] + count["RLE"] + count["RLI"]
        ltr_count = count["L"] + count["LRE"] + count["LRI"]
        return "rtl" if rtl_count > ltr_count else "ltr"
