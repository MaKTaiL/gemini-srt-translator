import json
import os
import re
import signal
import stat
import tempfile
import time
import typing
import unicodedata as ud
from collections import Counter
from datetime import datetime, timedelta
from typing import Literal

import json_repair
import srt
from google import genai
from google.genai import types
from google.genai.types import Content
from srt import Subtitle

from gemini_srt_translator.logger import (
    error,
    error_with_progress,
    get_last_chunk_size,
    highlight,
    highlight_with_progress,
    info,
    info_with_progress,
    input_prompt,
    input_prompt_with_progress,
    progress_bar,
    save_logs_to_file,
    save_thoughts_to_file,
    set_color_mode,
    success_with_progress,
    update_loading_animation,
    warning,
    warning_with_progress,
)

from .ffmpeg_utils import (
    check_ffmpeg_installation,
    extract_audio_from_video,
    extract_srt_from_video,
    get_audio_length,
)
from .helpers import (
    get_safety_settings,
    get_transcribe_instruction,
    get_transcribe_response_schema,
    get_translate_instruction,
    get_translate_response_schema,
)
from .utils import convert_timedelta_to_timestamp, convert_timestamp_to_timedelta


class SubtitleObject(typing.TypedDict):
    """
    TypedDict for subtitle objects used in translation
    """

    index: str
    text: str
    time_start: typing.Optional[str]
    time_end: typing.Optional[str]


class GeminiSRTTranslator:
    """
    A translator class that uses Gemini API to translate subtitles.
    """

    def __init__(
        self,
        gemini_api_key: str = None,
        gemini_api_key2: str = None,
        use_enterprise: bool = False,
        cloud_api_key: str = None,
        cloud_project: str = None,
        cloud_location: str = "global",
        request_type: Literal["shared", "dedicated"] = None,
        target_language: str = None,
        input_file: str = None,
        output_file: str = None,
        video_file: str = None,
        audio_file: str = None,
        audio_chunk_size: int = 300,
        extract_audio: bool = False,
        isolate_voice: bool = True,
        start_line: int = None,
        description: str = None,
        model_name: str = "gemini-3.5-flash",
        batch_size: int = 1000,
        streaming: bool = True,
        thinking: bool = True,
        thinking_budget: int = None,
        thinking_level: Literal["minimal", "low", "medium", "high"] = None,
        service_tier: Literal["standard", "flex", "priority"] = None,
        token_stats: bool = False,
        token_report: str = None,
        preserve_context: bool = True,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        free_quota: bool = True,
        use_colors: bool = True,
        progress_log: bool = False,
        thoughts_log: bool = False,
        resume: bool = None,
    ):
        """
        Initialize the translator with necessary parameters.
        """
        base_file = input_file or video_file or audio_file
        base_name = os.path.splitext(os.path.basename(base_file))[0] if base_file else "file"
        dir_path = os.path.dirname(base_file) if base_file else ""

        self.log_file_path = (
            os.path.join(dir_path, f"{base_name}.progress.log") if dir_path else f"{base_name}.progress.log"
        )
        self.thoughts_file_path = (
            os.path.join(dir_path, f"{base_name}.thoughts.log") if dir_path else f"{base_name}.thoughts.log"
        )

        if output_file:
            self.output_file = output_file
        else:
            if audio_file and not input_file and not video_file:
                suffix = "_transcribed.srt"
            else:
                suffix = "_translated.srt" if input_file else ".srt"
            self.output_file = os.path.join(dir_path, f"{base_name}{suffix}") if dir_path else f"{base_name}{suffix}"

        self.progress_file = os.path.join(dir_path, f"{base_name}.progress") if dir_path else f"{base_name}.progress"

        self.gemini_api_key = gemini_api_key
        self.gemini_api_key2 = gemini_api_key2
        self.current_api_key = gemini_api_key
        self.use_enterprise = use_enterprise
        self.cloud_api_key = cloud_api_key
        self.cloud_project = cloud_project
        self.cloud_location = cloud_location
        self.request_type = request_type
        self.target_language = target_language
        self.input_file = input_file
        self.video_file = video_file
        self.audio_file = audio_file
        self.audio_chunk_size = audio_chunk_size
        self.extract_audio = extract_audio
        self.isolate_voice = isolate_voice
        self.start_line = start_line
        self.description = description
        self.model_name = model_name
        self.batch_size = batch_size
        self.streaming = streaming
        self.thinking = thinking
        self.thinking_budget = thinking_budget
        self.thinking_level = thinking_level
        self.service_tier = service_tier
        self.token_stats = token_stats
        self.token_report = token_report
        self.preserve_context = preserve_context
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.free_quota = free_quota
        self.progress_log = progress_log
        self.thoughts_log = thoughts_log
        self.resume = resume
        self.use_colors = use_colors

        self.current_api_number = 1
        self.backup_api_number = 2
        self.batch_number = 1
        self.audio = None
        self.audio_part = None
        self.token_limit = 0
        self.token_count = 0
        self._report_prompt_tokens = 0
        self._report_thoughts_tokens = 0
        self._report_output_tokens = 0
        self._report_total_tokens = 0
        self._start_time = time.time()
        self.translated_batch = []
        self.srt_extracted = False
        self.audio_extracted = False
        self.ffmpeg_installed = check_ffmpeg_installation()
        self.consecutive_error_count = 0
        self.max_consecutive_errors = 3

        set_color_mode(use_colors)

    def _get_translate_config(self):
        """Get the configuration for the translation model."""
        thinking_compatible = True
        if "2.0" in self.model_name or "gemini" not in self.model_name:
            thinking_compatible = False
        elif "2.5" in self.model_name:
            self.thinking_level = None
            if "pro" in self.model_name:
                if self.thinking == False:
                    warning_with_progress(
                        "You cannot disable thinking for Gemini 2.5 Pro. Setting thinking budget to 128.",
                        ignore_quiet=True,
                    )
                    self.thinking_budget = 128
                if self.thinking_budget is not None and self.thinking_budget < 128:
                    warning_with_progress(
                        "Gemini 2.5 Pro requires a minimum thinking budget of 128. Setting to 128.",
                        ignore_quiet=True,
                    )
                    self.thinking_budget = 128
        elif "3" in self.model_name:
            self.thinking_budget = None
            if "pro" in self.model_name and self.thinking_level is not None and "minimal" in self.thinking_level:
                warning_with_progress(
                    "You cannot set thinking level to minimal for Gemini 3 Pro. Setting thinking level to low.",
                    ignore_quiet=True,
                )
                self.thinking_level = "low"
            if self.thinking == False:
                warning_with_progress(
                    "You cannot disable thinking for Gemini 3 models. Setting thinking level to lowest possible.",
                    ignore_quiet=True,
                )
                if "pro" in self.model_name:
                    self.thinking_level = "low"
                else:
                    self.thinking_level = "minimal"

        return types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=get_translate_response_schema(),
            safety_settings=get_safety_settings(),
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            system_instruction=get_translate_instruction(
                language=self.target_language,
                thinking=self.thinking,
                thinking_compatible=thinking_compatible,
                audio_file=self.audio_file,
                description=self.description,
            ),
            service_tier=self.service_tier if not self.use_enterprise else None,
            thinking_config=(
                types.ThinkingConfig(
                    include_thoughts=self.thinking,
                    thinking_budget=self.thinking_budget,
                    thinking_level=self.thinking_level,
                )
                if thinking_compatible
                else None
            ),
        )

    def _get_transcribe_config(self):
        """Get the configuration for the transcription model."""
        thinking_compatible = True
        if "2.0" in self.model_name or "gemini" not in self.model_name:
            thinking_compatible = False
        elif "2.5" in self.model_name:
            self.thinking_level = None
            if "pro" in self.model_name:
                if self.thinking == False:
                    warning_with_progress(
                        "You cannot disable thinking for Gemini 2.5 Pro. Setting thinking budget to 128.",
                        ignore_quiet=True,
                    )
                    self.thinking_budget = 128
                if self.thinking_budget is not None and self.thinking_budget < 128:
                    warning_with_progress(
                        "Gemini 2.5 Pro requires a minimum thinking budget of 128. Setting to 128.",
                        ignore_quiet=True,
                    )
                    self.thinking_budget = 128
        elif "3" in self.model_name:
            self.thinking_budget = None
            if "pro" in self.model_name and self.thinking_level is not None and "minimal" in self.thinking_level:
                warning_with_progress(
                    "You cannot set thinking level to minimal for Gemini 3 Pro. Setting thinking level to low.",
                    ignore_quiet=True,
                )
                self.thinking_level = "low"
            if self.thinking == False:
                warning_with_progress(
                    "You cannot disable thinking for Gemini 3 models. Setting thinking level to lowest possible.",
                    ignore_quiet=True,
                )
                if "pro" in self.model_name:
                    self.thinking_level = "low"
                else:
                    self.thinking_level = "minimal"

        return types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=get_transcribe_response_schema(),
            safety_settings=get_safety_settings(),
            system_instruction=get_transcribe_instruction(
                thinking=self.thinking,
                thinking_compatible=thinking_compatible,
                description=self.description,
            ),
            service_tier=self.service_tier if not self.use_enterprise else None,
            thinking_config=(
                types.ThinkingConfig(
                    include_thoughts=self.thinking,
                    thinking_budget=self.thinking_budget,
                    thinking_level=self.thinking_level,
                )
                if thinking_compatible
                else None
            ),
        )

    def _check_saved_progress(self):
        """Check if there's a saved progress file and load it if exists"""
        if not self.progress_file or not os.path.exists(self.progress_file):
            return

        if self.start_line != None:
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

                if saved_line > 1:
                    if self.resume is None:
                        resume = input_prompt(f"Found saved progress. Resume? (y/n): ", mode="resume").lower().strip()
                    elif self.resume is True:
                        resume = "y"
                    elif self.resume is False:
                        resume = "n"
                    if resume == "y" or resume == "yes":
                        if self.use_colors:
                            info(f"Resuming from line \033[31m{saved_line}")
                        else:
                            info(f"Resuming from line {saved_line}")
                        self.start_line = saved_line
                    else:
                        info("Starting from the beginning")
                        try:
                            os.remove(self.output_file)
                        except Exception as e:
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

    def _write_text_atomically(self, path: str, text: str):
        """Write text to a sibling temporary file before replacing the target."""
        directory = os.path.dirname(os.path.abspath(path)) or "."
        filename = os.path.basename(path)
        temp_path = None
        file_descriptor = None
        try:
            output_mode = stat.S_IMODE(os.stat(path).st_mode)
        except FileNotFoundError:
            current_umask = os.umask(0)
            os.umask(current_umask)
            output_mode = 0o666 & ~current_umask

        try:
            file_descriptor, temp_path = tempfile.mkstemp(
                prefix=f".{filename}.",
                suffix=".tmp",
                dir=directory,
                text=True,
            )
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as temp_file:
                file_descriptor = None
                temp_file.write(text)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.chmod(temp_path, output_mode)
            os.replace(temp_path, path)
            temp_path = None
        finally:
            if file_descriptor is not None:
                os.close(file_descriptor)
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def _write_translated_subtitles(self, translated_subtitle):
        self._write_text_atomically(self.output_file, srt.compose(translated_subtitle, reindex=False, strict=False))

    def _save_transcribe_progress(self, time_in_seconds):
        """Save current transcription progress to a temporary file."""
        if not self.progress_file:
            return
        try:
            source_file = self.audio_file if self.audio_file else self.video_file
            with open(self.progress_file, "w") as f:
                json.dump({"time": time_in_seconds, "input_file": source_file}, f)
        except Exception as e:
            warning_with_progress(f"Failed to save transcription progress: {e}", isTranscribing=True)

    def _check_saved_transcribe_progress(self):
        """Check for saved transcription progress and return start time and existing subtitles."""
        if not self.progress_file or not os.path.exists(self.progress_file):
            return 0, []

        try:
            with open(self.progress_file, "r") as f:
                data = json.load(f)

            saved_time = data.get("time", 0)
            saved_input = data.get("input_file")
            current_input = self.audio_file if self.audio_file else self.video_file

            if not saved_input or not current_input or os.path.abspath(saved_input) != os.path.abspath(current_input):
                warning("Found progress file for a different source file. Ignoring saved progress.")
                return 0, []

            if saved_time > 0 and os.path.exists(self.output_file):
                should_resume = False
                if self.resume is True:
                    should_resume = True
                elif self.resume is None:
                    resume_choice = (
                        input_prompt(
                            f"Found saved progress. Resume transcription? (y/n): ",
                            mode="resume",
                        )
                        .lower()
                        .strip()
                    )
                    if resume_choice in ["y", "yes"]:
                        should_resume = True

                if should_resume:
                    info(
                        f"Resuming transcription from {convert_timedelta_to_timestamp(timedelta(seconds=int(saved_time)))}"
                    )
                    try:
                        with open(self.output_file, "r", encoding="utf-8") as f_in:
                            content = f_in.read()
                            if not content:
                                return saved_time, []
                            existing_subs = list(srt.parse(content))
                        return saved_time, existing_subs
                    except Exception as e:
                        warning(
                            f"Could not load existing subtitles from {self.output_file}: {e}. Resuming from saved time, but starting with an empty subtitle list."
                        )
                        return saved_time, []
                else:
                    info("Starting from the beginning.")
                    try:
                        if self.resume is False or (self.resume is None and not should_resume):
                            if os.path.exists(self.output_file):
                                os.remove(self.output_file)
                            if os.path.exists(self.progress_file):
                                os.remove(self.progress_file)
                    except OSError:
                        pass
                    return 0, []
        except Exception as e:
            warning(f"Could not read progress file: {e}. Starting from the beginning.")

        return 0, []

    def getmodels(self):
        """Get available Gemini models that support content generation."""
        client = self._get_client()
        models = client.models.list()
        list_models = []
        for model in models:
            if self.use_enterprise:
                list_models.append(model.name.replace("publishers/google/models/", ""))
            else:
                supported_actions = model.supported_actions
                if supported_actions and "generateContent" in supported_actions:
                    list_models.append(model.name.replace("models/", ""))
        return list_models

    def translate(self):
        """
        Main translation method. Reads the input subtitle file, translates it in batches,
        and writes the translated subtitles to the output file.
        """

        if not self.ffmpeg_installed and self.video_file:
            error(
                "FFmpeg is not installed. Please install FFmpeg to use video features.",
                ignore_quiet=True,
            )
            exit(1)

        if self.video_file and self.extract_audio:
            if os.path.exists(self.video_file):
                self.audio_file = extract_audio_from_video(self.video_file, isolate_voice=self.isolate_voice)
                self.audio_extracted = True
            else:
                error(f"Video file {self.video_file} does not exist.", ignore_quiet=True)
                exit(1)

        if self.audio_file:
            from pydub import AudioSegment

            if os.path.exists(self.audio_file):
                self.audio = AudioSegment.from_file(self.audio_file)
            else:
                error(f"Audio file {self.audio_file} does not exist.", ignore_quiet=True)
                exit(1)

        if self.video_file and not self.input_file:
            if not os.path.exists(self.video_file):
                error(f"Video file {self.video_file} does not exist.", ignore_quiet=True)
                exit(1)
            self.input_file = extract_srt_from_video(self.video_file)
            if not self.input_file:
                error("Failed to extract subtitles from video file.", ignore_quiet=True)
                exit(1)
            self.srt_extracted = True

        if not self.target_language:
            error("Please provide a target language.", ignore_quiet=True)
            exit(1)

        if self.input_file and not os.path.exists(self.input_file):
            error(f"Input file {self.input_file} does not exist.", ignore_quiet=True)
            exit(1)

        elif not self.input_file:
            error("Please provide a subtitle or video file.", ignore_quiet=True)
            exit(1)

        if self.service_tier and self.service_tier not in ["standard", "flex", "priority"]:
            error("Service tier must be 'standard', 'flex' or 'priority'.", ignore_quiet=True)
            exit(1)

        if self.thinking_budget is not None:
            if "gemini" in self.model_name:
                if "3" in self.model_name or "2.0" in self.model_name:
                    pass
                elif "pro" not in self.model_name and (self.thinking_budget < 0 or self.thinking_budget > 24576):
                    error(
                        "Thinking budget must be between 0 and 24576. 0 disables thinking.",
                        ignore_quiet=True,
                    )
                    exit(1)
                elif "pro" in self.model_name and (self.thinking_budget < 128 or self.thinking_budget > 32768):
                    error(
                        "Thinking budget must be between 128 and 32768.",
                        ignore_quiet=True,
                    )
                    exit(1)

        if self.thinking_level is not None and "gemini-3" in self.model_name:
            if (
                self.thinking_level != "minimal"
                and self.thinking_level != "low"
                and self.thinking_level != "medium"
                and self.thinking_level != "high"
            ):
                error(
                    "Thinking level must be 'minimal', 'low', 'medium', or 'high'.",
                    ignore_quiet=True,
                )
                exit(1)

        if self.temperature is not None and (self.temperature < 0 or self.temperature > 2):
            error("Temperature must be between 0.0 and 2.0.", ignore_quiet=True)
            exit(1)

        if self.top_p is not None and (self.top_p < 0 or self.top_p > 1):
            error("Top P must be between 0.0 and 1.0.", ignore_quiet=True)
            exit(1)

        if self.top_k is not None and self.top_k < 0:
            error("Top K must be a non-negative integer.", ignore_quiet=True)
            exit(1)

        self._check_saved_progress()

        if not self.cloud_api_key:
            models = self.getmodels()
            if self.model_name not in models:
                error(
                    f"Model {self.model_name} is not available. Please choose a different model.",
                    ignore_quiet=True,
                )
                exit(1)
            self._get_token_limit()

        with open(self.input_file, "r", encoding="utf-8") as original_file:
            original_text = original_file.read()
            original_subtitle = list(srt.parse(original_text))
            try:
                with open(self.output_file, "r", encoding="utf-8") as translated_file_exists:
                    translated_subtitle = list(srt.parse(translated_file_exists.read()))
                if self.use_colors:
                    info(
                        f"Translated file \033[90m{self.output_file}\033[94m already exists. Loading existing translation...\n"
                    )
                else:
                    info(f"Translated file {self.output_file} already exists. Loading existing translation...\n")
                if self.start_line == None:
                    while True:
                        try:
                            self.start_line = int(
                                input_prompt(
                                    f"Enter the line number to start from (1 to {len(original_subtitle)}): ",
                                    mode="line",
                                    max_length=len(original_subtitle),
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
                self.start_line = 1

            if len(original_subtitle) != len(translated_subtitle):
                error(
                    f"Number of lines of existing translated file does not match the number of lines in the original file.",
                    ignore_quiet=True,
                )
                exit(1)

            if self.start_line > len(original_subtitle) or self.start_line < 1:
                error(
                    f"Start line must be between 1 and {len(original_subtitle)}. Please try again.",
                    ignore_quiet=True,
                )
                exit(1)

            if len(original_subtitle) < self.batch_size:
                self.batch_size = len(original_subtitle)

            delay = False
            delay_time = 30

            if "pro" in self.model_name and self.free_quota:
                delay = True
                if not self.gemini_api_key2:
                    info("Pro model and free user quota detected.\n")
                else:
                    delay_time = 15
                    if self.use_colors:
                        info("\033[33mPro model and free user quota detected, using secondary API key if needed.\n")
                    else:
                        info("Pro model and free user quota detected, using secondary API key if needed.\n")

            i = self.start_line - 1
            total = len(original_subtitle)
            batch = []
            previous_message = []
            if self.start_line > 1:
                start_idx = max(0, self.start_line - 2 - self.batch_size)
                parts_user = []
                subtitle_array: list[SubtitleObject] = []
                offset = 0
                for j in range(start_idx, self.start_line - 1):
                    if j == 0:
                        offset = original_subtitle[j].start.seconds
                    subtitle_kwargs = {
                        "index": str(j),
                        "text": original_subtitle[j].content,
                    }
                    if self.audio_file:
                        subtitle_kwargs["time_start"] = convert_timedelta_to_timestamp(
                            original_subtitle[j].start, offset=offset
                        )
                        subtitle_kwargs["time_end"] = convert_timedelta_to_timestamp(
                            original_subtitle[j].end, offset=offset
                        )
                    subtitle_array.append(SubtitleObject(**subtitle_kwargs))

                parts_user.append(
                    types.Part(
                        text=json.dumps(
                            subtitle_array,
                            ensure_ascii=False,
                        ),
                    )
                )

                parts_model = []
                parts_model.append(
                    types.Part(
                        text=json.dumps(
                            [
                                SubtitleObject(
                                    index=str(j),
                                    text=translated_subtitle[j].content,
                                )
                                for j in range(start_idx, self.start_line - 1)
                            ],
                            ensure_ascii=False,
                        )
                    )
                )

                previous_message = [
                    types.Content(
                        role="user",
                        parts=parts_user,
                    ),
                    types.Content(
                        role="model",
                        parts=parts_model,
                    ),
                ]

            highlight(f"Starting translation of {total - self.start_line + 1} lines...\n")
            if self.use_colors:
                progress_bar(
                    i,
                    total,
                    prefix="Translating:",
                    suffix=f"\033[31m{self.model_name}",
                    isSending=True,
                    token_stats=self.token_stats,
                )
            else:
                progress_bar(
                    i,
                    total,
                    prefix="Translating:",
                    suffix=f"{self.model_name}",
                    isSending=True,
                    token_stats=self.token_stats,
                )

            if self.gemini_api_key2:
                if self.use_colors:
                    info(f"Starting with \033[31mAPI Key {self.current_api_number}")
                else:
                    info(f"Starting with API Key {self.current_api_number}")

            def handle_interrupt(signal_received, frame):
                last_chunk_size = get_last_chunk_size()
                warning_with_progress(
                    f"Translation interrupted. Saving partial results to file. Progress saved.",
                    chunk_size=max(0, last_chunk_size - 1),
                )
                self._write_translated_subtitles(translated_subtitle)
                if self.progress_log:
                    save_logs_to_file(self.log_file_path)
                self._save_progress(max(1, i - len(batch) + max(0, last_chunk_size - 1) + 1))
                self._write_token_report("translate")
                exit(130)

            signal.signal(signal.SIGINT, handle_interrupt)

            self._save_progress(i)

            last_time = 0
            validated = False
            offset = 0
            offset_end = 0
            server_overload_retries = 0
            max_overload_retries = 3

            while i < total or len(batch) > 0:
                if i < total and len(batch) < self.batch_size:
                    if offset_end - offset < self.audio_chunk_size:
                        subtitle_kwargs = {
                            "index": str(i),
                            "text": original_subtitle[i].content,
                        }
                        if self.audio_file:
                            if len(batch) == 0:
                                offset = original_subtitle[i].start.seconds
                            subtitle_kwargs["time_start"] = convert_timedelta_to_timestamp(
                                original_subtitle[i].start, offset=offset
                            )
                            subtitle_kwargs["time_end"] = convert_timedelta_to_timestamp(
                                original_subtitle[i].end, offset=offset
                            )
                            offset_end = original_subtitle[i].end.seconds
                        batch.append(SubtitleObject(**subtitle_kwargs))
                        i += 1
                        continue
                    else:
                        if batch:
                            i -= 1
                            offset_end = original_subtitle[i].end.seconds
                            batch.pop()
                try:
                    while not validated:
                        if not self.use_enterprise:
                            info_with_progress(f"Validating token size...")
                            try:
                                validated = self._validate_token_size(json.dumps(batch, ensure_ascii=False))
                            except Exception as e:
                                error_with_progress(f"Error validating token size: {e}")
                                info_with_progress(f"Retrying validation...")
                                continue
                        else:
                            break
                        if not validated:
                            error_with_progress(
                                f"Token size ({int(self.token_count/0.9)}) exceeds limit ({self.token_limit}) for {self.model_name}."
                            )
                            user_prompt = "0"
                            while not user_prompt.isdigit() or int(user_prompt) <= 0:
                                user_prompt = input_prompt_with_progress(
                                    f"Please enter a new batch size (current: {self.batch_size}): ",
                                    batch_size=self.batch_size,
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
                        success_with_progress(f"Token size validated. Translating...", isSending=True)

                    if i == total and len(batch) < self.batch_size:
                        self.batch_size = len(batch)

                    if self.audio:
                        audio_bytes = self.audio[offset * 1000 : offset_end * 1000].export(format="mp3").read()
                        self.audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp3")

                    start_time = time.time()
                    previous_message = self._process_batch(batch, previous_message, translated_subtitle)
                    end_time = time.time()
                    offset = offset_end

                    server_overload_retries = 0
                    self.consecutive_error_count = 0

                    progress_bar(
                        i,
                        total,
                        prefix="Translating:",
                        suffix=(f"\033[31m{self.model_name}" if self.use_colors else f"{self.model_name}"),
                        isSending=True,
                        token_stats=self.token_stats,
                    )

                    self._save_progress(i + 1)

                    if delay and (end_time - start_time < delay_time) and i < total:
                        time.sleep(delay_time - (end_time - start_time))

                except Exception as e:
                    self.consecutive_error_count += 1
                    warning_with_progress(
                        f"Consecutive error count: {self.consecutive_error_count}/{self.max_consecutive_errors}"
                    )

                    if self.consecutive_error_count >= self.max_consecutive_errors:
                        error_with_progress(
                            f"Stopping script due to reaching {self.max_consecutive_errors} consecutive errors to prevent API quota waste."
                        )
                        signal.raise_signal(signal.SIGINT)
                        time.sleep(2)

                    e_str = str(e).lower()

                    if "quota" in e_str:
                        current_time = time.time()
                        if current_time - last_time > 60 and self._switch_api():
                            highlight_with_progress(
                                f"API {self.backup_api_number} quota exceeded! Switching to API {self.current_api_number}...",
                                isSending=True,
                            )
                        else:
                            for j in range(60, 0, -1):
                                warning_with_progress(f"All API quotas exceeded, waiting {j} seconds...")
                                time.sleep(1)
                        last_time = current_time
                        continue

                    elif any(err in e_str for err in ["429", "500", "503", "unavailable", "overloaded"]):
                        server_overload_retries += 1
                        if server_overload_retries <= max_overload_retries:
                            warning_with_progress(
                                f"Model is overloaded. Attempt {server_overload_retries}/{max_overload_retries}. Pausing for 60 seconds..."
                            )
                            time.sleep(60)
                            info_with_progress("Resuming translation...", isSending=True)
                            continue
                        else:
                            error_with_progress(
                                f"Model is still overloaded after {max_overload_retries} attempts. Aborting."
                            )
                            signal.raise_signal(signal.SIGINT)

                    else:
                        if isinstance(e, json.decoder.JSONDecodeError):
                            warning_with_progress(f"JSON response error.")
                        elif "line" in str(e) or "empty" in str(e):
                            warning_with_progress(f"{e}")
                        else:
                            warning_with_progress(f"An unexpected error occurred: {e}.")

                        start_index_in_batch = int(batch[0]["index"]) + 1
                        end_index_in_batch = int(batch[-1]["index"]) + 1
                        info_with_progress(
                            f"Retrying batch for lines {start_index_in_batch}-{end_index_in_batch}...",
                            isSending=True,
                        )

                        time.sleep(5)

                        if self.progress_log:
                            save_logs_to_file(self.log_file_path)

                        continue

            if self.use_colors:
                success_with_progress("\033[96m✅ \033[96mTranslation completed successfully!")
            else:
                success_with_progress("✅ Translation completed successfully!")
            if self.progress_log:
                save_logs_to_file(self.log_file_path)
            self._write_translated_subtitles(translated_subtitle)

            self._write_token_report("translate")

            if self.audio_file and os.path.exists(self.audio_file) and self.audio_extracted:
                os.remove(self.audio_file)

            if self.progress_file and os.path.exists(self.progress_file):
                os.remove(self.progress_file)

        if self.srt_extracted and os.path.exists(self.input_file):
            os.remove(self.input_file)

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
        if self.use_enterprise:
            if self.cloud_api_key:
                client = genai.Client(
                    api_key=self.cloud_api_key,
                    enterprise=True,
                    http_options=(
                        types.HttpOptions(
                            headers={
                                "X-Vertex-AI-LLM-Request-Type": self.request_type if self.request_type else "",
                                "X-Vertex-AI-LLM-Shared-Request-Type": self.service_tier if self.service_tier else "",
                            }
                        )
                    ),
                )
            elif self.cloud_project:
                client = genai.Client(
                    project=self.cloud_project,
                    location=self.cloud_location,
                    enterprise=True,
                    http_options=(
                        types.HttpOptions(
                            headers={
                                "X-Vertex-AI-LLM-Request-Type": self.request_type if self.request_type else "",
                                "X-Vertex-AI-LLM-Shared-Request-Type": self.service_tier if self.service_tier else "",
                            }
                        )
                    ),
                )
            else:
                error("Google API key or Cloud project is required for enterprise mode.")
                exit(1)
        elif self.current_api_key:
            client = genai.Client(api_key=self.current_api_key)
        else:
            error("Please provide a valid Gemini API key.")
            exit(1)
        return client

    def _get_token_limit(self):
        """
        Get the token limit for the current model.

        Returns:
            int: Token limit for the current model
        """
        if not self.use_enterprise:
            client = self._get_client()
            model = client.models.get(model=self.model_name)
            self.token_limit = model.output_token_limit

    def _validate_token_size(self, contents: str) -> bool:
        """
        Validate the token size of the input contents.

        Args:
            contents (str): Input contents to validate

        Returns:
            bool: True if token size is valid, False otherwise
        """
        client = self._get_client()
        token_count = client.models.count_tokens(model="gemini-3.1-flash-lite", contents=contents)
        self.token_count = token_count.total_tokens
        if token_count.total_tokens > self.token_limit * 0.9:
            return False
        return True

    def _accumulate_report_tokens(
        self,
        prompt_tokens: int | None = None,
        thoughts_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        if prompt_tokens is not None:
            self._report_prompt_tokens += prompt_tokens
        if thoughts_tokens is not None:
            self._report_thoughts_tokens += thoughts_tokens
        if output_tokens is not None:
            self._report_output_tokens += output_tokens
        if total_tokens is not None:
            self._report_total_tokens += total_tokens

    def _calculate_cost(self) -> dict | None:
        pricing_path = os.path.join(os.path.dirname(__file__), "pricing.json")
        try:
            with open(pricing_path, "r", encoding="utf-8") as f:
                pricing = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

        tier = self.service_tier or "standard"
        model_pricing = pricing.get("models", {}).get(self.model_name)
        if not model_pricing:
            return None

        rates = model_pricing.get(tier) or model_pricing.get("standard")
        if not rates:
            return None

        input_cost = self._report_prompt_tokens / 1_000_000 * rates["input"]
        output_cost = (self._report_thoughts_tokens + self._report_output_tokens) / 1_000_000 * rates["output"]
        return {
            "input": round(input_cost, 6),
            "output": round(output_cost, 6),
            "total": round(input_cost + output_cost, 6),
            "tier": tier,
        }

    def _write_token_report(self, mode: str) -> None:
        if not self.token_report:
            return
        report = {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "model": self.model_name,
            "output_file": self.output_file,
            "duration_seconds": round(time.time() - self._start_time, 2),
            "tokens": {
                "prompt": self._report_prompt_tokens,
                "thoughts": self._report_thoughts_tokens,
                "output": self._report_output_tokens,
                "total": self._report_total_tokens,
            },
        }
        if self.input_file:
            report["input_file"] = self.input_file
        if self.audio_file:
            report["audio_file"] = self.audio_file
        if mode == "translate" and self.target_language:
            report["target_language"] = self.target_language
        cost = self._calculate_cost()
        if cost:
            report["cost"] = cost
        try:
            report_dir = os.path.dirname(self.token_report)
            if report_dir and not os.path.exists(report_dir):
                os.makedirs(report_dir)
            with open(self.token_report, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except (PermissionError, OSError) as e:
            warning(f"Failed to write token report to {self.token_report}: {e}")

    def _process_batch(
        self,
        batch: list[SubtitleObject],
        previous_message: list[Content],
        translated_subtitle: list[Subtitle],
    ) -> Content:
        """
        Process a batch of subtitles for translation.

        Args:
            batch (list[SubtitleObject]): Batch of subtitles to translate
            previous_message (Content): Previous message for context
            translated_subtitle (list[Subtitle]): List to store translated subtitles

        Returns:
            Content: The model's response for context in next batch
        """
        client = self._get_client()
        parts = []
        parts.append(types.Part(text=json.dumps(batch, ensure_ascii=False)))
        if self.audio_part:
            parts.append(self.audio_part)

        current_message = types.Content(role="user", parts=parts)
        contents = []
        contents += previous_message
        contents.append(current_message)

        done = False
        retry = -1
        previous_prompt_tokens = 0
        previous_thoughts_tokens = 0
        previous_output_tokens = 0
        previous_total_tokens = 0
        while done == False:
            response_text = ""
            thoughts_text = ""
            chunk_size = 0
            self.translated_batch = []
            done_thinking = False
            retry += 1
            blocked = False
            thoughts_signature = None
            if not self.streaming:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=self._get_translate_config(),
                )
                if response.prompt_feedback:
                    blocked = True
                    break
                if not response.text:
                    raise ValueError("Gemini has returned an empty response.")
                for part in response.candidates[0].content.parts:
                    if part.thought_signature:
                        thoughts_signature = part.thought_signature
                    if not part.text:
                        continue
                    elif part.thought:
                        thoughts_text += part.text
                        continue
                    else:
                        response_text += part.text
                if self.thoughts_log and self.thinking:
                    if retry == 0:
                        info_with_progress(f"Batch {self.batch_number} thinking process saved to file.")
                    else:
                        info_with_progress(f"Batch {self.batch_number}.{retry} thinking process saved to file.")
                    save_thoughts_to_file(thoughts_text, self.thoughts_file_path, retry)
                self.translated_batch: list[SubtitleObject] = json_repair.loads(response_text)
                if not isinstance(self.translated_batch, list) or not all(
                    isinstance(item, dict) for item in self.translated_batch
                ):
                    self.translated_batch = self._flatten_repaired_json(self.translated_batch)
                chunk_size = len(self.translated_batch)
                update_loading_animation(
                    chunk_size=chunk_size,
                    token_stats=self.token_stats,
                    prompt_tokens=response.usage_metadata.prompt_token_count,
                    thoughts_tokens=response.usage_metadata.thoughts_token_count,
                    output_tokens=response.usage_metadata.candidates_token_count,
                    total_tokens=response.usage_metadata.total_token_count,
                )
                self._accumulate_report_tokens(
                    prompt_tokens=response.usage_metadata.prompt_token_count,
                    thoughts_tokens=response.usage_metadata.thoughts_token_count,
                    output_tokens=response.usage_metadata.candidates_token_count,
                    total_tokens=response.usage_metadata.total_token_count,
                )
            else:
                if blocked:
                    break
                response = client.models.generate_content_stream(
                    model=self.model_name,
                    contents=contents,
                    config=self._get_translate_config(),
                )
                for chunk in response:
                    if chunk.prompt_feedback:
                        blocked = True
                        break
                    if chunk.candidates[0].content.parts:
                        for part in chunk.candidates[0].content.parts:
                            if part.thought_signature:
                                thoughts_signature = part.thought_signature
                            if not part.text:
                                continue
                            elif part.thought:
                                thoughts_text += part.text
                            else:
                                if not done_thinking:
                                    done_thinking = True
                                    if self.thoughts_log and self.thinking:
                                        if retry == 0:
                                            info_with_progress(
                                                f"Batch {self.batch_number} thinking process saved to file."
                                            )
                                        else:
                                            info_with_progress(
                                                f"Batch {self.batch_number}.{retry} thinking process saved to file."
                                            )
                                        save_thoughts_to_file(thoughts_text, self.thoughts_file_path, retry)
                                response_text += part.text
                                self.translated_batch: list[SubtitleObject] = json_repair.loads(
                                    response_text, stream_stable=True
                                )
                                if not isinstance(self.translated_batch, list) or not all(
                                    isinstance(item, dict) for item in self.translated_batch
                                ):
                                    self.translated_batch = self._flatten_repaired_json(self.translated_batch)
                    chunk_size = len(self.translated_batch)
                    if chunk_size > 0:
                        self._process_translated_lines(
                            translated_lines=self.translated_batch,
                            translated_subtitle=translated_subtitle,
                            batch=batch,
                            finished=False,
                        )
                    update_loading_animation(
                        chunk_size=chunk_size,
                        isThinking=self.thinking and not done_thinking,
                        token_stats=self.token_stats,
                        prompt_tokens=(
                            chunk.usage_metadata.prompt_token_count - previous_prompt_tokens
                            if chunk.usage_metadata and chunk.usage_metadata.prompt_token_count
                            else None
                        ),
                        thoughts_tokens=(
                            chunk.usage_metadata.thoughts_token_count - previous_thoughts_tokens
                            if chunk.usage_metadata and chunk.usage_metadata.thoughts_token_count
                            else None
                        ),
                        output_tokens=(
                            chunk.usage_metadata.candidates_token_count - previous_output_tokens
                            if chunk.usage_metadata and chunk.usage_metadata.candidates_token_count
                            else None
                        ),
                        total_tokens=(
                            chunk.usage_metadata.total_token_count - previous_total_tokens
                            if chunk.usage_metadata and chunk.usage_metadata.total_token_count
                            else None
                        ),
                    )
                    self._accumulate_report_tokens(
                        prompt_tokens=(
                            chunk.usage_metadata.prompt_token_count - previous_prompt_tokens
                            if chunk.usage_metadata and chunk.usage_metadata.prompt_token_count
                            else None
                        ),
                        thoughts_tokens=(
                            chunk.usage_metadata.thoughts_token_count - previous_thoughts_tokens
                            if chunk.usage_metadata and chunk.usage_metadata.thoughts_token_count
                            else None
                        ),
                        output_tokens=(
                            chunk.usage_metadata.candidates_token_count - previous_output_tokens
                            if chunk.usage_metadata and chunk.usage_metadata.candidates_token_count
                            else None
                        ),
                        total_tokens=(
                            chunk.usage_metadata.total_token_count - previous_total_tokens
                            if chunk.usage_metadata and chunk.usage_metadata.total_token_count
                            else None
                        ),
                    )
                    previous_prompt_tokens = (
                        chunk.usage_metadata.prompt_token_count
                        if chunk.usage_metadata and chunk.usage_metadata.prompt_token_count
                        else 0
                    )
                    previous_thoughts_tokens = (
                        chunk.usage_metadata.thoughts_token_count
                        if chunk.usage_metadata and chunk.usage_metadata.thoughts_token_count
                        else 0
                    )
                    previous_output_tokens = (
                        chunk.usage_metadata.candidates_token_count
                        if chunk.usage_metadata and chunk.usage_metadata.candidates_token_count
                        else 0
                    )
                    previous_total_tokens = (
                        chunk.usage_metadata.total_token_count
                        if chunk.usage_metadata and chunk.usage_metadata.total_token_count
                        else 0
                    )

            if len(self.translated_batch) == len(batch):
                self._process_translated_lines(
                    translated_lines=self.translated_batch,
                    translated_subtitle=translated_subtitle,
                    batch=batch,
                    finished=True,
                )
                done = True
                self.batch_number += 1
            else:
                raise ValueError(f"Expected {len(batch)} lines, got {len(self.translated_batch)}.")

        if blocked:
            error_with_progress(
                "Gemini has blocked the translation for unknown reasons. Try changing your description (if you have one) and/or the batch size and try again."
            )
            signal.raise_signal(signal.SIGINT)
        parts = []
        if thoughts_signature:
            parts.append(types.Part(text=response_text, thought_signature=thoughts_signature))
        else:
            parts.append(types.Part(text=response_text))
        previous_content = [
            types.Content(
                role="user",
                parts=[types.Part(text=json.dumps(batch, ensure_ascii=False))],
            ),
            types.Content(role="model", parts=parts),
        ]
        batch.clear()
        return previous_content if self.preserve_context else []

    @staticmethod
    def _flatten_repaired_json(data) -> list:
        """
        Flatten nested structures produced by json_repair when partial JSON
        contains \\n-[ patterns (newline + bracket in subtitle text).

        json_repair can misinterpret these as array boundaries, producing
        nested lists like [[{...}, {...}], ["text"]] instead of [{...}, {...}].
        This extracts all valid dict items from the nested structure.
        """
        result = []
        if not isinstance(data, list):
            return result
        for item in data:
            if isinstance(item, dict):
                result.append(item)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, dict):
                        result.append(sub)
        return result

    def _process_translated_lines(
        self,
        translated_lines: list[SubtitleObject],
        translated_subtitle: list[Subtitle],
        batch: list[SubtitleObject],
        finished: bool,
    ) -> bool:
        """
        Process the translated lines and update the subtitle list.

        Args:
            translated_lines (list[SubtitleObject]): List of translated lines
            translated_subtitle (list[Subtitle]): List to store translated subtitles
            batch (list[SubtitleObject]): Batch of subtitles to translate
            finished (bool): Whether the translation is finished
        """
        if not translated_lines:
            raise ValueError("Gemini has returned an empty translation.")

        i = 0
        indexes = [x["index"] for x in batch]
        last_translated_line = translated_lines[-1]
        for line in translated_lines:
            if "text" not in line or "index" not in line:
                if line != last_translated_line or finished:
                    raise ValueError(f"Gemini has returned a malformed object for line {int(indexes[i]) + 1}.")
                else:
                    continue
            if line["index"] not in indexes:
                raise ValueError(f"Gemini has returned an unexpected line: {int(line['index']) + 1}.")
            if line["text"] == "" and batch[i]["text"] != "":
                if line != last_translated_line or finished:
                    raise ValueError(f"Gemini has returned an empty translation for line {int(line['index']) + 1}.")
                else:
                    continue
            if self._dominant_strong_direction(line["text"]) == "rtl":
                translated_subtitle[int(line["index"])].content = f"\u202b{line['text']}\u202c"
            else:
                translated_subtitle[int(line["index"])].content = line["text"]
            i += 1

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

    def extract(self, type: typing.Literal["audio", "srt"] = "audio"):
        """
        Extract audio or subtitles from the video file using FFmpeg.
        """
        if not self.ffmpeg_installed:
            error(
                "FFmpeg is not installed. Please install FFmpeg to use this feature.",
                ignore_quiet=True,
            )
            exit(1)

        if not self.video_file or not os.path.exists(self.video_file):
            error("Please provide a valid video file for extraction.", ignore_quiet=True)
            exit(1)

        if type == "audio":
            self.audio_file = extract_audio_from_video(self.video_file, isolate_voice=self.isolate_voice)
            if not self.audio_file:
                error("Failed to extract audio from the video file.", ignore_quiet=True)
                exit(1)
        elif type == "srt":
            self.input_file = extract_srt_from_video(self.video_file)
            if not self.input_file:
                error(
                    "Failed to extract subtitles from the video file.",
                    ignore_quiet=True,
                )
                exit(1)
        else:
            error("Invalid extraction type. Use 'audio' or 'srt'.", ignore_quiet=True)
            exit(1)

    def transcribe(self):
        """
        Transcribe audio file into subtitles.
        """
        from pydub import AudioSegment

        extracted = False
        if self.video_file and not self.audio_file:
            self.audio_file = extract_audio_from_video(self.video_file, isolate_voice=self.isolate_voice)
            if not self.audio_file:
                error("Failed to extract audio from the video file.", ignore_quiet=True)
                exit(1)
            extracted = True

        if not self.audio_file:
            error(
                "Please provide a valid audio or video file for transcription.",
                ignore_quiet=True,
            )
            exit(1)

        if not os.path.exists(self.audio_file):
            error(f"Audio file {self.audio_file} does not exist.", ignore_quiet=True)
            exit(1)

        if "gemini" not in self.model_name or ("2.5" not in self.model_name and "3" not in self.model_name):
            error(
                f"Model {self.model_name} is not available for transcription. Please use a Gemini 2.5 or 3.0 model.",
                ignore_quiet=True,
            )
            exit(1)

        if self.service_tier and self.service_tier not in ["standard", "flex", "priority"]:
            error("Service tier must be 'standard', 'flex' or 'priority'.", ignore_quiet=True)
            exit(1)

        if self.thinking_budget is not None:
            if "gemini" in self.model_name:
                if "3" in self.model_name or "2.0" in self.model_name:
                    pass
                elif "pro" not in self.model_name and (self.thinking_budget < 0 or self.thinking_budget > 24576):
                    error(
                        "Thinking budget must be between 0 and 24576. 0 disables thinking.",
                        ignore_quiet=True,
                    )
                    exit(1)
                elif "pro" in self.model_name and (self.thinking_budget < 128 or self.thinking_budget > 32768):
                    error(
                        "Thinking budget must be between 128 and 32768.",
                        ignore_quiet=True,
                    )
                    exit(1)

        if self.thinking_level is not None and "gemini-3" in self.model_name:
            if (
                self.thinking_level != "minimal"
                and self.thinking_level != "low"
                and self.thinking_level != "medium"
                and self.thinking_level != "high"
            ):
                error(
                    "Thinking level must be 'minimal', 'low', 'medium', or 'high'.",
                    ignore_quiet=True,
                )
                exit(1)

        if self.temperature is not None and (self.temperature < 0 or self.temperature > 2):
            error("Temperature must be between 0.0 and 2.0.", ignore_quiet=True)
            exit(1)

        if self.top_p is not None and (self.top_p < 0 or self.top_p > 1):
            error("Top P must be between 0.0 and 1.0.", ignore_quiet=True)
            exit(1)

        if self.top_k is not None and self.top_k < 0:
            error("Top K must be a non-negative integer.", ignore_quiet=True)
            exit(1)

        current_length, transcribed_subtitle_objects = self._check_saved_transcribe_progress()

        last_saved_time = current_length

        def handle_interrupt(signal_received, frame):
            warning_with_progress(
                f"Transcription interrupted. Saving partial results and progress...",
                isTranscribing=True,
            )
            self._save_transcribe_progress(last_saved_time)
            if transcribed_subtitle_objects:
                transcribed_subtitle = srt.compose(transcribed_subtitle_objects)
                with open(self.output_file, "w", encoding="utf-8") as f:
                    f.write(transcribed_subtitle)
            if self.progress_log:
                save_logs_to_file(self.log_file_path)
            self._write_token_report("transcribe")
            exit(0)

        signal.signal(signal.SIGINT, handle_interrupt)

        client = self._get_client()
        try:
            audio_file = AudioSegment.from_mp3(self.audio_file)
            audio_length = get_audio_length(self.audio_file)

            index = len(transcribed_subtitle_objects) + 1
            last_time = 0

            self._save_transcribe_progress(current_length)

            while current_length < int(audio_length):
                chunk_end = min(current_length + self.audio_chunk_size, int(audio_length))
                audio_chunk = audio_file[current_length * 1000 : chunk_end * 1000].export(format="mp3").read()
                audio_part = types.Part.from_bytes(data=audio_chunk, mime_type="audio/mp3")
                current_message = types.Content(role="user", parts=[audio_part])
                progress_bar(
                    current_length,
                    audio_length,
                    prefix="Transcribing:",
                    suffix=(f"\033[31m{self.model_name}" if self.use_colors else f"{self.model_name}"),
                    isSending=True,
                    isTranscribing=True,
                    token_stats=self.token_stats,
                )
                if self.use_colors:
                    info_with_progress(
                        f"Transcribing audio segment \033[93m{convert_timedelta_to_timestamp(timedelta(seconds=current_length))} \033[94mto \033[93m{convert_timedelta_to_timestamp(timedelta(seconds=chunk_end))}.",
                        isTranscribing=True,
                        isSending=True,
                    )
                else:
                    info_with_progress(
                        f"Transcribing audio segment {convert_timedelta_to_timestamp(timedelta(seconds=current_length))} to {convert_timedelta_to_timestamp(timedelta(seconds=chunk_end))}.",
                        isTranscribing=True,
                        isSending=True,
                    )

                max_retries = 3
                server_error_retries = 0
                chunk_processed_successfully = False

                while not chunk_processed_successfully:

                    def _normalize_timestamp(ts_str: str) -> str:
                        """Converts 'HH:MM:SS' back to the non-standard 'MM:SS' that the old utils expects."""
                        parts = ts_str.split(":")
                        if len(parts) == 3:
                            try:
                                h, m, s = map(int, parts)
                                total_minutes = (h * 60) + m
                                return f"{total_minutes:02}:{s:02}"
                            except (ValueError, TypeError):
                                return ts_str
                        return ts_str

                    try:
                        done = False
                        retry = -1
                        blocked = False
                        done_thinking = False
                        previous_prompt_tokens = 0
                        previous_thoughts_tokens = 0
                        previous_output_tokens = 0
                        previous_total_tokens = 0
                        processed_seconds = 0
                        while not done:
                            response_text = ""
                            thoughts_text = ""
                            retry += 1
                            if not self.streaming:
                                response = client.models.generate_content(
                                    model=self.model_name,
                                    contents=[current_message],
                                    config=self._get_transcribe_config(),
                                )
                                if response.prompt_feedback:
                                    blocked = True
                                    break
                                if not response.text:
                                    raise ValueError("Gemini has returned an empty response.")
                                for part in response.candidates[0].content.parts:
                                    if not part.text:
                                        continue
                                    elif part.thought:
                                        thoughts_text += part.text
                                        continue
                                    else:
                                        response_text += part.text
                                if self.thoughts_log and self.thinking:
                                    if retry == 0:
                                        info_with_progress(
                                            f"Batch {self.batch_number} thinking process saved to file.",
                                            isTranscribing=True,
                                        )
                                    else:
                                        info_with_progress(
                                            f"Batch {self.batch_number}.{retry} thinking process saved to file.",
                                            isTranscribing=True,
                                        )
                                    save_thoughts_to_file(thoughts_text, self.thoughts_file_path, retry)
                                update_loading_animation(
                                    chunk_size=processed_seconds if processed_seconds else 0,
                                    isTranscribing=True,
                                    isThinking=self.thinking and not done_thinking,
                                    token_stats=self.token_stats,
                                    prompt_tokens=response.usage_metadata.prompt_token_count,
                                    thoughts_tokens=response.usage_metadata.thoughts_token_count,
                                    output_tokens=response.usage_metadata.candidates_token_count,
                                    total_tokens=response.usage_metadata.total_token_count,
                                )
                                self._accumulate_report_tokens(
                                    prompt_tokens=response.usage_metadata.prompt_token_count,
                                    thoughts_tokens=response.usage_metadata.thoughts_token_count,
                                    output_tokens=response.usage_metadata.candidates_token_count,
                                    total_tokens=response.usage_metadata.total_token_count,
                                )
                            else:
                                if blocked:
                                    break
                                response = client.models.generate_content_stream(
                                    model=self.model_name,
                                    contents=[current_message],
                                    config=self._get_transcribe_config(),
                                )
                                for chunk in response:
                                    if chunk.prompt_feedback:
                                        blocked = True
                                        break
                                    if chunk.candidates[0].content.parts:
                                        for part in chunk.candidates[0].content.parts:
                                            if not part.text:
                                                continue
                                            if part.thought:
                                                thoughts_text += part.text
                                            else:
                                                if not done_thinking:
                                                    done_thinking = True
                                                    if self.thoughts_log and self.thinking:
                                                        if retry == 0:
                                                            info_with_progress(
                                                                f"Batch {self.batch_number} thinking process saved to file.",
                                                                isTranscribing=True,
                                                            )
                                                        else:
                                                            info_with_progress(
                                                                f"Batch {self.batch_number}.{retry} thinking process saved to file.",
                                                                isTranscribing=True,
                                                            )
                                                        save_thoughts_to_file(
                                                            thoughts_text,
                                                            self.thoughts_file_path,
                                                            retry,
                                                        )
                                                response_text += part.text
                                                transcription_json = json_repair.loads(
                                                    response_text, stream_stable=True
                                                )
                                                if not isinstance(transcription_json, list) or not all(
                                                    isinstance(item, dict) for item in transcription_json
                                                ):
                                                    transcription_json = self._flatten_repaired_json(transcription_json)
                                                if len(transcription_json) > 1:
                                                    if "time_end" in transcription_json[-2]:
                                                        ts_for_anim = _normalize_timestamp(
                                                            transcription_json[-2]["time_end"]
                                                        )
                                                        processed_seconds = convert_timestamp_to_timedelta(
                                                            ts_for_anim
                                                        ).total_seconds()

                                        update_loading_animation(
                                            chunk_size=processed_seconds if processed_seconds else 0,
                                            isTranscribing=True,
                                            isThinking=self.thinking and not done_thinking,
                                            token_stats=self.token_stats,
                                            prompt_tokens=(
                                                chunk.usage_metadata.prompt_token_count - previous_prompt_tokens
                                                if chunk.usage_metadata and chunk.usage_metadata.prompt_token_count
                                                else None
                                            ),
                                            thoughts_tokens=(
                                                chunk.usage_metadata.thoughts_token_count - previous_thoughts_tokens
                                                if chunk.usage_metadata and chunk.usage_metadata.thoughts_token_count
                                                else None
                                            ),
                                            output_tokens=(
                                                chunk.usage_metadata.candidates_token_count - previous_output_tokens
                                                if chunk.usage_metadata and chunk.usage_metadata.candidates_token_count
                                                else None
                                            ),
                                            total_tokens=(
                                                chunk.usage_metadata.total_token_count - previous_total_tokens
                                                if chunk.usage_metadata and chunk.usage_metadata.total_token_count
                                                else None
                                            ),
                                        )
                                        self._accumulate_report_tokens(
                                            prompt_tokens=(
                                                chunk.usage_metadata.prompt_token_count - previous_prompt_tokens
                                                if chunk.usage_metadata and chunk.usage_metadata.prompt_token_count
                                                else None
                                            ),
                                            thoughts_tokens=(
                                                chunk.usage_metadata.thoughts_token_count - previous_thoughts_tokens
                                                if chunk.usage_metadata and chunk.usage_metadata.thoughts_token_count
                                                else None
                                            ),
                                            output_tokens=(
                                                chunk.usage_metadata.candidates_token_count - previous_output_tokens
                                                if chunk.usage_metadata and chunk.usage_metadata.candidates_token_count
                                                else None
                                            ),
                                            total_tokens=(
                                                chunk.usage_metadata.total_token_count - previous_total_tokens
                                                if chunk.usage_metadata and chunk.usage_metadata.total_token_count
                                                else None
                                            ),
                                        )
                                        previous_prompt_tokens = (
                                            chunk.usage_metadata.prompt_token_count
                                            if chunk.usage_metadata and chunk.usage_metadata.prompt_token_count
                                            else 0
                                        )
                                        previous_thoughts_tokens = (
                                            chunk.usage_metadata.thoughts_token_count
                                            if chunk.usage_metadata and chunk.usage_metadata.thoughts_token_count
                                            else 0
                                        )
                                        previous_output_tokens = (
                                            chunk.usage_metadata.candidates_token_count
                                            if chunk.usage_metadata and chunk.usage_metadata.candidates_token_count
                                            else 0
                                        )
                                        previous_total_tokens = (
                                            chunk.usage_metadata.total_token_count
                                            if chunk.usage_metadata and chunk.usage_metadata.total_token_count
                                            else 0
                                        )

                            if blocked:
                                raise Exception("Content blocked by the API.")

                            transcription_json = json_repair.loads(response_text)
                            if not isinstance(transcription_json, list) or not all(
                                isinstance(item, dict) for item in transcription_json
                            ):
                                transcription_json = self._flatten_repaired_json(transcription_json)

                            for i in range(len(transcription_json)):
                                start_ts = _normalize_timestamp(transcription_json[i]["time_start"])
                                end_ts = _normalize_timestamp(transcription_json[i]["time_end"])
                                subtitle_kwargs = {
                                    "index": str(index),
                                    "content": transcription_json[i]["text"],
                                    "start": convert_timestamp_to_timedelta(start_ts, offset=current_length),
                                    "end": convert_timestamp_to_timedelta(end_ts, offset=current_length),
                                }
                                if self._dominant_strong_direction(subtitle_kwargs["content"]) == "rtl":
                                    subtitle_kwargs["content"] = f"\u202b{subtitle_kwargs['content']}\u202c"
                                transcribed_subtitle_objects.append(Subtitle(**subtitle_kwargs))
                                index += 1

                            done = True

                        chunk_processed_successfully = True

                    except Exception as e:
                        e_str = str(e).lower()

                        if "quota" in e_str:
                            current_time = time.time()
                            if current_time - last_time > 60 and self._switch_api():
                                highlight_with_progress(
                                    f"API {self.backup_api_number} quota exceeded! Switching to API {self.current_api_number}...",
                                    isSending=True,
                                )
                            else:
                                for j in range(60, 0, -1):
                                    warning_with_progress(f"All API quotas exceeded, waiting {j} seconds...")
                                    time.sleep(1)
                            last_time = current_time
                            continue

                        elif any(err in e_str for err in ["429", "500", "503", "unavailable", "overloaded"]):
                            server_error_retries += 1
                            if server_error_retries < max_retries:
                                warning_with_progress(
                                    f"Model is overloaded. Attempt {server_error_retries + 1}/{max_retries}. Pausing for 60 seconds...",
                                    isTranscribing=True,
                                )
                                time.sleep(60)
                                continue
                            else:
                                error_with_progress(
                                    f"Failed to process segment after multiple retries. Aborting...",
                                    isTranscribing=True,
                                )
                                raise e

                        else:
                            warning_with_progress(
                                f"An unexpected error occurred: {e_str}. Retrying immediately...",
                                isTranscribing=True,
                            )
                            time.sleep(1)
                            continue

                server_error_retries = 0

                current_length = chunk_end
                last_saved_time = current_length
                self._save_transcribe_progress(current_length)
                self.batch_number += 1
                if self.progress_log:
                    save_logs_to_file(self.log_file_path)

            transcribed_subtitle = srt.compose(transcribed_subtitle_objects)
            progress_bar(
                audio_length,
                audio_length,
                prefix="Transcribing:",
                suffix=(f"\033[31m{self.model_name}" if self.use_colors else f"{self.model_name}"),
                isTranscribing=True,
            )
            with open(self.output_file, "w", encoding="utf-8") as f:
                f.write(transcribed_subtitle)

            if self.use_colors:
                success_with_progress(
                    f"\n\033[96mTranscription saved to\033[92m {self.output_file}",
                    isTranscribing=True,
                )
            else:
                success_with_progress(f"\nTranscription saved to {self.output_file}", isTranscribing=True)

            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)
            if self.progress_log:
                save_logs_to_file(self.log_file_path)
            self._write_token_report("transcribe")
            if extracted and self.audio_file and os.path.exists(self.audio_file):
                os.remove(self.audio_file)

        except Exception as e:
            error(f"Error during transcription: {e}", ignore_quiet=True)
            self._save_transcribe_progress(last_saved_time)
            self._write_token_report("transcribe")
            exit(1)
