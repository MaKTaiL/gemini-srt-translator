"""
# Gemini SRT Translator
    A tool to translate subtitles using Google Generative AI.

## Usage:

### Translate Subtitles
    You can translate subtitles using the `translate` command:
    ```
    import gemini_srt_translator as gst

    gst.gemini_api_key = "your_gemini_api_key_here"
    gst.target_language = "French"
    gst.input_file = "subtitle.srt"

    gst.translate()
    ```
    This will translate the subtitles in the `subtitle.srt` file to French.

### List Models
    You can list the available models using the `listmodels` command:
    ```
    import gemini_srt_translator as gst

    gst.gemini_api_key = "your_gemini_api_key_here"
    gst.listmodels()
    ```
    This will print a list of available models to the console.

"""

import os
from typing import Literal

from .logger import set_quiet_mode
from .main import GeminiSRTTranslator
from .utils import upgrade_package

gemini_api_key: str = os.getenv("GEMINI_API_KEY", None)
gemini_api_key2: str = os.getenv("GEMINI_API_KEY2", None)
target_language: str = None
input_file: str = None
output_file: str = None
video_file: str = None
audio_file: str = None
audio_chunk_size: int = None
extract_audio: bool = None
isolate_voice: bool = None
start_line: int = None
description: str = None
include_timestamps: bool = False
model_name: str = None
batch_size: int = None
streaming: bool = None
thinking: bool = None
thinking_budget: int = None
thinking_level: Literal["minimal", "low", "medium", "high"] = None
temperature: float = None
top_p: float = None
top_k: int = None
free_quota: bool = None
skip_upgrade: bool = None
use_colors: bool = True
progress_log: bool = None
thoughts_log: bool = None
quiet: bool = None
resume: bool = None

if not skip_upgrade:
    try:
        upgrade_package("gemini-srt-translator", use_colors=use_colors)
        raise Exception("Upgrade completed.")
    except Exception:
        pass


def getmodels():
    """
    ## Retrieves available models from the Gemini API.
        This function configures the genai library with the provided Gemini API key
        and retrieves a list of available models.

    Example:
    ```
    import gemini_srt_translator as gst

    # Your Gemini API key
    gst.gemini_api_key = "your_gemini_api_key_here"

    models = gst._getmodels()
    print(models)
    ```

    Raises:
        Exception: If the Gemini API key is not provided.
    """
    translator = GeminiSRTTranslator(gemini_api_key=gemini_api_key)
    return translator.getmodels()


def listmodels():
    """
    ## Lists available models from the Gemini API.
        This function configures the genai library with the provided Gemini API key
        and retrieves a list of available models. It then prints each model to the console.

    Example:
    ```
    import gemini_srt_translator as gst

    # Your Gemini API key
    gst.gemini_api_key = "your_gemini_api_key_here"

    gst.listmodels()
    ```

    Raises:
        Exception: If the Gemini API key is not provided.
    """
    translator = GeminiSRTTranslator(gemini_api_key=gemini_api_key)
    models = translator.getmodels()
    if models:
        print("Available models:\n")
        for model in models:
            print(model)
    else:
        print("No models available or an error occurred while fetching models.")


def translate():
    """
    ## Translates a subtitle file using the Gemini API.
        This function configures the genai library with the provided Gemini API key
        and translates the dialogues in the subtitle file to the target language.
        The translated dialogues are then written to a new subtitle file.

    Example:
    ```
    import gemini_srt_translator as gst

    # Your Gemini API key
    gst.gemini_api_key = "your_gemini_api_key_here"

    # Target language for translation
    gst.target_language = "French"

    # Path to the subtitle file to translate
    gst.input_file = "subtitle.srt"

    # (Optional) Gemini API key 2 for additional quota
    gst.gemini_api_key2 = "your_gemini_api_key2_here"

    # (Optional) Path to video file for srt extraction (if needed) and/or for audio context
    gst.video_file = "movie.mkv"

    # (Optional) Path to audio file for audio context
    gst.audio_file = "audio.mp3"

    # (Optional) Audio chunk size for processing (default: 600 seconds)
    gst.audio_chunk_size = 600

    # (Optional) Whether to extract and use audio context from video file
    gst.extract_audio = False

    # (Optional) Whether to isolate voice from the audio context
    gst.isolate_voice = False

    # (Optional) Path to save the translated subtitle file
    gst.output_file = "translated_subtitle.srt"

    # (Optional) Line number to start translation from
    gst.start_line = 120

    # (Optional) Additional description of the translation task
    gst.description = "This subtitle is from a TV Series called 'Friends'."

    # (Optional) Model name to use for translation (default: "gemini-2.5-flash")
    gst.model_name = "gemini-2.5-flash"

    # (Optional) Batch size for translation (default: 300)
    gst.batch_size = 300

    # (Optional) Whether to use streamed responses (default: True)
    gst.streaming = True

    # (Optional) Whether to use thinking (default: True)
    gst.thinking = True

    # (Optional) Thinking budget for translation (default: 2048, range: 0-24576, 0 disables thinking)
    gst.thinking_budget = 2048

    # (Optional) Temperature for the translation model (range: 0.0-2.0)
    gst.temperature = 0.5

    # (Optional) Top P for the translation model (range: 0.0-1.0)
    gst.top_p = 0.9

    # (Optional) Top K for the translation model (range: >=0)
    gst.top_k = 10

    # (Optional) Signal GST that you are using the free quota (default: True)
    gst.free_quota = True

    # (Optional) Skip package upgrade check (default: False)
    gst.skip_upgrade = False

    # (Optional) Use colors in the output (default: True)
    gst.use_colors = True

    # (Optional) Enable progress logging (default: False)
    gst.progress_log = False

    # (Optional) Enable thoughts logging (default: False)
    gst.thoughts_log = False

    # (Optional) Enable quiet mode (default: False)
    gst.quiet = False

    # (Optional) Skip prompt and set automatic resume mode
    gst.resume = False

    gst.translate()
    ```
    Raises:
        Exception: If the Gemini API key is not provided.
        Exception: If the target language is not provided.
        Exception: If the subtitle file is not provided.
    """
    params = {
        "gemini_api_key": gemini_api_key,
        "gemini_api_key2": gemini_api_key2,
        "target_language": target_language,
        "input_file": input_file,
        "output_file": output_file,
        "video_file": video_file,
        "audio_file": audio_file,
        "audio_chunk_size": audio_chunk_size,
        "extract_audio": extract_audio,
        "isolate_voice": isolate_voice,
        "start_line": start_line,
        "description": description,
        "include_timestamps": include_timestamps,
        "model_name": model_name,
        "batch_size": batch_size,
        "streaming": streaming,
        "thinking": thinking,
        "thinking_budget": thinking_budget,
        "thinking_level": thinking_level,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "free_quota": free_quota,
        "use_colors": use_colors,
        "progress_log": progress_log,
        "thoughts_log": thoughts_log,
        "resume": resume,
    }

    if quiet:
        set_quiet_mode(quiet)

    # Filter out None values
    filtered_params = {k: v for k, v in params.items() if v is not None}
    translator = GeminiSRTTranslator(**filtered_params)
    translator.translate()


def extract(type: Literal["audio", "srt"]):
    """
    ## Extracts audio or subtitles (SRT) from a video file.
        This function extracts either the audio or the subtitles from the specified video file.
        The extraction is done using FFmpeg.

    Args:
        type (str): The type of extraction. Must be "audio" or "srt".

    Example:
    ```
    import gemini_srt_translator as gst

    # Path to the video file
    gst.video_file = "movie.mkv"

    # (Optional) Whether to isolate voice from the audio context
    gst.isolate_voice = True

    # Extract audio
    gst.extract("audio")

    # Extract subtitles
    gst.extract("srt")
    ```

    Raises:
        Exception: If the video file is not provided.
        ValueError: If the type is not "audio" or "srt".
    """

    params = {
        "video_file": video_file,
        "isolate_voice": isolate_voice,
        "use_colors": use_colors,
    }

    if params["isolate_voice"] is None:
        params["isolate_voice"] = False
    filtered_params = {k: v for k, v in params.items() if v is not None}
    translator = GeminiSRTTranslator(**filtered_params)
    if type == "audio":
        translator.extract("audio")
    elif type == "srt":
        translator.extract("srt")
    else:
        raise ValueError("Invalid extraction type. Must be 'audio' or 'srt'.")


def transcribe():
    """
    ## Transcribes audio from a video file using the Gemini API.
        This function configures the genai library with the provided Gemini API key
        and transcribes the audio from the video file to a subtitle file.

    Example:
    ```
    import gemini_srt_translator as gst

    # Your Gemini API key
    gst.gemini_api_key = "your_gemini_api_key_here"

    # Path to the audio file to transcribe
    gst.audio_file = "audio.mp3"

    # (Optional) Path to save the transcription output
    gst.output_file = "transcription.srt"

    # (Optional) Resume transcription if progress is found
    gst.resume = True

    gst.transcribe()
    ```

    Raises:
        Exception: If the Gemini API key is not provided.
        Exception: If the audio file is not provided.
    """

    params = {
        "gemini_api_key": gemini_api_key,
        "gemini_api_key2": gemini_api_key2,
        "video_file": video_file,
        "audio_file": audio_file,
        "model_name": model_name,
        "description": description,
        "audio_chunk_size": audio_chunk_size,
        "isolate_voice": isolate_voice,
        "output_file": output_file,
        "streaming": streaming,
        "thinking": thinking,
        "thinking_budget": thinking_budget,
        "thinking_level": thinking_level,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "use_colors": use_colors,
        "thoughts_log": thoughts_log,
        "resume": resume,
    }

    if params["isolate_voice"] is None:
        params["isolate_voice"] = False

    filtered_params = {k: v for k, v in params.items() if v is not None}
    translator = GeminiSRTTranslator(**filtered_params)
    translator.transcribe()
