#!/usr/bin/env python3
"""
Command Line Interface for Gemini SRT Translator
"""

import argparse
import getpass
import os
import sys
from typing import Optional

import gemini_srt_translator as gst

from .logger import error, info, success


def get_api_key_from_input() -> str:
    """Get API key from user input."""
    return getpass.getpass("Enter your Gemini API key: ").strip()


def get_api_key_from_env(key: str) -> Optional[str]:
    """Get API key from environment variable."""
    api_key = os.getenv(key, None)
    return api_key.strip() if api_key else None


def validate_file_path(file_path: str, extension: str = None) -> bool:
    """Validate if file exists and has correct extension."""
    if not os.path.isfile(file_path):
        error(f"File does not exist: {file_path}")
        return False

    if extension and not file_path.lower().endswith(extension):
        error(f"File must have {extension} extension: {file_path}")
        return False

    return True


def select_model_interactive(available_models: list) -> str:
    """Interactive model selection."""
    print("\nAvailable models:")
    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model}")

    while True:
        try:
            choice = int(input("\nEnter model number: "))
            if 1 <= choice <= len(available_models):
                return available_models[choice - 1]
            else:
                error("Invalid choice. Please try again.")
        except ValueError:
            error("Please enter a valid number.")


def cmd_translate(args) -> None:
    """Handle translate command."""
    # Set API keys
    gst.gemini_api_key = args.api_key or get_api_key_from_env("GEMINI_API_KEY") or get_api_key_from_input()
    gst.gemini_api_key2 = args.api_key2 or get_api_key_from_env("GEMINI_API_KEY2")

    # Validate input file
    if args.input_file:
        if not validate_file_path(args.input_file, ".srt"):
            sys.exit(1)
        gst.input_file = args.input_file
    if args.video_file:
        if not validate_file_path(args.video_file):
            sys.exit(1)
        gst.video_file = args.video_file
    if not args.input_file and not args.video_file:
        error("Either --input-file and/or --video-file must be provided.")
        sys.exit(1)

    # Set target language
    gst.target_language = args.target_language or input("Enter target language: ").strip()

    # Model selection
    if args.model:
        gst.model_name = args.model
    elif args.interactive:
        available_models = gst.getmodels()
        gst.model_name = select_model_interactive(available_models)

    # Set optional parameters
    if args.output_file:
        gst.output_file = args.output_file
    if args.audio_file:
        if validate_file_path(args.audio_file):
            gst.audio_file = args.audio_file
    if args.extract_audio:
        gst.extract_audio = args.extract_audio
    if args.audio_chunk_size:
        gst.audio_chunk_size = args.audio_chunk_size
    if args.start_line:
        gst.start_line = args.start_line
    if args.description:
        gst.description = args.description
    if args.batch_size:
        gst.batch_size = args.batch_size
    if args.temperature:
        gst.temperature = args.temperature
    if args.top_p:
        gst.top_p = args.top_p
    if args.top_k:
        gst.top_k = args.top_k
    if args.thinking_budget:
        gst.thinking_budget = args.thinking_budget
    if args.thinking_level:
        gst.thinking_level = args.thinking_level

    # Set boolean flags
    if args.no_voice_isolation:
        gst.isolate_voice = not args.no_voice_isolation
    if args.no_streaming:
        gst.streaming = not args.no_streaming
    if args.no_thinking:
        gst.thinking = not args.no_thinking
    if args.paid_quota:
        gst.free_quota = not args.paid_quota
    if args.skip_upgrade:
        gst.skip_upgrade = args.skip_upgrade
    if args.no_colors:
        gst.use_colors = not args.no_colors
    if args.progress_log:
        gst.progress_log = args.progress_log
    if args.thoughts_log:
        gst.thoughts_log = args.thoughts_log
    if args.quiet:
        gst.quiet = args.quiet
    if args.resume:
        gst.resume = args.resume
    if args.include_timestamps:
        gst.include_timestamps = args.include_timestamps

    # Execute translation
    try:
        gst.translate()
    except Exception as e:
        error(f"Translation failed: {e}")
        sys.exit(1)


def cmd_list_models(args) -> None:
    """Handle list-models command."""
    gst.gemini_api_key = args.api_key or get_api_key_from_env("GEMINI_API_KEY") or get_api_key_from_input()

    try:
        gst.listmodels()
    except Exception as e:
        error(f"Failed to list models: {e}")
        sys.exit(1)


def cmd_extract(args) -> None:
    """Handle extract command."""
    if args.video_file:
        if not validate_file_path(args.video_file):
            sys.exit(1)
        gst.video_file = args.video_file
    if args.isolate_voice:
        gst.isolate_voice = args.isolate_voice

    if args.srt:
        try:
            gst.extract("srt")
        except Exception as e:
            error(f"SRT extraction failed: {e}")
            sys.exit(1)

    if args.audio:
        try:
            gst.extract("audio")
        except Exception as e:
            error(f"Audio extraction failed: {e}")
            sys.exit(1)


def cmd_transcribe(args) -> None:
    """Handle transcribe command."""
    gst.gemini_api_key = args.api_key or get_api_key_from_env("GEMINI_API_KEY") or get_api_key_from_input()

    if args.video_file:
        if not validate_file_path(args.video_file):
            sys.exit(1)
        gst.video_file = args.video_file
    if args.audio_file:
        if not validate_file_path(args.audio_file):
            sys.exit(1)
        gst.audio_file = args.audio_file
    if args.output_file:
        gst.output_file = args.output_file
    if args.description:
        gst.description = args.description
    if args.audio_chunk_size:
        gst.audio_chunk_size = args.audio_chunk_size
    if args.thinking_budget:
        gst.thinking_budget = args.thinking_budget
    if args.temperature:
        gst.temperature = args.temperature
    if args.top_p:
        gst.top_p = args.top_p
    if args.top_k:
        gst.top_k = args.top_k
    if args.no_streaming:
        gst.streaming = not args.no_streaming
    if args.no_thinking:
        gst.thinking = not args.no_thinking
    if args.no_colors:
        gst.use_colors = not args.no_colors
    if args.progress_log:
        gst.progress_log = args.progress_log
    if args.thoughts_log:
        gst.thoughts_log = args.thoughts_log

    # Model selection
    if args.model:
        gst.model_name = args.model
    elif args.interactive:
        available_models = gst.getmodels()
        gst.model_name = select_model_interactive(available_models)

    # Execute transcription
    try:
        gst.transcribe()
    except Exception as e:
        error(f"Transcription failed: {e}")
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        prog="gst",
        description="Translate SRT subtitle files using Google Gemini AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using environment variable (recommended)
    export GEMINI_API_KEY="your_api_key_here"
    gst translate -i subtitle.srt -l French

  # Using command line argument
    gst translate -i subtitle.srt -l French -k YOUR_API_KEY

  # Set output file name
    gst translate -i subtitle.srt -l French -o translated_subtitle.srt

  # Extract subtitles from video and translate (requires FFmpeg)
    gst translate -v movie.mp4 -l Spanish

  # Extract and use audio from video for context (requires FFmpeg)
    gst translate -v movie.mp4 -l Spanish --extract-audio

  # Interactive model selection
    gst translate -i subtitle.srt -l "Brazilian Portuguese" --interactive

  # Resume translation from a specific line
    gst translate -i subtitle.srt -l French --start-line 20

  # Suppress output
    gst translate -i subtitle.srt -l French --quiet
  
  # List available models
    gst list-models -k YOUR_API_KEY

  # Extract audio and SRT from video with voice isolation
    gst extract -v movie.mp4 --srt --audio --isolate-voice

  # Extract audio from video and transcribe to subtitle file
    gst transcribe -v movie.mp4 -k YOUR_API_KEY -o transcription.srt
    """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Translate command
    translate_parser = subparsers.add_parser("translate", help="Translate subtitle files")
    required_group_translate = translate_parser.add_argument_group("required arguments")
    required_group_translate.add_argument("-i", "--input-file", help="Input SRT file path")
    required_group_translate.add_argument("-v", "--video-file", help="Video file path (for SRT/Audio extraction)")
    translate_parser.add_argument("-l", "--target-language", help="Target language for translation")
    translate_parser.add_argument("-k", "--api-key", help="Gemini API key")
    translate_parser.add_argument("-k2", "--api-key2", help="Secondary Gemini API key for additional quota")
    translate_parser.add_argument("-o", "--output-file", help="Output file path")
    translate_parser.add_argument("-a", "--audio-file", help="Audio file for context")
    translate_parser.add_argument("-s", "--start-line", type=int, help="Starting line number")
    translate_parser.add_argument("-d", "--description", help="Description for translation context")
    translate_parser.add_argument("-m", "--model", help="Gemini model to use")
    translate_parser.add_argument("-b", "--batch-size", type=int, help="Batch size for translation")
    translate_parser.add_argument("--audio-chunk-size", type=int, help="Audio chunk size for processing in seconds")
    translate_parser.add_argument("--temperature", type=float, help="Temperature (0.0-2.0)")
    translate_parser.add_argument("--top-p", type=float, help="Top P (0.0-1.0)")
    translate_parser.add_argument("--top-k", type=int, help="Top K (>=0)")
    translate_parser.add_argument("--thinking-budget", type=int, help="Thinking budget (0-32768)")
    translate_parser.add_argument("--thinking-level", type=str, help="Thinking level (minimal, low, medium, high)")
    translate_parser.add_argument("--no-streaming", action="store_true", default=None, help="Disable streaming")
    translate_parser.add_argument("--no-thinking", action="store_true", default=None, help="Disable thinking mode")
    translate_parser.add_argument("--skip-upgrade", action="store_true", default=None, help="Skip upgrade check")
    translate_parser.add_argument("--no-colors", action="store_true", default=None, help="Disable colored output")
    translate_parser.add_argument("--progress-log", action="store_true", default=None, help="Enable progress logging")
    translate_parser.add_argument("--thoughts-log", action="store_true", default=None, help="Enable thoughts logging")
    translate_parser.add_argument("--quiet", action="store_true", default=None, help="Suppress output")
    translate_parser.add_argument("--resume", action="store_true", default=None, help="Resume interrupted translation")
    translate_parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start from beginning")
    translate_parser.add_argument(
        "--no-voice-isolation", action="store_true", default=None, help="Don't isolate voice in audio extraction"
    )
    translate_parser.add_argument(
        "--paid-quota", action="store_true", default=None, help="Remove artificial limits for paid quota users"
    )
    translate_parser.add_argument(
        "--interactive", action="store_true", default=None, help="Interactive model selection"
    )
    translate_parser.add_argument(
        "--extract-audio", action="store_true", default=None, help="Extract audio from video for context"
    )
    translate_parser.add_argument(
        "--include-timestamps", action="store_true", default=None,
        help="Include timestamps in translation for context matching with description"
    )

    # Extract audio command
    extract_parser = subparsers.add_parser("extract", help="Extract audio and/or srt from video file")
    required_group_extract = extract_parser.add_argument_group("required arguments")
    required_group_extract.add_argument("-v", "--video-file", help="Video file path")
    extract_parser.add_argument("--srt", action="store_true", default=None, help="Extract SRT subtitles from video")
    extract_parser.add_argument("--audio", action="store_true", default=None, help="Extract audio from video")
    extract_parser.add_argument(
        "--isolate-voice", action="store_true", default=None, help="Isolate voice in audio extraction"
    )
    extract_parser.add_argument("--no-colors", action="store_true", default=None, help="Disable colored output")

    # Transcribe command
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe audio to text")
    required_group_transcribe = transcribe_parser.add_argument_group("required arguments")
    required_group_transcribe.add_argument("-v", "--video-file", help="Video file path for audio extraction")
    required_group_transcribe.add_argument("-a", "--audio-file", help="Audio file path for transcription")
    transcribe_parser.add_argument("-k", "--api-key", help="Gemini API key")
    transcribe_parser.add_argument("-o", "--output-file", help="Output file path for transcription results")
    transcribe_parser.add_argument("-m", "--model", help="Gemini model to use")
    transcribe_parser.add_argument("-d", "--description", help="Description for transcription context")
    transcribe_parser.add_argument("--audio-chunk-size", type=int, help="Audio chunk size for processing in seconds")
    transcribe_parser.add_argument("--thinking-budget", type=int, help="Thinking budget (0-32768)")
    transcribe_parser.add_argument("--thinking-level", type=str, help="Thinking level (minimal, low, medium, high)")
    transcribe_parser.add_argument("--temperature", type=float, help="Temperature (0.0-2.0)")
    transcribe_parser.add_argument("--top-p", type=float, help="Top P (0.0-1.0)")
    transcribe_parser.add_argument("--top-k", type=int, help="Top K (>=0)")
    transcribe_parser.add_argument("--no-streaming", action="store_true", default=None, help="Disable streaming")
    transcribe_parser.add_argument("--no-thinking", action="store_true", default=None, help="Disable thinking mode")
    transcribe_parser.add_argument("--no-colors", action="store_true", default=None, help="Disable colored output")
    transcribe_parser.add_argument("--progress-log", action="store_true", default=None, help="Enable progress logging")
    transcribe_parser.add_argument("--thoughts-log", action="store_true", default=None, help="Enable thoughts logging")
    transcribe_parser.add_argument(
        "--interactive", action="store_true", default=None, help="Interactive model selection"
    )

    # List models command
    list_parser = subparsers.add_parser("list-models", help="List available Gemini models")
    list_parser.add_argument("-k", "--api-key", help="Gemini API key")

    return parser


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "translate":
            cmd_translate(args)
        elif args.command == "list-models":
            cmd_list_models(args)
        elif args.command == "extract":
            cmd_extract(args)
        elif args.command == "transcribe":
            cmd_transcribe(args)
    except KeyboardInterrupt:
        info("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
