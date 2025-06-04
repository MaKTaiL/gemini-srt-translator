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
    elif args.video_file:
        if not validate_file_path(args.video_file):
            sys.exit(1)
        gst.video_file = args.video_file
    else:
        error("Either --input-file or --video-file must be provided.")
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
    if args.start_line:
        gst.start_line = args.start_line
    if args.description:
        gst.description = args.description
    if args.batch_size:
        gst.batch_size = args.batch_size
    if args.temperature is not None:
        gst.temperature = args.temperature
    if args.top_p is not None:
        gst.top_p = args.top_p
    if args.top_k is not None:
        gst.top_k = args.top_k
    if args.thinking_budget is not None:
        gst.thinking_budget = args.thinking_budget

    # Set boolean flags
    gst.streaming = not args.no_streaming
    gst.thinking = not args.no_thinking
    gst.free_quota = not args.paid_quota
    gst.skip_upgrade = args.skip_upgrade
    gst.use_colors = not args.no_colors
    gst.progress_log = args.progress_log
    gst.thoughts_log = args.thoughts_log
    gst.quiet = args.quiet
    if args.resume is not None:
        gst.resume = args.resume

    # Execute translation
    try:
        gst.translate()
        success("Translation completed successfully!")
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

  # Extract subtitles from video and translate
    gst translate -v movie.mp4 -l Spanish

  # Interactive model selection
    gst translate -i subtitle.srt -l Portuguese --interactive

  # Resume translation from a specific line
    gst translate -i subtitle.srt -l French --start-line 20

  # Suppress output
    gst translate -i subtitle.srt -l French --quiet
  
  # List available models
    gst list-models -k YOUR_API_KEY
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Translate command
    translate_parser = subparsers.add_parser("translate", help="Translate subtitle files")

    # Required arguments group
    required_group = translate_parser.add_argument_group("required arguments")
    input_group = translate_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-i", "--input-file", help="Input SRT file path")
    input_group.add_argument("-v", "--video-file", help="Video file path (for SRT extraction)")

    translate_parser.add_argument("-l", "--target-language", help="Target language for translation")
    translate_parser.add_argument("-k", "--api-key", help="Gemini API key")

    # Optional arguments
    translate_parser.add_argument("-k2", "--api-key2", help="Secondary Gemini API key for additional quota")
    translate_parser.add_argument("-o", "--output-file", help="Output file path")
    translate_parser.add_argument("-a", "--audio-file", help="Audio file for context")
    translate_parser.add_argument("--extract-audio", action="store_true", help="Extract audio from video for context")
    translate_parser.add_argument("-s", "--start-line", type=int, help="Starting line number")
    translate_parser.add_argument("-d", "--description", help="Description for translation context")
    translate_parser.add_argument("-m", "--model", help="Gemini model to use")
    translate_parser.add_argument("--interactive", action="store_true", help="Interactive model selection")
    translate_parser.add_argument("-b", "--batch-size", type=int, help="Batch size for translation")
    translate_parser.add_argument("--temperature", type=float, help="Temperature (0.0-2.0)")
    translate_parser.add_argument("--top-p", type=float, help="Top P (0.0-1.0)")
    translate_parser.add_argument("--top-k", type=int, help="Top K (>=0)")
    translate_parser.add_argument("--thinking-budget", type=int, help="Thinking budget (0-24576)")

    # Boolean flags
    translate_parser.add_argument("--no-streaming", action="store_true", help="Disable streaming")
    translate_parser.add_argument("--no-thinking", action="store_true", help="Disable thinking mode")
    translate_parser.add_argument(
        "--paid-quota", action="store_true", help="Remove artificial limits for paid quota users"
    )
    translate_parser.add_argument("--skip-upgrade", action="store_true", help="Skip upgrade check")
    translate_parser.add_argument("--no-colors", action="store_true", help="Disable colored output")
    translate_parser.add_argument("--progress-log", action="store_true", help="Enable progress logging")
    translate_parser.add_argument("--thoughts-log", action="store_true", help="Enable thoughts logging")
    translate_parser.add_argument("--quiet", action="store_true", help="Suppress output")
    translate_parser.add_argument("--resume", action="store_true", help="Resume interrupted translation")
    translate_parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start from beginning")

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
    except KeyboardInterrupt:
        info("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
