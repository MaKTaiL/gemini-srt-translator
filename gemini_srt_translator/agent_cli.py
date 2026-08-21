"""
Agent CLI Module for Gemini SRT Translator
Provides structured JSON-based CLI subcommands for AI Agents to interact with the subtitle translation & transcription pipelines.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

from .session import SubtitleSession


def _sanitize_for_json(data: Any) -> Any:
    """Recursively sanitize data structures for JSON serialization (stripping raw bytes)."""
    if isinstance(data, dict):
        return {k: _sanitize_for_json(v) for k, v in data.items() if k != "audio_bytes"}
    elif isinstance(data, list):
        return [_sanitize_for_json(item) for item in data]
    elif isinstance(data, bytes):
        return None
    return data


def _print_json(data: Dict[str, Any], pretty: bool = False):
    """Output JSON to stdout for machine consumption."""
    clean_data = _sanitize_for_json(data)
    if pretty:
        print(json.dumps(clean_data, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(clean_data, ensure_ascii=False))


# ==============================================================================
# Subtitle Translation Commands
# ==============================================================================


def cmd_agent_start(args) -> int:
    """Initialize a subtitle translation session and return the first batch."""
    try:
        session = SubtitleSession(
            input_file=args.input_file,
            target_language=args.target_language,
            output_file=args.output_file,
            video_file=getattr(args, "video_file", None),
            audio_file=getattr(args, "audio_file", None),
            batch_size=getattr(args, "batch_size", 100),
            resume_context_size=getattr(args, "context_size", 20),
            description=getattr(args, "description", None),
            resume=getattr(args, "resume", True),
        )

        next_batch = session.get_next_batch()
        _print_json(
            {
                "status": "ready" if next_batch else "completed",
                "session": session.get_status(),
                "next_batch": next_batch,
            },
            pretty=args.pretty,
        )
        return 0
    except Exception as e:
        _print_json({"status": "error", "error": str(e)}, pretty=args.pretty)
        return 1


def cmd_agent_next(args) -> int:
    """Get the current pending batch for an in-progress subtitle file."""
    try:
        session = SubtitleSession(
            input_file=args.input_file,
            target_language=getattr(args, "target_language", "English") or "English",
            output_file=getattr(args, "output_file", None),
            batch_size=getattr(args, "batch_size", 100),
            resume=True,
        )

        next_batch = session.get_next_batch()
        _print_json(
            {
                "status": "ok" if next_batch else "completed",
                "session": session.get_status(),
                "next_batch": next_batch,
            },
            pretty=args.pretty,
        )
        return 0
    except Exception as e:
        _print_json({"status": "error", "error": str(e)}, pretty=args.pretty)
        return 1


def cmd_agent_commit(args) -> int:
    """Commit a translated batch, save to file, and return the next batch."""
    try:
        data = None
        if getattr(args, "data", None):
            data = args.data
        elif getattr(args, "data_file", None):
            if not os.path.exists(args.data_file):
                _print_json({"status": "error", "error": f"File not found: {args.data_file}"}, pretty=args.pretty)
                return 1
            with open(args.data_file, "r", encoding="utf-8") as f:
                data = f.read()
        else:
            if not sys.stdin.isatty():
                data = sys.stdin.read()

        if not data:
            _print_json(
                {
                    "status": "error",
                    "error": "No translation data provided. Use --data, --data-file, or pipe via stdin.",
                },
                pretty=args.pretty,
            )
            return 1

        session = SubtitleSession(
            input_file=args.input_file,
            target_language=getattr(args, "target_language", "English") or "English",
            output_file=getattr(args, "output_file", None),
            batch_size=getattr(args, "batch_size", 100),
            resume=True,
        )

        commit_result = session.commit_batch(data)
        if not commit_result.get("success"):
            _print_json(
                {
                    "status": "error",
                    "error": commit_result.get("error", "Validation failed"),
                    "session": session.get_status(),
                },
                pretty=args.pretty,
            )
            return 1

        next_batch = session.get_next_batch()
        _print_json(
            {
                "status": "completed" if session.is_complete() else "committed",
                "commit_result": commit_result,
                "session": session.get_status(),
                "next_batch": next_batch,
            },
            pretty=args.pretty,
        )
        return 0
    except Exception as e:
        _print_json({"status": "error", "error": str(e)}, pretty=args.pretty)
        return 1


def cmd_agent_status(args) -> int:
    """Get translation session status."""
    try:
        session = SubtitleSession(
            input_file=args.input_file,
            output_file=getattr(args, "output_file", None),
            resume=True,
        )
        _print_json(
            {
                "status": "ok",
                "session": session.get_status(),
            },
            pretty=args.pretty,
        )
        return 0
    except Exception as e:
        _print_json({"status": "error", "error": str(e)}, pretty=args.pretty)
        return 1


def cmd_agent_reset(args) -> int:
    """Reset translation progress."""
    try:
        session = SubtitleSession(
            input_file=args.input_file,
            output_file=getattr(args, "output_file", None),
            resume=False,
        )
        session.reset_progress()
        _print_json(
            {
                "status": "reset",
                "session": session.get_status(),
            },
            pretty=args.pretty,
        )
        return 0
    except Exception as e:
        _print_json({"status": "error", "error": str(e)}, pretty=args.pretty)
        return 1


# ==============================================================================
# Argument Parsing Setup
# ==============================================================================


def add_agent_subparser(subparsers: argparse._SubParsersAction):
    """Add the agent subcommands to the main argument parser."""
    agent_parser = subparsers.add_parser(
        "agent",
        help="Commands for AI Agents (Antigravity, Codex, Cursor, etc.) to translate subtitles step-by-step",
    )
    agent_subparsers = agent_parser.add_subparsers(dest="agent_command", help="Agent action")

    # Common translation arguments helper
    def add_common_translation_args(parser):
        parser.add_argument("input_file", help="Input subtitle (.srt, .ass) or video file")
        parser.add_argument("-o", "--output-file", help="Custom output file path")
        parser.add_argument("-b", "--batch-size", type=int, default=100, help="Batch size (number of subtitle lines)")
        parser.add_argument("--pretty", action="store_true", help="Pretty print JSON output")

    def setup_translation_subparsers(subparser_container):
        # Start
        start_p = subparser_container.add_parser("start", help="Start translation session and get the first batch")
        add_common_translation_args(start_p)
        start_p.add_argument("-l", "--target-language", required=True, help="Target translation language")
        start_p.add_argument("-v", "--video-file", help="Video file for audio/subtitle extraction")
        start_p.add_argument("-a", "--audio-file", help="Audio file for context")
        start_p.add_argument("-d", "--description", help="Additional context/notes for translation")
        start_p.add_argument("--context-size", type=int, default=20, help="Number of previous lines for context")
        start_p.add_argument("--no-resume", dest="resume", action="store_false", default=True, help="Don't resume")

        # Next
        next_p = subparser_container.add_parser("next", help="Get the current pending batch")
        add_common_translation_args(next_p)
        next_p.add_argument("-l", "--target-language", help="Target translation language")

        # Commit
        commit_p = subparser_container.add_parser("commit", help="Commit a translated batch")
        add_common_translation_args(commit_p)
        commit_p.add_argument("-l", "--target-language", help="Target translation language")
        commit_p.add_argument("--data", help="Raw JSON string of translated batch items")
        commit_p.add_argument("--data-file", "--file", help="Path to JSON file containing translated batch items")

        # Status
        status_p = subparser_container.add_parser("status", help="Get translation status")
        status_p.add_argument("input_file", help="Input subtitle file")
        status_p.add_argument("-o", "--output-file", help="Custom output file path")
        status_p.add_argument("--pretty", action="store_true", help="Pretty print JSON output")

        # Reset
        reset_p = subparser_container.add_parser("reset", help="Reset translation progress")
        reset_p.add_argument("input_file", help="Input subtitle file")
        reset_p.add_argument("-o", "--output-file", help="Custom output file path")
        reset_p.add_argument("--pretty", action="store_true", help="Pretty print JSON output")

    # 1. Grouped Translate Subparsers: `gst agent translate <start|next|commit|status|reset>`
    translate_parser = agent_subparsers.add_parser("translate", help="Translate subtitle files step-by-step")
    translate_subparsers = translate_parser.add_subparsers(dest="translate_command", help="Translation action")
    setup_translation_subparsers(translate_subparsers)

    # 2. Direct Top-level aliases for translation: `gst agent <start|next|commit|status|reset>`
    setup_translation_subparsers(agent_subparsers)


def handle_agent_command(args) -> int:
    """Route agent subcommands."""
    if not getattr(args, "agent_command", None):
        print(
            "Please specify an agent subcommand: translate, start, next, commit, status, reset",
            file=sys.stderr,
        )
        return 1

    # Grouped translation: gst agent translate <action>
    if args.agent_command == "translate":
        subcmd = getattr(args, "translate_command", None)
        if not subcmd:
            print("Please specify a translate action: start, next, commit, status, reset", file=sys.stderr)
            return 1
        if subcmd == "start":
            return cmd_agent_start(args)
        elif subcmd == "next":
            return cmd_agent_next(args)
        elif subcmd == "commit":
            return cmd_agent_commit(args)
        elif subcmd == "status":
            return cmd_agent_status(args)
        elif subcmd == "reset":
            return cmd_agent_reset(args)

    # Direct top-level translation aliases
    elif args.agent_command == "start":
        return cmd_agent_start(args)
    elif args.agent_command == "next":
        return cmd_agent_next(args)
    elif args.agent_command == "commit":
        return cmd_agent_commit(args)
    elif args.agent_command == "status":
        return cmd_agent_status(args)
    elif args.agent_command == "reset":
        return cmd_agent_reset(args)

    return 1
