import os
import re
import shutil
import sys
from datetime import timedelta
from enum import Enum
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass


# Global variable to control color output
_use_colors = True
_loading_bars = ["—", "\\", "|", "/"]
_loading_bars_index = -1
_thoughts_list = []
_quiet_mode = False
_line_number = "1"


class Color(Enum):
    """ANSI color codes"""

    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    @staticmethod
    def supports_color() -> bool:
        """Check if the terminal supports color output"""
        # If NO_COLOR env var is set, disable color
        if os.environ.get("NO_COLOR"):
            return False

        # If FORCE_COLOR env var is set, enable color
        if os.environ.get("FORCE_COLOR"):
            return True

        # Check if stdout is a TTY
        is_a_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

        return (
            is_a_tty
            or "ANSICON" in os.environ
            or "WT_SESSION" in os.environ
            or os.environ.get("TERM_PROGRAM") == "vscode"
        )


def set_color_mode(enabled: bool) -> None:
    """Set whether to use colors in output"""
    global _use_colors
    _use_colors = enabled


def set_quiet_mode(enabled: bool) -> None:
    """Set whether to suppress all output"""
    global _quiet_mode
    _quiet_mode = enabled


def set_line_number(line_number: str) -> None:
    """Set the line number for input prompts"""
    global _line_number
    _line_number = line_number


def info(message: Any, ignore_quiet: bool = False) -> None:
    """Print an information message in cyan color"""
    if _quiet_mode and not ignore_quiet:
        return
    if _use_colors and Color.supports_color():
        print(f"{Color.CYAN.value}{message}{Color.RESET.value}")
    else:
        print(message)


def warning(message: Any, ignore_quiet: bool = False) -> None:
    """Print a warning message in yellow color"""
    if _quiet_mode and not ignore_quiet:
        return
    if _use_colors and Color.supports_color():
        print(f"{Color.YELLOW.value}{message}{Color.RESET.value}")
    else:
        print(message)


def error(message: Any, ignore_quiet: bool = False) -> None:
    """Print an error message in red color"""
    if _quiet_mode and not ignore_quiet:
        return
    if _use_colors and Color.supports_color():
        print(f"{Color.RED.value}{message}{Color.RESET.value}")
    else:
        print(message)


def success(message: Any, ignore_quiet: bool = False) -> None:
    """Print a success message in green color"""
    if _quiet_mode and not ignore_quiet:
        return
    if _use_colors and Color.supports_color():
        print(f"{Color.GREEN.value}{message}{Color.RESET.value}")
    else:
        print(message)


def progress(message: Any, ignore_quiet: bool = False) -> None:
    """Print a progress/status update message in blue color"""
    if _quiet_mode and not ignore_quiet:
        return
    if _use_colors and Color.supports_color():
        print(f"{Color.BLUE.value}{message}{Color.RESET.value}")
    else:
        print(message)


def highlight(message: Any, ignore_quiet: bool = False) -> None:
    """Print an important message in magenta color"""
    if _quiet_mode and not ignore_quiet:
        return
    if _use_colors and Color.supports_color():
        print(f"{Color.MAGENTA.value}{Color.BOLD.value}{message}{Color.RESET.value}")
    else:
        print(message)


def input_prompt(message: Any, mode: str = None, max_length: int = 0) -> str:
    """Display a colored input prompt and return user input"""
    if _quiet_mode:
        if mode == "resume":
            return "y"
        if mode == "line":
            if int(_line_number) < 1 or int(_line_number) > max_length:
                error(
                    f"Line number must be between 1 and {max_length}, got {int(_line_number)}",
                    ignore_quiet=True,
                )
                exit(1)
            else:
                return _line_number
    if _use_colors and Color.supports_color():
        return input(f"{Color.WHITE.value}{Color.BOLD.value}{message}{Color.RESET.value}")
    else:
        return input(message)


# Store the last progress bar state for message updates
_last_progress = None
_has_started = False
_previous_messages = []
_token_stats = False
_last_chunk_size = 0
_prompt_token_count = 0
_thoughts_token_count = 0
_output_token_count = 0
_total_token_count = 0
_last_printed_lines = 0


def visible_len(text: str) -> int:
    """Calculate visible length of string ignoring ANSI color escape sequences."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return len(ansi_escape.sub('', str(text)))


def progress_bar(
    current: int,
    total: int,
    bar_length: int = 30,
    prefix: str = "",
    suffix: str = "",
    message: str = "",
    message_color: Color = None,
    isDone: bool = True,
    isPrompt: bool = False,
    isLoading: bool = False,
    isSending: bool = False,
    isThinking: bool = False,
    isTranscribing: bool = False,
    token_stats: bool | None = None,
    prompt_tokens: int | None = None,
    thoughts_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
    chunk_size: int = 0,
) -> None:
    """
    Display a colored progress bar with an optional message underneath

    Args:
        current: Current progress value
        total: Total value for 100% completion
        bar_length: Length of the progress bar in characters
        prefix: Text to display before the progress bar
        suffix: Text to display after the progress bar
        message: Optional message to display below the progress bar
        message_color: Color to use for the message
    """
    global _last_progress, _has_started, _previous_messages, _loading_bars_index, _last_chunk_size, _prompt_token_count, _thoughts_token_count, _output_token_count, _total_token_count, _token_stats, _last_printed_lines

    # Save the current state for message updates
    _last_progress = {
        "current": current,
        "total": total,
        "bar_length": bar_length,
        "prefix": prefix,
        "suffix": suffix,
    }

    if token_stats is not None:
        _token_stats = token_stats

    if chunk_size > _last_chunk_size or isDone:
        _last_chunk_size = chunk_size
    if prompt_tokens is not None:
        _prompt_token_count += prompt_tokens
    if thoughts_tokens is not None:
        _thoughts_token_count += thoughts_tokens
    if output_tokens is not None:
        _output_token_count += output_tokens
    if total_tokens is not None:
        _total_token_count += total_tokens

    # Get terminal width
    terminal_width = shutil.get_terminal_size().columns
    if terminal_width <= 0:
        terminal_width = 80

    # Create the progress bar
    progress_ratio = (current + _last_chunk_size) / total if total > 0 else 0
    filled_length = int(bar_length * progress_ratio)
    bar = "█" * filled_length + "░" * (bar_length - filled_length)
    percentage = int(100 * progress_ratio)
    if not isTranscribing:
        progress_text = f"{prefix} |{bar}| {percentage}% ({current + _last_chunk_size}/{total})"
    else:
        def format_time(s: float) -> str:
            td = timedelta(seconds=s)
            hours, remainder = divmod(td.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{hours:02}:{minutes:02}:{seconds:02},{td.microseconds // 1000:03}"
            
        current_text = format_time(current + _last_chunk_size)
        total_text = format_time(total)
        progress_text = f"{prefix} |{bar}| {percentage}% ({current_text}/{total_text})"
    # Format the progress bar line
    if suffix:
        progress_text = f"{progress_text} {suffix}"
    if isLoading:
        progress_text = f"{progress_text} | Processing {_loading_bars[_loading_bars_index]}"
    elif isThinking:
        progress_text = f"{progress_text} | Thinking {_loading_bars[_loading_bars_index]}"
    elif current < total and isSending:
        progress_text = f"{progress_text} | Sending batch ↑↑↑"

    # Handle the clearing of lines from the previous iteration
    if not _quiet_mode:
        if _has_started and _last_printed_lines > 0:
            sys.stdout.write("\r")
            for _ in range(_last_printed_lines):
                sys.stdout.write("\033[F")  # Move up one line
                sys.stdout.write("\033[K")  # Clear the line
        else:
            _has_started = True

    # Apply colors if enabled
    if _use_colors and Color.supports_color():
        colored_progress_text = progress_text.replace("█", f"{Color.GREEN.value}█{Color.BLUE.value}")
        colored_progress_text = colored_progress_text.replace("↑", f"{Color.GREEN.value}↑{Color.BLUE.value}")
        for i in range(len(_loading_bars)):
            colored_progress_text = colored_progress_text.replace(
                _loading_bars[i], f"{Color.GREEN.value}{_loading_bars[i]}{Color.BLUE.value}"
            )
        colored_progress_text = f"{Color.BLUE.value}{colored_progress_text}{Color.RESET.value}"
    else:
        colored_progress_text = progress_text

    if len(_previous_messages) > 0 and "waiting" in _previous_messages[-1]["message"].lower():
        _previous_messages.pop()

    if message:
        if not isPrompt:
            _previous_messages.append({"message": message, "color": message_color})

    # Prepare output lines and calculate physical lines for clearing in next iteration
    lines_plain = []
    lines_colored = []

    lines_plain.append(progress_text)
    lines_colored.append(colored_progress_text)

    if _token_stats:
        plain_stats = f"Prompt Tokens: {_prompt_token_count} | Thoughts Tokens: {_thoughts_token_count} | Output Tokens: {_output_token_count} | Total Tokens: {_total_token_count}"
        if _use_colors and Color.supports_color():
            colored_stats = f"Prompt Tokens: {Color.BLUE.value}{_prompt_token_count}{Color.RESET.value} | Thoughts Tokens: {Color.BLUE.value}{_thoughts_token_count}{Color.RESET.value} | Output Tokens: {Color.BLUE.value}{_output_token_count}{Color.RESET.value} | Total Tokens: {Color.BLUE.value}{_total_token_count}{Color.RESET.value}"
        else:
            colored_stats = plain_stats
        lines_plain.append(plain_stats)
        lines_colored.append(colored_stats)

    for msg in _previous_messages:
        msg_str = msg["message"]
        lines_plain.append(msg_str)
        if _use_colors and Color.supports_color():
            color_code = msg["color"].value if msg["color"] else Color.YELLOW.value
            lines_colored.append(f"{color_code}{msg_str}{Color.RESET.value}")
        else:
            lines_colored.append(msg_str)

    user_prompt = None
    if not _quiet_mode:
        # Print progress text and stats followed by empty line separator
        if _token_stats:
            sys.stdout.write(lines_colored[0] + "\n" + lines_colored[1] + "\n\n")
        else:
            sys.stdout.write(lines_colored[0] + "\n\n")

        # Print messages
        start_idx = 2 if _token_stats else 1
        for i in range(start_idx, len(lines_colored)):
            sys.stdout.write(lines_colored[i] + "\n")

        if message and isPrompt:
            if _use_colors and Color.supports_color():
                color_code = message_color.value if message_color else Color.YELLOW.value
                sys.stdout.write(f"{color_code}{Color.BOLD.value}{message}{Color.RESET.value}")
            else:
                sys.stdout.write(message)
            sys.stdout.flush()
            user_prompt = input()
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")
            sys.stdout.flush()

        sys.stdout.flush()

    # Calculate exact physical lines rendered on terminal
    physical_lines = 0
    for plain_line in lines_plain:
        v_len = visible_len(plain_line)
        physical_lines += max(1, (v_len + terminal_width - 1) // terminal_width)
    # Plus 1 for the empty line separator after progress header (\n\n)
    physical_lines += 1

    _last_printed_lines = physical_lines

    return user_prompt if isPrompt else None


def info_with_progress(
    message: Any, chunk_size: int = 0, isSending: bool = False, isTranscribing: bool = False
) -> None:
    """Update the progress bar with an info message"""
    if _quiet_mode:
        return
    progress_bar(
        **_last_progress,
        message=message,
        message_color=Color.CYAN,
        chunk_size=chunk_size,
        isSending=isSending,
        isTranscribing=isTranscribing,
    )


def warning_with_progress(
    message: Any,
    chunk_size: int = 0,
    isSending: bool = False,
    isTranscribing: bool = False,
    ignore_quiet: bool = False,
) -> None:
    """Update the progress bar with a warning message"""
    if _quiet_mode and not ignore_quiet:
        return
    progress_bar(
        **_last_progress,
        message=message,
        message_color=Color.YELLOW,
        chunk_size=chunk_size,
        isSending=isSending,
        isTranscribing=isTranscribing,
    )


def error_with_progress(
    message: Any,
    chunk_size: int = 0,
    isSending: bool = False,
    isTranscribing: bool = False,
    ignore_quiet: bool = False,
) -> None:
    """Update the progress bar with an error message"""
    if _quiet_mode and not ignore_quiet:
        return
    if _quiet_mode:
        error(message, ignore_quiet=True)
        return
    progress_bar(
        **_last_progress,
        message=message,
        message_color=Color.RED,
        chunk_size=chunk_size,
        isSending=isSending,
        isTranscribing=isTranscribing,
    )


def success_with_progress(
    message: Any, chunk_size: int = 0, isSending: bool = False, isTranscribing: bool = False
) -> None:
    """Update the progress bar with a success message"""
    if _quiet_mode:
        return
    progress_bar(
        **_last_progress,
        message=message,
        message_color=Color.GREEN,
        chunk_size=chunk_size,
        isSending=isSending,
        isTranscribing=isTranscribing,
    )


def highlight_with_progress(
    message: Any, chunk_size: int = 0, isSending: bool = False, isTranscribing: bool = False
) -> None:
    """Update the progress bar with a highlighted message"""
    if _quiet_mode:
        return
    progress_bar(
        **_last_progress,
        message=message,
        message_color=Color.MAGENTA,
        chunk_size=chunk_size,
        isSending=isSending,
        isTranscribing=isTranscribing,
    )


def input_prompt_with_progress(message: Any, batch_size: int, isTranscribing: bool = False) -> str:
    """Update the progress bar with an input prompt message"""
    if _quiet_mode:
        return f"{max(1, batch_size - 50)}"
    return progress_bar(
        **_last_progress,
        message=message,
        message_color=Color.WHITE,
        isPrompt=True,
        isTranscribing=isTranscribing,
    )


def update_loading_animation(
    chunk_size: int = 0,
    isThinking: bool = False,
    isTranscribing: bool = False,
    token_stats: bool = False,
    prompt_tokens: int | None = None,
    thoughts_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
) -> None:
    """Update the loading animation in the progress bar"""
    global _loading_bars_index
    if _quiet_mode:
        return
    _loading_bars_index = (_loading_bars_index + 1) % len(_loading_bars)
    progress_bar(
        **_last_progress,
        message="",
        message_color=None,
        isDone=False,
        isLoading=not isThinking,
        isThinking=isThinking,
        isTranscribing=isTranscribing,
        chunk_size=chunk_size,
        token_stats=token_stats,
        prompt_tokens=prompt_tokens,
        thoughts_tokens=thoughts_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def get_last_chunk_size() -> int:
    """Get the last chunk size used in the progress bar"""
    return _last_chunk_size


def save_logs_to_file(log_file_path: str = "progress.log") -> bool:
    """
    Save the current progress to a file.

    Args:
        log_file_path (str): Path to the log file. Defaults to 'progress.log'.

    Returns:
        bool: True if logs were saved successfully, False otherwise.
    """
    try:
        # Ensure the directory exists
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        with open(log_file_path, "w", encoding="utf-8") as f:
            if _last_progress:
                # Write progress information in the same format as shown in terminal
                current = _last_progress["current"] + _last_chunk_size
                total = _last_progress["total"]
                bar_length = _last_progress["bar_length"]
                prefix = _last_progress["prefix"]
                suffix = _last_progress["suffix"]

                # Create the progress bar
                progress_ratio = current / total if total > 0 else 0
                filled_length = int(bar_length * progress_ratio)
                bar = "█" * filled_length + "░" * (bar_length - filled_length)
                percentage = int(100 * progress_ratio)

                # Format progress text just like in terminal
                progress_text = f"{prefix} |{bar}| {percentage}% ({current}/{total})"
                if suffix:
                    progress_text = f"{progress_text} {suffix}"

                f.write(f"{progress_text}\n\n")

                # Write all the stored messages
                if _previous_messages:
                    for msg in _previous_messages:
                        f.write(f"{msg['message']}\n")
        return True
    except (PermissionError, OSError) as e:
        warning_with_progress(f"Failed to save logs to {log_file_path}: {e}")
        return False


def save_thoughts_to_file(thoughts: str, file_path: str = "thoughts.log", retry: int = 0) -> bool:
    """
    Save the current thoughts to a file.

    Args:
        thoughts (str): The thoughts to save.
        file_path (str): Path to the file. Defaults to 'thoughts.txt'.

    Returns:
        bool: True if thoughts were saved successfully, False otherwise.
    """
    global _thoughts_list

    _thoughts_list.append({"text": thoughts, "retry": retry})

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            for i in range(len(_thoughts_list)):
                f.write("=" * 80 + "\n\n")
                if _thoughts_list[i]["retry"] > 0:
                    f.write(f"Batch {batch_number}.{_thoughts_list[i]['retry']} thoughts (retry):\n\n")
                else:
                    batch_number = i + 1
                    f.write(f"Batch {batch_number} thoughts:\n\n")
                f.write("=" * 80 + "\n\n")
                f.write(_thoughts_list[i]["text"])
                f.write("\n\n")

        return True
    except (PermissionError, OSError) as e:
        warning_with_progress(f"Failed to save thoughts to {file_path}: {e}")
        return False
