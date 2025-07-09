import os
import shutil
import sys
from enum import Enum
from typing import Any

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
                error(f"Line number must be between 1 and {max_length}, got {int(_line_number)}", ignore_quiet=True)
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
_last_chunk_size = 0


def progress_bar(
    current: int,
    total: int,
    bar_length: int = 30,
    prefix: str = "",
    suffix: str = "",
    message: str = "",
    message_color: Color = None,
    isPrompt: bool = False,
    isLoading: bool = False,
    isSending: bool = False,
    isThinking: bool = False,
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
    global _last_progress, _has_started, _previous_messages, _loading_bars_index, _last_chunk_size

    # Save the current state for message updates
    _last_progress = {
        "current": current,
        "total": total,
        "bar_length": bar_length,
        "prefix": prefix,
        "suffix": suffix,
    }

    _last_chunk_size = chunk_size

    # Get terminal width
    terminal_width = shutil.get_terminal_size().columns

    # Create the progress bar
    progress_ratio = (current + chunk_size) / total if total > 0 else 0
    filled_length = int(bar_length * progress_ratio)
    bar = "█" * filled_length + "░" * (bar_length - filled_length)
    percentage = int(100 * progress_ratio)
    progress_text = f"{prefix} |{bar}| {percentage}% ({current + chunk_size}/{total})"
    # Format the progress bar line
    if suffix:
        progress_text = f"{progress_text} {suffix}"
    if isLoading:
        progress_text = f"{progress_text} | Processing {_loading_bars[_loading_bars_index]}"
    elif isThinking:
        progress_text = f"{progress_text} | Thinking {_loading_bars[_loading_bars_index]}"
    elif current < total and isSending:
        progress_text = f"{progress_text} | Sending batch ↑↑↑"

    # Calculate how many lines we need to clear based on previous messages and terminal width
    # Start with at least 2 lines (progress bar + empty line)
    lines_to_clear = 2

    # Get the command used to start the script
    command_line = " ".join([sys.executable] + sys.argv)
    # Check if the command line is too long and needs to be wrapped
    if len(command_line) > terminal_width:
        # Add additional lines needed for wrapped command line
        lines_to_clear += len(command_line) // terminal_width - 1

    # Calculate how many lines the progress bar itself might take due to wrapping
    progress_text_length = len(progress_text)
    if progress_text_length > terminal_width:
        # Add additional lines needed for wrapped progress bar text
        lines_to_clear += progress_text_length // terminal_width

    # Add lines for each previous message, accounting for wrapping
    for msg in _previous_messages:
        msg_text = msg["message"]
        # Calculate how many lines this message would take (accounting for wrapping)
        msg_lines = (len(msg_text) // terminal_width) + 1
        lines_to_clear += msg_lines

    # Handle the clearing of lines
    if not _quiet_mode:
        if _has_started:
            # Move cursor to beginning of line
            sys.stdout.write("\r")

            # Clear each line individually by moving up and clearing
            for _ in range(lines_to_clear):
                sys.stdout.write("\033[F")  # Move up one line
                sys.stdout.write("\033[K")  # Clear the line
        else:
            _has_started = True

    # Apply colors if enabled
    if _use_colors and Color.supports_color():
        progress_text = progress_text.replace("█", f"{Color.GREEN.value}█{Color.BLUE.value}")
        progress_text = progress_text.replace("↑", f"{Color.GREEN.value}↑{Color.BLUE.value}")
        for i in range(len(_loading_bars)):
            progress_text = progress_text.replace(
                _loading_bars[i], f"{Color.GREEN.value}{_loading_bars[i]}{Color.BLUE.value}"
            )
        progress_text = f"{Color.BLUE.value}{progress_text}{Color.RESET.value}"

    if not _quiet_mode:
        sys.stdout.write(progress_text)
        sys.stdout.write("\n\n")

    if len(_previous_messages) > 0 and "waiting" in _previous_messages[-1]["message"].lower():
        _previous_messages.pop()

    if _quiet_mode:
        return

    for i in range(len(_previous_messages)):
        if _use_colors and Color.supports_color():
            color_code = _previous_messages[i]["color"].value if _previous_messages[i]["color"] else Color.YELLOW.value
            sys.stdout.write(f"{color_code}{_previous_messages[i]['message']}{Color.RESET.value}\n")
        else:
            sys.stdout.write(_previous_messages[i]["message"] + "\n")

    if message:
        if not isPrompt:
            _previous_messages.append({"message": message, "color": message_color})
        if _use_colors and Color.supports_color():
            color_code = message_color.value if message_color else Color.YELLOW.value
            if isPrompt:
                sys.stdout.write(f"{color_code}{Color.BOLD.value}{message}{Color.RESET.value}")
                user_prompt = input()
                sys.stdout.write("\033[F")
                sys.stdout.write("\033[K")
            else:
                sys.stdout.write(f"{color_code}{message}{Color.RESET.value}" + "\n")
        else:
            if isPrompt:
                sys.stdout.write(message)
                user_prompt = input()
                sys.stdout.write("\033[F")
                sys.stdout.write("\033[K")
            else:
                sys.stdout.write(message + "\n")
    sys.stdout.flush()
    return user_prompt if isPrompt else None


def info_with_progress(message: Any, chunk_size: int = 0, isSending: bool = False) -> None:
    """Update the progress bar with an info message"""
    if _quiet_mode:
        return
    progress_bar(
        **_last_progress,
        message=message,
        message_color=Color.CYAN,
        chunk_size=chunk_size,
        isSending=isSending,
    )


def warning_with_progress(message: Any, chunk_size: int = 0, isSending: bool = False) -> None:
    """Update the progress bar with a warning message"""
    if _quiet_mode:
        return
    progress_bar(
        **_last_progress,
        message=message,
        message_color=Color.YELLOW,
        chunk_size=chunk_size,
        isSending=isSending,
    )


def error_with_progress(message: Any, chunk_size: int = 0, isSending: bool = False) -> None:
    """Update the progress bar with an error message"""
    if _quiet_mode:
        return
    progress_bar(
        **_last_progress,
        message=message,
        message_color=Color.RED,
        chunk_size=chunk_size,
        isSending=isSending,
    )


def success_with_progress(message: Any, chunk_size: int = 0, isSending: bool = False) -> None:
    """Update the progress bar with a success message"""
    if _quiet_mode:
        return
    progress_bar(
        **_last_progress,
        message=message,
        message_color=Color.GREEN,
        chunk_size=chunk_size,
        isSending=isSending,
    )


def highlight_with_progress(message: Any, chunk_size: int = 0, isSending: bool = False) -> None:
    """Update the progress bar with a highlighted message"""
    if _quiet_mode:
        return
    progress_bar(
        **_last_progress,
        message=message,
        message_color=Color.MAGENTA,
        chunk_size=chunk_size,
        isSending=isSending,
    )


def input_prompt_with_progress(message: Any, batch_size: int) -> str:
    """Update the progress bar with an input prompt message"""
    if _quiet_mode:
        return f"{max(1, batch_size - 50)}"
    return progress_bar(**_last_progress, message=message, message_color=Color.WHITE, isPrompt=True)


def update_loading_animation(chunk_size: int = 0, isThinking: bool = False) -> None:
    """Update the loading animation in the progress bar"""
    global _loading_bars_index
    if _quiet_mode:
        return
    _loading_bars_index = (_loading_bars_index + 1) % len(_loading_bars)
    progress_bar(
        **_last_progress,
        message="",
        message_color=None,
        isLoading=not isThinking,
        isThinking=isThinking,
        chunk_size=chunk_size,
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
