import sys
import os
import shutil
from enum import Enum
from typing import Any, Optional

# Global variable to control color output
_use_colors = True

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
        # If the NO_COLOR env var is set, then we shouldn't use color
        if os.environ.get('NO_COLOR', '') != '':
            return False
        
        # If the FORCE_COLOR env var is set, then we should use color
        if os.environ.get('FORCE_COLOR', '') != '':
            return True
        
        # isatty is not always implemented
        is_a_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        
        # Windows has specific checks for color support
        if sys.platform == 'win32':
            return is_a_tty and ('ANSICON' in os.environ or 
                               'WT_SESSION' in os.environ or 
                               os.environ.get('TERM_PROGRAM') == 'vscode')
        
        # For all other platforms, assume color support if it's a TTY
        return is_a_tty

def set_color_mode(enabled: bool) -> None:
    """Set whether to use colors in output"""
    global _use_colors
    _use_colors = enabled

def info(message: Any) -> None:
    """Print an information message in cyan color"""
    if _use_colors and Color.supports_color():
        print(f"{Color.CYAN.value}{message}{Color.RESET.value}")
    else:
        print(message)

def warning(message: Any) -> None:
    """Print a warning message in yellow color"""
    if _use_colors and Color.supports_color():
        print(f"{Color.YELLOW.value}{message}{Color.RESET.value}")
    else:
        print(message)

def error(message: Any) -> None:
    """Print an error message in red color"""
    if _use_colors and Color.supports_color():
        print(f"{Color.RED.value}{message}{Color.RESET.value}")
    else:
        print(message)

def success(message: Any) -> None:
    """Print a success message in green color"""
    if _use_colors and Color.supports_color():
        print(f"{Color.GREEN.value}{message}{Color.RESET.value}")
    else:
        print(message)

def progress(message: Any) -> None:
    """Print a progress/status update message in blue color"""
    if _use_colors and Color.supports_color():
        print(f"{Color.BLUE.value}{message}{Color.RESET.value}")
    else:
        print(message)

def highlight(message: Any) -> None:
    """Print an important message in magenta color"""
    if _use_colors and Color.supports_color():
        print(f"{Color.MAGENTA.value}{Color.BOLD.value}{message}{Color.RESET.value}")
    else:
        print(message)

def input_prompt(message: Any) -> str:
    """Display a colored input prompt and return user input"""
    if _use_colors and Color.supports_color():
        return input(f"{Color.WHITE.value}{Color.BOLD.value}{message}{Color.RESET.value}")
    else:
        return input(message)

# Store the last progress bar state for message updates
_last_progress = {"current": 0, "total": 0, "prefix": "", "suffix": "", "bar_length": 30}
_has_started = False
_previous_messages = []

def progress_bar(current: int, total: int, bar_length: int = 30, prefix: str = "", suffix: str = "", message: str = "", message_color: Color = None) -> None:
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
    global _last_progress, _has_started, _previous_messages
    
    # Save the current state for message updates
    _last_progress = {
        "current": current,
        "total": total,
        "bar_length": bar_length,
        "prefix": prefix,
        "suffix": suffix
    }
    
    # Get terminal width
    terminal_width = shutil.get_terminal_size().columns
    
    # Create the progress bar
    progress_ratio = current / total if total > 0 else 0
    filled_length = int(bar_length * progress_ratio)
    bar = "█" * filled_length + "░" * (bar_length - filled_length)
    percentage = int(100 * progress_ratio)
    
    # Format the progress bar line
    if suffix:
        progress_text = f"{prefix} |{bar}| {percentage}% ({current}/{total}) {suffix}"
    else:
        progress_text = f"{prefix} |{bar}| {percentage}% ({current}/{total})"
    
    # Handle the clearing of lines based on whether we've shown a message

    if _has_started:
        sys.stdout.write(" " * terminal_width)
        for i in range(len(_previous_messages)):
            sys.stdout.write("\033[F" + " " * terminal_width + "\r")
        sys.stdout.write("\033[F" + " " * terminal_width)
        sys.stdout.write("\033[F" + " " * terminal_width + "\r")
    else:
        _has_started = True
    
    # Apply colors if enabled
    if _use_colors and Color.supports_color():
        colored_bar = bar.replace("█", f"{Color.GREEN.value}█{Color.BLUE.value}")
        progress_text = f"{Color.BLUE.value}{prefix} |{colored_bar}| {percentage}% ({current}/{total}) {suffix}{Color.RESET.value}"
    
    sys.stdout.write(progress_text)
    sys.stdout.write("\n\n")

    for i in range(len(_previous_messages)):
        if _use_colors and Color.supports_color():
            color_code = _previous_messages[i]["color"].value if _previous_messages[i]["color"] else Color.YELLOW.value
            sys.stdout.write(f"{color_code}{_previous_messages[i]['message']}{Color.RESET.value}\n")
        else:
            sys.stdout.write(_previous_messages[i]["message"] + "\n")
    if message:
        _previous_messages.append({
            "message": message,
            "color": message_color
        })
        if _use_colors and Color.supports_color():
            color_code = message_color.value if message_color else Color.YELLOW.value
            sys.stdout.write(f"{color_code}{message}{Color.RESET.value}")
        else:
            sys.stdout.write(message)
    sys.stdout.flush()

def info_with_progress(message: Any) -> None:
    """Update the progress bar with an info message"""
    progress_bar(**_last_progress, message=message, message_color=Color.CYAN)

def warning_with_progress(message: Any) -> None:
    """Update the progress bar with a warning message"""
    progress_bar(**_last_progress, message=message, message_color=Color.YELLOW)

def error_with_progress(message: Any) -> None:
    """Update the progress bar with an error message"""
    progress_bar(**_last_progress, message=message, message_color=Color.RED)

def success_with_progress(message: Any) -> None:
    """Update the progress bar with a success message"""
    progress_bar(**_last_progress, message=message, message_color=Color.GREEN)

def highlight_with_progress(message: Any) -> None:
    """Update the progress bar with a highlighted message"""
    progress_bar(**_last_progress, message=message, message_color=Color.MAGENTA)
