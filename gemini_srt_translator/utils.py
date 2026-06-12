import importlib.metadata
import subprocess
import sys
import threading
import time
from datetime import timedelta

import requests
from packaging import version

from gemini_srt_translator.logger import error

from .logger import highlight, info, input_prompt, set_color_mode, success


def get_installed_version(package_name):
    """Returns the installed version of a package, or None if not installed."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_latest_pypi_version(package_name):
    """Fetches the latest version of a package from PyPI, ignoring pre-release versions."""
    url = f"https://pypi.org/pypi/{package_name}/json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()

        # Filter out pre-release versions (alpha, beta, rc, dev, etc.)
        releases = data["releases"]
        stable_versions = [
            version
            for version in releases.keys()
            if not any(pre in version for pre in ["a", "b", "rc", "dev", ".post", ".pre"])
        ]

        if stable_versions:
            # Sort versions and get the latest one
            latest_version = max(stable_versions, key=version.parse)
            return latest_version

        # If no stable versions, return the current version
        return data["info"]["version"]
    return None


def display_progress_bar(stop_event, error_event, package_name):
    """Display a custom progress bar."""
    width = 40
    position = 0
    direction = 1  # 1 for right, -1 for left
    bar_length = 10  # Length of the moving part
    has_started = False

    while not stop_event.is_set():
        # Update position
        position += direction

        # Change direction if hitting the boundaries
        if position >= width - bar_length or position <= 0:
            direction *= -1

        # Create the progress bar
        bar = "[" + " " * position + "#" * bar_length + " " * (width - position - bar_length) + "]"
        if has_started:
            sys.stdout.write("\033[A")
            sys.stdout.write(f"\rInstalling {package_name}: {bar}\n")
        else:
            sys.stdout.write(f"\rInstalling {package_name}: {bar}\n")
            has_started = True
        sys.stdout.flush()
        time.sleep(0.1)

    # Show complete when done
    if not error_event.is_set():
        bar = "[" + "#" * width + "]"
        sys.stdout.write("\033[A")
        sys.stdout.write(f"\rInstalling {package_name}: {bar}\n")
        sys.stdout.flush()


def upgrade_package(package_name, use_colors=True):
    """Upgrades the package using pip if an update is available."""
    installed_version = get_installed_version(package_name)
    latest_version = get_latest_pypi_version(package_name)

    set_color_mode(use_colors)
    stop_event = threading.Event()
    error_event = threading.Event()
    progress_thread = threading.Thread(target=display_progress_bar, args=(stop_event, error_event, package_name))

    if installed_version < latest_version:
        info(f"There is a new version of {package_name} available: {latest_version}.")
        answer = input_prompt(
            f"Do you want to upgrade {package_name} from version {installed_version} to {latest_version}? (y/n): "
        )
        if answer.lower() == "y":

            try:
                # Try to use 'uv' if available, otherwise fallback to pip
                try:
                    subprocess.check_call(
                        ["uv", "--version"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    highlight(f"Upgrading {package_name} using uv...\n")
                    use_uv = True
                except Exception:
                    highlight(f"Upgrading {package_name} using pip...\n")
                    use_uv = False

                progress_thread.start()

                if use_uv:
                    subprocess.check_call(
                        [
                            "uv",
                            "pip",
                            "install",
                            "--upgrade",
                            package_name,
                            "--quiet",
                            "--disable-pip-version-check",
                        ],
                    )
                else:
                    subprocess.check_call(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            "--upgrade",
                            package_name,
                            "--quiet",
                            "--disable-pip-version-check",
                        ],
                    )
                stop_event.set()
                progress_thread.join()
                success(f"{package_name} upgraded to version {latest_version}.")
                info("Please restart your script.")
                raise Exception("Upgrade completed.")
            except subprocess.CalledProcessError as e:
                error_event.set()
                stop_event.set()
                progress_thread.join()
                error(f"Error: Failed to upgrade the package.")
                raise Exception("Upgrade failed.")

        else:
            info(f"{package_name} upgrade skipped.\n")


def convert_timedelta_to_timestamp(td, offset=0):
    """Converts a timedelta object to a string in the format MM:SS."""
    if not isinstance(td, timedelta):
        raise TypeError("Expected a timedelta object.")

    total_seconds = td.seconds - offset
    minutes, seconds = divmod(total_seconds, 60)

    return f"{minutes:02}:{seconds:02}"


def convert_timestamp_to_timedelta(timestamp, offset=0):
    """Converts a timestamp string in the format MM:SS to a timedelta object."""
    if not isinstance(timestamp, str):
        raise TypeError("Expected a string in the format MM:SS.")

    parts = timestamp.split(":")
    if len(parts) != 2:
        raise ValueError("Timestamp must be in the format MM:SS.")

    try:
        minutes = int(parts[0])
        seconds = int(parts[1])
    except ValueError:
        raise ValueError("Minutes and seconds must be integers.")

    return timedelta(minutes=minutes, seconds=seconds + offset)
