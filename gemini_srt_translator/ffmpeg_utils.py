import json
import os
import re
import subprocess
import sys

# Assuming a logger utility exists in the same directory as per the original file structure.
# If this script is run standalone, these can be replaced with print().
from .logger import error, info, success, warning

# --- Utility Functions ---


def _run_command(cmd, capture_output=True, text=True):
    """Runs a command-line utility, handling exceptions."""
    try:
        result = subprocess.run(cmd, capture_output=capture_output, text=text, check=True, encoding="utf-8")
        return result
    except FileNotFoundError:
        error(f"Command '{cmd[0]}' not found. Please ensure FFmpeg is installed and in your PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        error(f"Error executing command: {' '.join(cmd)}")
        if e.stderr:
            error(f"FFmpeg Error: {e.stderr.strip()}")
        raise  # Re-raise the exception to be caught by the main logic


def get_file_size_mb(file_path):
    """Gets the size of a file in megabytes."""
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except FileNotFoundError:
        return 0


def get_audio_properties(video_path):
    """Gets primary audio stream properties (duration, channels, index) using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        "-select_streams",
        "a",  # Select only audio streams
        video_path,
    ]
    result = _run_command(cmd)
    data = json.loads(result.stdout)

    if not data.get("streams"):
        raise ValueError("No audio stream found in the video file.")

    # Prioritize the "default" audio stream, otherwise use the first one
    primary_stream = next((s for s in data["streams"] if s.get("disposition", {}).get("default")), data["streams"][0])

    duration = float(data["format"].get("duration", 0))
    if duration <= 0:
        raise ValueError("Could not determine a valid audio duration.")

    properties = {
        "duration": duration,
        "channels": primary_stream.get("channels", 1),
        "stream_index": primary_stream["index"],
    }
    return properties


# --- Core Processing ---


def extract_audio_from_video(video_path, isolate_voice=False, target_mb=20):
    """
    Extracts, processes, and compresses audio from a video file into a final MP3.

    This function performs the following steps:
    1. Gets audio properties from the video.
    2. Builds an FFmpeg filter chain for voice isolation and channel mixing.
    3. Extracts the audio to a temporary WAV file using the filter.
    4. Calculates a target bitrate to meet the desired file size.
    5. Compresses the WAV file to a final MP3 file.
    6. Cleans up the temporary WAV file.
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.dirname(video_path)

    temp_wav_path = os.path.join(output_dir, f"{base_name}_temp.wav")
    output_path = os.path.join(output_dir, f"{base_name}_extracted.mp3")

    if os.path.exists(output_path):
        warning(f"Output file '{os.path.basename(output_path)}' already exists. Skipping.")
        return output_path

    try:
        info("Step 1: Analyzing audio properties...")
        props = get_audio_properties(video_path)

        # --- Build Filter Chain for Extraction ---
        filters = []
        # Downmix multi-channel audio to mono for consistency
        if props["channels"] > 1:
            # If isolating voice on a 5.1/7.1 stream, extract the center channel (usually contains dialogue).
            if props["channels"] >= 6 and isolate_voice:
                info(f"- Using center channel for voice isolation on {props['channels']}-channel audio.")
                filters.append("pan=mono|c0=FC")
            else:
                info(f"- Downmixing {props['channels']}-channel audio to mono.")
                # Use a weighted average of all channels for downmixing
                # Average all channels into mono: c0=(1/N)*c0+(1/N)*c1+...+(1/N)*cN
                n = props["channels"]
                pan_expr = " + ".join([f"{1/n:.3f}*c{i}" for i in range(n)])
                filters.append(f"pan=mono|c0={pan_expr}")
                if isolate_voice:
                    info("- Applying voice isolation filters...")
                    filters.extend(
                        [
                            "highpass=f=80",  # Remove low-frequency rumble
                            "lowpass=f=3400",  # Remove high-frequency noise
                        ]
                    )

        # --- Extract to Temporary WAV File ---
        info("Step 2: Extracting audio to a temporary WAV file...")
        extract_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-map",
            f"0:{props['stream_index']}",  # Select the correct audio stream
            "-vn",  # No video
            "-acodec",
            "pcm_s16le",  # Uncompressed WAV format
        ]
        if filters:
            extract_cmd.extend(["-af", ",".join(filters)])
        extract_cmd.append(temp_wav_path)
        _run_command(extract_cmd)

        # --- Compress WAV to MP3 ---
        info("Step 3: Compressing audio to MP3...")
        # Calculate bitrate in kbps to hit the target size. (Size in MB * 8192) / Duration in seconds.
        # Use a fudge factor (0.95) to account for metadata and overhead.
        target_bitrate_kbps = int((target_mb * 8192 * 0.95) / props["duration"])
        # Clamp bitrate to a reasonable range for MP3 (e.g., 16k to 192k)
        clamped_bitrate = max(16, min(target_bitrate_kbps, 192))

        compress_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            temp_wav_path,
            "-acodec",
            "libmp3lame",
            "-b:a",
            f"{clamped_bitrate}k",  # Set average audio bitrate
            output_path,
        ]
        _run_command(compress_cmd)

        final_size = get_file_size_mb(output_path)
        success("--- Success! ---")
        success(f"Audio saved to: {output_path}")
        success(f"Final file size: {final_size:.2f} MB")
        return output_path

    except (ValueError, subprocess.CalledProcessError) as e:
        error("--- An error occurred ---")
        error(f"{e}")
        return None

    finally:
        # --- Cleanup ---
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)


def extract_srt_from_video(video_path):
    """Extracts the first subtitle track from a video into an SRT file."""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.dirname(video_path)
    srt_path = os.path.join(output_dir, f"{base_name}_extracted.srt")

    if os.path.exists(srt_path):
        warning(f"Subtitle file '{os.path.basename(srt_path)}' already exists. Skipping.")
        return srt_path

    cmd = ["ffmpeg", "-y", "-i", video_path, "-map", "0:s:0", srt_path]
    try:
        info("Extracting subtitles...")
        _run_command(cmd)
        success("--- Success! ---")
        success(f"Subtitles saved to: {srt_path}")
        return srt_path
    except subprocess.CalledProcessError:
        warning("Could not find a subtitle stream in the video file.")
        return None


def check_ffmpeg_installation():
    """Checks if FFmpeg is installed and accessible."""
    try:
        _run_command(["ffmpeg", "-version"], capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_audio_length(audio_path: str) -> float:
    """
    Get the length of the audio file in seconds.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        audio_path,
    ]
    try:
        output = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding="utf-8")
        return float(output.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0
