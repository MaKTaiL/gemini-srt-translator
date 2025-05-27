import os
import subprocess
import sys
import json


def get_file_size_mb(file_path):
    """Get file size in MB"""
    if os.path.exists(file_path):
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    return 0


def check_filter_availability(filter_name):
    """Check if a specific filter is available in the current FFmpeg build"""
    try:
        cmd = ['ffmpeg', '-hide_banner', '-filters']
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return filter_name in result.stdout
    except subprocess.CalledProcessError:
        return False


def compress_audio(extracted_audio_path, target_size_mb=20):
    """Compress audio file to target size using MP3"""
    current_size = get_file_size_mb(extracted_audio_path)

    if current_size <= target_size_mb:
        print(f"Audio file size is {current_size:.2f}MB, no compression needed")
        target_size_mb = target_size_mb - target_size_mb * 0.1

    print(f"Audio file size is {current_size:.2f}MB, compressing to MP3 (~{target_size_mb}MB)")

    # Create compressed filename (change to .mp3)
    base_name = os.path.splitext(extracted_audio_path)[0]
    compressed_path = f"{extracted_audio_path}_compressed.mp3"

    # Calculate target bitrate based on file duration and target size
    try:
        # Verify the input file exists
        if not os.path.exists(extracted_audio_path):
            raise FileNotFoundError(f"Input file not found: {extracted_audio_path}")

        print(f"Analyzing file: {extracted_audio_path}")

        # Get audio duration - use extracted_audio_path instead of input_path
        cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', extracted_audio_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Check if we got valid output
        if not result.stdout.strip():
            raise ValueError("ffprobe returned empty output")

        data = json.loads(result.stdout)

        # Check if format data exists
        if 'format' not in data or 'duration' not in data['format']:
            raise ValueError("Could not get duration information from audio file")

        duration = float(data['format']['duration'])

        if duration <= 0:
            raise ValueError(f"Invalid duration: {duration}")

        # Calculate target bitrate for MP3
        target_bitrate_kbps = int((target_size_mb * 8 * 1024) / duration * 0.9)

        # Set reasonable bitrate ranges for MP3
        if target_bitrate_kbps > 128:
            target_bitrate_kbps = 128
        elif target_bitrate_kbps > 96:
            target_bitrate_kbps = 96
        elif target_bitrate_kbps > 64:
            target_bitrate_kbps = 64
        elif target_bitrate_kbps > 48:
            target_bitrate_kbps = 48
        else:
            target_bitrate_kbps = 32

        print(f"Duration: {duration:.2f}s, Target MP3 bitrate: {target_bitrate_kbps}kbps")

        # Compress to MP3 using libmp3lame
        cmd = [
            'ffmpeg', '-i', extracted_audio_path, '-y',
            '-acodec', 'libmp3lame',
            '-b:a', f'{target_bitrate_kbps}k',
            '-ac', '1',  # Mono
            '-ar', '22050',  # Sample rate
            compressed_path
        ]

        print(f"Compressing to MP3: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Verify the compressed file was created
        if not os.path.exists(compressed_path):
            raise FileNotFoundError("Compressed file was not created")

        # Check if compression was successful
        new_size = get_file_size_mb(compressed_path)
        print(f"Compressed MP3 size: {new_size:.2f}MB")

        # If still too large, try lower bitrate
        if new_size > target_size_mb * 1.2:
            print("Still too large, trying lower bitrate...")

            lower_bitrate = max(16, target_bitrate_kbps // 2)
            aggressive_path = f"{base_name}_compressed_low.mp3"

            cmd = [
                'ffmpeg', '-i', extracted_audio_path, '-y',  # Use extracted_audio_path here too
                '-acodec', 'libmp3lame',
                '-b:a', f'{lower_bitrate}k',
                '-ac', '1',
                '-ar', '16000',
                aggressive_path
            ]

            subprocess.run(cmd, capture_output=True, text=True, check=True)

            if os.path.exists(compressed_path):
                os.remove(compressed_path)

            os.rename(aggressive_path, compressed_path)
            new_size = get_file_size_mb(compressed_path)
            print(f"Final compressed MP3 size: {new_size:.2f}MB")

        # Only remove the original file if it's different from the compressed output
        if extracted_audio_path != compressed_path and os.path.exists(extracted_audio_path):
            os.remove(extracted_audio_path)

        # Update the path to reflect the new MP3 file
        final_path = base_name + ".mp3"

        # Only rename if the paths are different
        if compressed_path != final_path:
            if os.path.exists(final_path):
                os.remove(final_path)
            os.rename(compressed_path, final_path)

        return final_path

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg/FFprobe command failed: {e}")
        print(f"Command: {' '.join(e.cmd)}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        if os.path.exists(compressed_path):
            os.remove(compressed_path)
        return extracted_audio_path

    except json.JSONDecodeError as e:
        print(f"Failed to parse ffprobe output: {e}")
        if os.path.exists(compressed_path):
            os.remove(compressed_path)
        return extracted_audio_path

    except Exception as e:
        print(f"Error during MP3 compression: {e}")
        if os.path.exists(compressed_path):
            os.remove(compressed_path)
        return extracted_audio_path


def get_audio_info(video_path):
    """Get audio channel information from video file, selecting the primary audio stream"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', '-select_streams', 'a', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        data = json.loads(result.stdout)

        if not data['streams']:
            raise ValueError("No audio stream found in video")

        # Find the primary audio stream
        primary_stream = None
        primary_index = 0

        # Look for streams with disposition "default" first
        for i, stream in enumerate(data['streams']):
            disposition = stream.get('disposition', {})
            if disposition.get('default', 0) == 1:
                primary_stream = stream
                primary_index = i
                break

        # If no default stream found, use the first audio stream
        if primary_stream is None:
            primary_stream = data['streams'][0]
            primary_index = 0

        channels = primary_stream.get('channels', 0)
        channel_layout = primary_stream.get('channel_layout', '')
        codec_name = primary_stream.get('codec_name', 'unknown')

        print(f"Selected audio stream {primary_index}: {codec_name}, {channels} channels, layout: {channel_layout}")

        return channels, channel_layout, primary_index

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to analyze audio: {e}")
    except json.JSONDecodeError:
        raise RuntimeError("Failed to parse audio information")


def create_voice_isolation_filter(channels):
    """Create audio filter chain for voice isolation with compatible filters"""
    filters = []

    # 1. High-pass filter to remove low-frequency rumble and bass
    filters.append("highpass=f=80")

    # 2. Low-pass filter to remove very high frequencies (keeps speech range)
    filters.append("lowpass=f=3400")

    # 3. Band-pass filter focused on human speech frequencies (300Hz - 3400Hz)
    filters.append("bandpass=f=1850:width_type=h:w=3100")

    # 4. Dynamic range compression to even out volume levels
    filters.append("compand=attacks=0.3:decays=0.8:points=-80/-80|-45/-15|-27/-9|0/-7|20/-7")

    # 5. Alternative to gate filter - use volume filter with very low threshold
    # This acts as a simple noise gate by reducing very quiet sounds
    filters.append("volume=volume=1:eval=frame:precision=float")

    # 6. Check if deesser is available, if not skip it
    if check_filter_availability('deesser'):
        filters.append("deesser")
    else:
        print("Warning: deesser filter not available, skipping")

    return ",".join(filters)


def create_basic_voice_filter():
    """Create a basic voice isolation filter using only common filters"""
    filters = []

    # Basic frequency filtering for voice range
    filters.append("highpass=f=300")  # Remove low frequencies
    filters.append("lowpass=f=3000")  # Remove high frequencies

    # Basic dynamic range compression
    filters.append("compand=attacks=0.1:decays=0.8:points=-80/-80|-20/-10|0/-3")

    # Normalize volume
    filters.append("volume=1.5")

    return ",".join(filters)


def extract_audio(video_path, output_path, channels, channel_layout, stream_index=0, isolate_voice=False):
    """Extract and process audio based on channel configuration with optional voice isolation"""

    # Base ffmpeg command with specific audio stream selection
    cmd = ['ffmpeg', '-i', video_path, '-map', f'0:a:{stream_index}', '-y']

    # Build audio filter chain
    audio_filters = []

    # Channel processing based on configuration
    if channels == 1:
        print("Audio is already mono")

    elif channels == 2:
        print("Converting stereo to mono")
        audio_filters.append("pan=1c|c0=0.5*c0+0.5*c1")

    elif channels > 2:
        print(f"Surround sound detected ({channels} channels)")

        if channels >= 6:  # 5.1 or 7.1
            print("Extracting center channel from 5.1/7.1 surround")
            # For 5.1: FL FR FC LFE BL BR (center is index 2)
            audio_filters.append("pan=1c|c0=1*c2")  # Extract center channel only
        elif channels == 5:  # 5.0 surround
            print("Extracting center channel from 5.0 surround")
            audio_filters.append("pan=1c|c0=1*c2")  # Extract center channel
        elif channels == 4:  # Quad
            print("Quad audio detected (no center channel), downmixing to mono")
            audio_filters.append("pan=1c|c0=0.25*c0+0.25*c1+0.25*c2+0.25*c3")
        elif channels == 3:  # 2.1 or 3.0
            print("3-channel audio, attempting to extract center (channel 2)")
            audio_filters.append("pan=1c|c0=1*c2")
        else:
            print(f"Unknown surround configuration ({channels} channels), downmixing to mono")
            audio_filters.append("pan=1c|c0=c0")

    # Add voice isolation filters
    if isolate_voice:
        print("Applying voice isolation filters...")
        try:
            # Try advanced filters first
            voice_filter = create_voice_isolation_filter(channels)
            audio_filters.append(voice_filter)
        except Exception as e:
            print(f"Advanced filters failed: {e}")
            print("Falling back to basic voice filtering...")
            # Fall back to basic filtering
            basic_filter = create_basic_voice_filter()
            audio_filters.append(basic_filter)

    # Combine all filters
    if audio_filters:
        filter_chain = ",".join(audio_filters)
        cmd.extend(['-vn', '-af', filter_chain, '-acodec', 'pcm_s16le'])
    else:
        cmd.extend(['-vn', '-acodec', 'pcm_s16le'])

    cmd.append(output_path)

    try:
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            print(f"FFmpeg stderr: {result.stderr}")
            # If the advanced filter fails, try with basic extraction only
            if isolate_voice and "No such filter" in result.stderr:
                print("Advanced filters not supported, trying basic extraction...")
                return extract_audio_basic(video_path, output_path, channels, stream_index)
            else:
                raise RuntimeError(f"FFmpeg failed with return code {result.returncode}")

        print(f"Audio extracted successfully to: {output_path}")
        return True

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed: {e}")


def extract_audio_basic(video_path, output_path, channels, stream_index=0):
    """Basic audio extraction without advanced filters"""
    print("Using basic audio extraction...")

    cmd = ['ffmpeg', '-i', video_path, '-map', f'0:a:{stream_index}', '-y']

    # Basic channel processing
    if channels == 1:
        cmd.extend(['-vn', '-acodec', 'pcm_s16le'])
    elif channels == 2:
        cmd.extend(['-vn', '-af', 'pan=1c|c0=0.5*c0+0.5*c1', '-acodec', 'pcm_s16le'])
    elif channels >= 6:  # 5.1 or 7.1 - extract center channel
        cmd.extend(['-vn', '-af', 'pan=1c|c0=1*c2', '-acodec', 'pcm_s16le'])
    else:
        # For other configurations, just convert to mono
        cmd.extend(['-vn', '-ac', '1', '-acodec', 'pcm_s16le'])

    cmd.append(output_path)

    print(f"Running basic extraction: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Basic audio extracted successfully to: {output_path}")
    return True


def process_video(video_path, output_path, isolate_voice):
    """Main function to process video file"""

    # Validate input file
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Processing: {video_path}")
    print(f"Output will be saved as: {output_path}")

    # Get audio information
    try:
        channels, channel_layout, stream_index = get_audio_info(video_path)
        print(f"Audio info - Channels: {channels}, Layout: {channel_layout}")

        # Extract and process audio
        extracted_audio_path = "extracted_audio.wav"
        success = extract_audio(video_path, extracted_audio_path, channels, channel_layout, stream_index, isolate_voice)

        if not success:
            return None

        # Check file size and compress if necessary
        file_size = get_file_size_mb(extracted_audio_path)
        print(f"Extracted audio size: {file_size:.2f}MB")

        if file_size > 2.0:
            output_path = compress_audio(extracted_audio_path)

        return output_path

    except Exception as e:
        print(f"Error processing video: {e}")
        return None


def prepare_audio(video_path, output_path, isolate_voice=False):
    # If output path is provided, use it; otherwise generate one
    if not output_path:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.dirname(video_path)
        suffix = "_voice" if isolate_voice else "_audio"
        original_ext = os.path.splitext(video_path)[1]
        output_path = os.path.join(output_dir, f"{base_name}{suffix}{original_ext}")

    # Check if ffmpeg and ffprobe are available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: FFmpeg and FFprobe are required but not found in PATH")
        print("Please install FFmpeg with MP3 support: https://ffmpeg.org/download.html")
        sys.exit(1)

    # Check filter availability
    print("Checking filter availability...")
    available_filters = []
    test_filters = ['gate', 'deesser', 'compand', 'highpass', 'lowpass', 'bandpass']
    for filter_name in test_filters:
        if check_filter_availability(filter_name):
            available_filters.append(filter_name)

    print(f"Available filters: {', '.join(available_filters) if available_filters else 'Using basic filters only'}")

    # Process the video
    result = process_video(video_path, output_path, isolate_voice)

    if result:
        final_size = get_file_size_mb(result)
        print(f"\nSuccess! Audio saved as: {result}")
        print(f"Final file size: {final_size:.2f}MB")
        if isolate_voice:
            print("Voice isolation filters applied!")
        return result
    else:
        print("\nFailed to process video")
        sys.exit(1)
