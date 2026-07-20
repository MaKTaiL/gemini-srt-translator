> [!WARNING]
> If you are experiencing too many model overloaded messages, switch off streaming mode.

# 🌟 Gemini SRT Translator

[![PyPI version](https://img.shields.io/pypi/v/gemini-srt-translator)](https://pypi.org/project/gemini-srt-translator)
[![Python support](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FMaKTaiL%2Fgemini-srt-translator%2Frefs%2Fheads%2Fmain%2Fpyproject.toml&color=red)](https://pypi.org/project/gemini-srt-translator)
[![Downloads](https://img.shields.io/pypi/dw/gemini-srt-translator)](https://pypi.org/project/gemini-srt-translator)
[![GitHub contributors](https://img.shields.io/github/contributors/MaKTaiL/gemini-srt-translator)](https://github.com/MaKTaiL/gemini-srt-translator/graphs/contributors)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-orange?logo=buy-me-a-coffee)](https://www.buymeacoffee.com/maktail)

> Translate SRT and ASS subtitle files using the power of Google Gemini AI! 🚀

---

## ✨ Overview

**Gemini SRT Translator** is a powerful python tool to translate subtitle files using the power of Google Gemini AI. Perfect for anyone needing fast, accurate, and customizable translations for videos, movies, and series.

---

- 🔤 **Subtitle Translation**: Translate `.srt` and `.ass` subtitle files to a wide range of languages supported by Google Gemini AI.
- 🎙️ **Transcription**: Transcribe audio or video files directly into subtitles using Gemini's audio capabilities.
- ⏱️ **Timing & Format**: Ensures that the translated subtitles maintain the exact timestamps and native SRT/ASS formatting of the original file.
- 💾 **Quick Resume**: Easily resume interrupted translations from where you left off.
- 🧠 **Advanced AI**: Leverages thinking and reasoning capabilities for more contextually accurate translations (available on Gemini 2.5 and 3 models).
- 🖥️ **CLI Support**: Full command-line interface for easy automation and scripting.
- ⚙️ **Customizable**: Tune model parameters, adjust batch size, and access other advanced settings.
- 🎞️ **Subtitle Extraction**: Extract and translate native SRT or ASS subtitles from video files automatically (requires [FFmpeg](https://ffmpeg.org/)).
- 🎵 **Audio Context**: Extract audio from a video file or provide your own to improve translation accuracy (requires [FFmpeg](https://ffmpeg.org/)).
- 📜 **Description Support**: Add a description to your translation job to guide the AI in using specific terminology or context.
- 📋 **List Models**: Easily list all currently available Gemini models to choose the best fit for your needs.
- 🔄 **Auto-Update**: Keep the tool updated with automatic version checking and update prompts.
- 📝 **Logging**: Optional saving of progress and 'thinking' process logs for review.

---

## 📦 Installation

### Basic:

```sh
pip install --upgrade gemini-srt-translator
```

### Recommended: Use a Virtual Environment

It's best practice to use a virtual environment. This is especially recommended as gemini-srt-translator installs several dependencies that could potentially conflict with your existing packages:

```sh
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install inside the virtual environment
pip install --upgrade gemini-srt-translator
```

---

## 🔑 How to Get Your API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey).
2. Sign in with your Google account.
3. Click on **Generate API Key**.
4. Copy and keep your key safe.

### 🔐 Setting Your API Key

You can provide your API key in several ways:

1. **Environment Variable (Recommended)**: Set the `GEMINI_API_KEY` environment variable. This is the most secure and recommended method.

- **macOS/Linux:**

  ```bash
  export GEMINI_API_KEY="your_api_key_here"
  export GEMINI_API_KEY2="your_second_api_key_here"
  ```

- **Windows (Command Prompt):**

  ```cmd
  set GEMINI_API_KEY=your_api_key_here
  set GEMINI_API_KEY2=your_second_api_key_here
  ```

- **Windows (PowerShell):**
  ```powershell
  $env:GEMINI_API_KEY="your_api_key_here"
  $env:GEMINI_API_KEY2="your_second_api_key_here"
  ```

2. **Command Line Argument**: Use the `-k` or `--api-key` flag

```bash
gst translate -i subtitle.srt -l French -k YOUR_API_KEY
```

3. **Interactive Prompt**: The tool will prompt you if no key is found

```bash
gst translate -i subtitle.srt -l French
```

4. **Python API**: Set the `gemini_api_key` variable in your script

```python
import gemini_srt_translator as gst
gst.gemini_api_key = "your_api_key_here"
```

### ☁️ Agent Platform (Vertex AI) Support

You can also use Google Cloud's **Agent Platform** (formerly known as Vertex AI) instead of the standard Google AI Studio Gemini API.

You can configure it using environment variables or directly via command-line arguments / Python API options.

#### Setup Modes

##### 1. Using Application Default Credentials (ADC)

Recommended when running inside a Google Cloud environment or in local environments configured via `gcloud auth application-default login`.

- **Environment Variables**:

  ```bash
  export GOOGLE_GENAI_USE_ENTERPRISE="true"
  export GOOGLE_CLOUD_PROJECT="your-google-cloud-project-id"
  export GOOGLE_CLOUD_LOCATION="us-central1" # (optional, defaults to 'global')
  ```

- **CLI Flags**:

  ```bash
  gst translate -i subtitle.srt -l French --use-enterprise --cloud-project "your-project-id" --cloud-location "us-central1" --request-type "shared"
  ```

- **Python API**:
  ```python
  import gemini_srt_translator as gst
  gst.use_enterprise = True
  gst.cloud_project = "your-google-cloud-project-id"
  gst.cloud_location = "us-central1" # (optional, defaults to 'global')
  gst.request_type = "shared" # (optional)
  ```

##### 2. Using API Key (Express Mode)

Authenticate using a specific Google Cloud enterprise API key.

- **Environment Variables**:

  ```bash
  export GOOGLE_GENAI_USE_ENTERPRISE="true"
  export GOOGLE_API_KEY="your_google_api_key"
  ```

- **CLI Flags**:

  ```bash
  gst translate -i subtitle.srt -l French --use-enterprise --cloud-api-key "your-google-api-key" --request-type "dedicated"
  ```

- **Python API**:
  ```python
  import gemini_srt_translator as gst
  gst.use_enterprise = True
  gst.cloud_api_key = "your_google_api_key"
  gst.request_type = "dedicated" # (optional)
  ```

---

## 🚀 Quick Start

### 🖥️ Using the Command Line Interface (CLI)

#### Basic Translation

```bash
# Using environment variable (recommended)
export GEMINI_API_KEY="your_api_key_here"
gst translate -i subtitle.srt -l French

# Using command line argument
gst translate -i subtitle.srt -l French -k YOUR_API_KEY

# Set output file name (can be .srt or .ass)
gst translate -i subtitle.srt -l French -o translated_subtitle.srt

# Extract subtitles from video and translate (requires FFmpeg)
gst translate -v movie.mp4 -l Spanish

# Extract and use audio from video for context (requires FFmpeg)
gst translate -v movie.mp4 -l Spanish --extract-audio

# Interactive model selection
gst translate -i subtitle.srt -l "Brazilian Portuguese" --interactive

# Resume translation from a specific line
gst translate -i subtitle.srt -l French --start-line 20

# Limit saved-progress resume context to 50 previous lines (0 disables it)
gst translate -i subtitle.srt -l French --resume --resume-context-size 50

# Suppress output
gst translate -i subtitle.srt -l French --quiet
```

#### Advanced Options

```bash
# Full-featured translation with custom settings
gst translate \
  -i input.srt \
  -v video.mp4 \
  -l French \
  -k YOUR_API_KEY \
  -k2 YOUR_SECOND_API_KEY \
  -o output_french.srt \
  --model gemini-3.5-flash \
  --service-tier standard \
  --batch-size 150 \
  --temperature 0.7 \
  --description "Medical TV series, use medical terminology" \
  --progress-log \
  --thoughts-log \
  --extract-audio \
  --token-stats \
  --token-report \
  --no-context
```

#### Transcribing Audio/Video

```bash
# Transcribe a video file to SRT
gst transcribe -v video.mp4 -o transcription.srt

# Transcribe an audio file
gst transcribe -a audio.mp3 -o transcription.srt

# Transcribe with custom settings
gst transcribe \
  -v video.mp4 \
  -o transcription.srt \
  --model gemini-3.5-flash \
  --service-tier standard \
  --description "Meeting recording about project X" \
  --thinking-level high \
  --token-stats \
  --token-report
```

#### Extracting Audio/Subtitles

```bash
# Extract SRT from video
gst extract -v video.mp4 --srt

# Extract Audio from video
gst extract -v video.mp4 --audio

# Extract both with voice isolation (default)
gst extract -v video.mp4 --srt --audio

# Extract audio without voice isolation
gst extract -v video.mp4 --audio --no-voice-isolation
```

#### CLI Help

```bash
# Show all available commands and options
gst --help

# Show specific command help
gst translate --help
gst transcribe --help
gst extract --help
```

### 🐍 Using Python API

#### Translating an SRT file

```python
import gemini_srt_translator as gst

gst.gemini_api_key = "your_api_key_here"
gst.target_language = "French"
gst.input_file = "subtitle.srt"

gst.translate()
```

#### Resuming an Interrupted Translation

Just run again with the same parameters, or specify the start line:

```python
import gemini_srt_translator as gst

gst.gemini_api_key = "your_api_key_here"
gst.target_language = "French"
gst.input_file = "subtitle.srt"
gst.start_line = 20

gst.translate()
```

#### Transcribing Audio/Video

```python
import gemini_srt_translator as gst

gst.gemini_api_key = "your_api_key"
gst.video_file = "video.mp4" # Or gst.audio_file = "audio.mp3"
gst.output_file = "transcription.srt"
gst.model_name = "gemini-3.5-flash"
gst.token_stats = True

gst.transcribe()
```

#### Extracting from Video

```python
import gemini_srt_translator as gst

gst.video_file = "video.mp4"

# Extract SRT
gst.extract("srt")

# Extract Audio (with voice isolation by default)
gst.extract("audio")

# Extract Audio (without voice isolation)
gst.isolate_voice = False
gst.extract("audio")
```

---

## ⚙️ Advanced Configuration

#### 🔧 GST Parameters

- `gemini_api_key2`: Second key for more quota (useful for free Pro models).
- `use_enterprise`: Enable Enterprise / Agent Platform mode (default: False).
- `cloud_api_key`: Google Cloud API key for Agent Platform Express mode.
- `cloud_project`: Google Cloud Project ID for agent platform authentication (ADC).
- `cloud_location`: Google Cloud Location for agent platform authentication (default: "global").
- `request_type`: Agent Platform request type (options: `shared`, `dedicated`).
- `video_file`: Path to a video file to extract subtitles and/or audio for context (requires [FFmpeg](https://ffmpeg.org/)).
- `audio_file`: Path to an audio file to use as context for translation (requires [FFmpeg](https://ffmpeg.org/)).
- `extract_audio`: Whether to extract and use audio context from the video file (default: False).
- `isolate_voice`: Whether to isolate voice from audio (default: True).
- `audio_chunk_size`: Audio chunk size in seconds for processing (default: 600).
- `output_file`: Name of the translated file.
- `start_line`: Starting line for translation.
- `resume_context_size`: Number of previous lines to include as context when resuming (default: 50, 0 disables resume context).
- `description`: Description of the translation job.
- `batch_size`: Batch size (default: 1000).
- `free_quota`: Signal GST that you are using a free quota (default: True).
- `skip_upgrade`: Skip version upgrade check (default: False).
- `use_colors`: Activate colors in terminal (default: True).
- `progress_log`: Enable progress logging to a file (default: False).
- `thoughts_log`: Enable logging of the 'thinking' process to a file (default: False).
- `quiet`: Suppress all output (default: False).
- `resume`: Skip prompt and set automatic resume mode.
- `token_stats`: Show token usage information (default: False).
- `token_report`: Save token usage and run statistics to a JSON file. If passed via CLI without arguments, it defaults to using the first available input filename with a `_token_report.json` suffix (default: None).
- `preserve_context`: Preserve context between batches (default: True).

#### 🔬 Model Tuning Parameters

- `model_name`: Gemini model (default: "gemini-3.5-flash").
- `temperature`: Controls randomness in output. Lower for more deterministic, higher for more creative (range: 0.0-2.0).
- `top_p`: Nucleus sampling parameter (range: 0.0-1.0).
- `top_k`: Top-k sampling parameter (range: >=0).
- `streaming`: Enable streamed responses (default: True).
  - Set to `False` for bad internet connections or when using slower models.
- `thinking`: Enable thinking capability for potentially more accurate translations (default: True).
  - Only available for Gemini 2.5 and 3 models.
- `thinking_budget`: Token budget for the thinking process (range: 0-32768, 0 also disables thinking).
  - Only available for Gemini 2.5 models.
- `thinking_level`: Controls the depth of thinking process (options: minimal, low, medium, high).
  - Only available for Gemini 3 models.
- `service_tier`: Service tier for Gemini API (options: `standard`, `flex`, `priority`).
  - Only available on paid plans.

#### 💡 Full example:

```python
import gemini_srt_translator as gst

gst.gemini_api_key = "your_api_key_here"
gst.gemini_api_key2 = "your_api_key2_here"
gst.target_language = "French"
gst.input_file = "subtitle.srt"
gst.output_file = "subtitle_translated.srt"
gst.video_file = "video.mp4"
gst.audio_file = "audio.mp3"
gst.extract_audio = False
gst.start_line = 20
gst.description = "Medical TV series, use medical terms"
gst.model_name = "gemini-3.5-flash"
gst.batch_size = 150
gst.streaming = True
gst.thinking = True
gst.thinking_budget = 4096
gst.thinking_level = "high"
gst.temperature = 0.7
gst.top_p = 0.95
gst.top_k = 20
gst.free_quota = False
gst.skip_upgrade = True
gst.use_colors = False
gst.progress_log = True
gst.thoughts_log = True
gst.quiet = False
gst.resume = True
gst.resume_context_size = 50
gst.token_stats = True
gst.token_report = "token_report.json"
gst.preserve_context = True
gst.service_tier = "standard"

gst.translate()
```

---

## 📚 Listing Available Models

### CLI

```bash
gst list-models -k YOUR_API_KEY
```

### Python API

```python
import gemini_srt_translator as gst

gst.gemini_api_key = "your_api_key_here"
gst.listmodels()
```

---

## 🎨 Unofficial GUI Applications

If you prefer a user-friendly graphical interface over command-line usage, be sure to check out:

- **[🔗 Gemini SRT Translator GUI](https://github.com/mkaflowski/Gemini-SRT-translator-GUI) (by @mkaflowski)**
- **[🔗 Gemini SRT Translator GUI](https://github.com/dane-9/gemini-srt-translator-gui) (by @dane-9)**

Perfect for users who want the same powerful translation capabilities with an intuitive visual interface!

---

## 📝 License

Distributed under the MIT License. See the [LICENSE](https://github.com/MaKTaiL/gemini-srt-translator?tab=MIT-1-ov-file) file for details.

---

## 👥 Contributors

Thank you to all who have contributed to this project:

- [MaKTaiL](https://github.com/MaKTaiL) - Creator and maintainer
- [CevreMuhendisi](https://github.com/CevreMuhendisi)
- [angelitto2005](https://github.com/angelitto2005)
- [sjiampojamarn](https://github.com/sjiampojamarn)
- [mkaflowski](https://github.com/mkaflowski)
- [iceman1010](https://github.com/iceman1010)
- [kingzleshe](https://github.com/kingzleshe)

Special thanks to all users who have reported issues, suggested features, and helped improve the project.
