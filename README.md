# 🌟 Gemini SRT Translator

[![PyPI version](https://img.shields.io/pypi/v/gemini-srt-translator)](https://pypi.org/project/gemini-srt-translator)
[![Python support](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FMaKTaiL%2Fgemini-srt-translator%2Frefs%2Fheads%2Fmain%2Fpyproject.toml&color=red)](https://pypi.org/project/gemini-srt-translator)
[![Downloads](https://img.shields.io/pypi/dw/gemini-srt-translator)](https://pypi.org/project/gemini-srt-translator)

> Translate SRT subtitle files using the power of Google Gemini AI! 🚀

---

## ✨ Overview

**Gemini SRT Translator** is a powerful python tool to translate subtitle files using the power of Google Gemini AI. Perfect for anyone needing fast, accurate, and customizable translations for videos, movies, and series.

---

- 🔤 **SRT Translation**: Translate `.srt` subtitle files to a wide range of languages supported by Google Gemini AI.
- ⏱️ **Timing & Format**: Ensures that the translated subtitles maintain the exact timestamps and basic SRT formatting of the original file.
- 💾 **Quick Resume**: Easily resume interrupted translations from where you left off.
- 🧠 **Advanced AI**: Leverages thinking and reasoning capabilities for more contextually accurate translations (available on Gemini 2.5 models).
- ⚙️ **Customizable**: Tune model parameters, adjust batch size, and access other advanced settings.
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

1. Go to [Google AI Studio API Key](https://aistudio.google.com/apikey).
2. Sign in with your Google account.
3. Click on **Generate API Key**.
4. Copy and keep your key safe.

---

## 🚀 Quick Start

### 🌐 Translating an SRT file

```python
import gemini_srt_translator as gst

gst.gemini_api_key = "your_api_key_here"
gst.target_language = "French"
gst.input_file = "subtitle.srt"

gst.translate()
```

---

### ⏸️ Resuming an Interrupted Translation

Just run again with the same parameters, or specify the start line:

```python
import gemini_srt_translator as gst

gst.gemini_api_key = "your_api_key_here"
gst.target_language = "French"
gst.input_file = "subtitle.srt"
gst.start_line = 20

gst.translate()
```

---

## ⚙️ Advanced Configuration

#### 🔧 GST Parameters

- `gemini_api_key2`: Second key for more quota (useful for free Pro models).
- `output_file`: Name of the translated file.
- `start_line`: Starting line for translation.
- `description`: Description of the translation job.
- `batch_size`: Batch size (default: 300).
- `free_quota`: Signal GST that you are using a free quota (default: True).
- `skip_upgrade`: Skip version upgrade check (default: False).
- `use_colors`: Activate colors in terminal (default: True).
- `progress_log`: Enable progress logging to a file (default: False).
- `thoughts_log`: Enable logging of the 'thinking' process to a file (default: False).

#### 🔬 Model Tuning Parameters

- `model_name`: Gemini model (default: "gemini-2.5-flash-preview-05-20").
- `temperature`: Controls randomness in output. Lower for more deterministic, higher for more creative (range: 0.0-2.0).
- `top_p`: Nucleus sampling parameter (range: 0.0-1.0).
- `top_k`: Top-k sampling parameter (range: >=0).
- `streaming`: Enable streamed responses (default: True).
  - Set to `False` for bad internet connections or when using slower models.
- `thinking`: Enable thinking capability for potentially more accurate translations (default: True).
  - Only available for Gemini 2.5 models.
- `thinking_budget`: Token budget for the thinking process (default: 2048, range: 0-24576, 0 also disables thinking).
  - Only available for Gemini 2.5 Flash (for now).

#### 💡 Full example:

```python
import gemini_srt_translator as gst

gst.gemini_api_key = "your_api_key_here"
gst.gemini_api_key2 = "your_api_key2_here"
gst.target_language = "French"
gst.input_file = "subtitle.srt"
gst.output_file = "subtitle_translated.srt"
gst.start_line = 20
gst.description = "Medical TV series, use medical terms"
gst.model_name = "gemini-2.5-pro-preview-03-25"
gst.batch_size = 150
gst.streaming = True
gst.thinking = True
gst.thinking_budget = 4096
gst.temperature = 0.7
gst.top_p = 0.95
gst.top_k = 20
gst.free_quota = False
gst.skip_upgrade = True
gst.use_colors = False
gst.progress_log = True
gst.thoughts_log = True

gst.translate()
```

---

## 📚 Listing Available Models

See all available Gemini models:

```python
import gemini_srt_translator as gst

gst.gemini_api_key = "your_api_key_here"
gst.listmodels()
```

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

Special thanks to all users who have reported issues, suggested features, and helped improve the project.

---
