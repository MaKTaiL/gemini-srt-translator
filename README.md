# 🌟 Gemini SRT Translator

[![PyPI version](https://img.shields.io/pypi/v/gemini-srt-translator)](https://pypi.org/project/gemini-srt-translator)
[![Python support](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FMaKTaiL%2Fgemini-srt-translator%2Frefs%2Fheads%2Fmain%2Fpyproject.toml&color=red)](https://pypi.org/project/gemini-srt-translator)
[![Downloads](https://img.shields.io/pypi/dw/gemini-srt-translator)](https://pypi.org/project/gemini-srt-translator)

> Translate SRT subtitle files using the power of Google Gemini AI! 🚀

---

## ✨ Overview

**Gemini SRT Translator** is a powerful python tool to translate subtitle (.srt) files using the Google Gemini AI. Perfect for anyone needing fast, accurate, and customizable translations for videos, movies, and series.

---

## 🛠️ Features

- 🔤 Translate SRT files to any language.
- ⏱️ Preserve the original timing and formatting.
- 💾 Easily resume interrupted translations.
- ⚙️ Flexible configuration: model, batch size, and more.
- 📋 List available models.
- 🔄 Automatic version checking and updates.
- 📝 Optional error log saving.

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

Customize even further:

- `gemini_api_key2`: Second key for more quota (useful for free Pro models).
- `output_file`: Name of the translated file.
- `start_line`: Starting line for translation.
- `description`: Description of the translation job.
- `model_name`: Gemini model (default: "gemini-2.0-flash").
- `batch_size`: Batch size (default: 100).
- `free_quota`: Signal GST that you are using a free quota (default: True).
- `skip_upgrade`: Skip version upgrade check (default: False).
- `use_colors`: Activate colors in terminal (default: True).
- `error_log`: Save error logs to a file (default: False).
- `disable_streaming`: Disable streamed responses. (default: False).
  - Good for bad internet connections or when using slower models.

#### Full example:

```python
import gemini_srt_translator as gst

gst.gemini_api_key = "your_api_key_here"
gst.gemini_api_key2 = "your_api_key2_here"
gst.target_language = "French"
gst.input_file = "subtitle.srt"
gst.output_file = "subtitle_translated.srt"
gst.start_line = 20
gst.description = "Medical TV series, use medical terms"
gst.model_name = "gemini-2.0-flash"
gst.batch_size = 50
gst.free_quota = False
gst.skip_upgrade = True
gst.use_colors = False
gst.error_log = True
gst.disable_streaming = True

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
