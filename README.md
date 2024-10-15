# Gemini SRT Translator

[![PyPI version](https://badge.fury.io/py/gemini-srt-translator.svg)](https://badge.fury.io/py/gemini-srt-translator)
![Python support](https://img.shields.io/pypi/pyversions/gemini-srt-translator)

## Overview

Gemini SRT Translator is a tool designed to translate subtitle files using Google Generative AI. It leverages the power of the Gemini API to provide accurate and efficient translations for your subtitle files.

## Features

- Translate subtitle files to a specified target language.
- Customize translation settings such as model name and batch size.
- List available models from the Gemini API.

## Installation

To install Gemini SRT Translator, use pip:

```sh
pip install gemini-srt-translator
```

## Setup

### Generating Your API Key

To use Gemini SRT Translator, you need to generate a free API key from Google AI Studio. Follow these steps:

1. Go to [Google AI Studio API Key](https://aistudio.google.com/apikey).
2. Sign in with your Google account.
3. Click on "Generate API Key".
4. Copy the generated API key and keep it secure.

You will use this API key for setting up the script before starting the translation process.

## Usage

### Translate Subtitles

You can translate subtitles using the `translate` command:

```python
import gemini_srt_translator as gst

gst.gemini_api_key = "your_gemini_api_key_here"
gst.target_language = "French"
gst.input_file = "subtitle.srt"

gst.translate()
```

This will translate the subtitles in the `subtitle.srt` file to French.

## Configuration

You can further customize the translation settings by providing optional parameters:

- `output_file`: Path to save the translated subtitle file.
- `model_name`: Model name to use for translation.
- `batch_size`: Batch size for translation.

Example:

```python
import gemini_srt_translator as gst

gst.gemini_api_key = "your_gemini_api_key_here"
gst.target_language = "French"
gst.input_file = "subtitle.srt"
gst.output_file = "subtitle_translated.srt"
gst.model_name = "gemini-1.5-flash"
gst.batch_size = 30

gst.translate()
```

### List Models

You can list the available models using the `listmodels` command:

```python
import gemini_srt_translator as gst

gst.gemini_api_key = "your_gemini_api_key_here"
gst.listmodels()
```

This will print a list of available models to the console.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/MaKTaiL/gemini-srt-translator?tab=MIT-1-ov-file) file for details.
