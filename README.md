# Translator Project

This project provides tools to interact with Google Generative AI for listing available models and translating subtitle files.

## Setup

1. **Create and activate a virtual environment**:

   ```sh
   python3.12 -m venv venv
   # On Windows:
   python -m venv venv
   venv\Scripts\activate.bat
   # On Unix or MacOS:
   python3.12 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**:

   ```sh
   pip install -r requirements.txt
   ```

## Configuration

Configure the necessary variables in `translator.py` located in the project directory:

- **`gemini_key`**: Your Google Generative AI API key.

  ```python
  gemini_key = "YOUR_GOOGLE_GENERATIVE_AI_API_KEY"
  ```

- **`origin_language`**: The language of the original subtitles.

  ```python
  origin_language = "English"
  ```

- **`target_language`**: The language to translate the subtitles into.

  ```python
  target_language = "Brazilian Portuguese"
  ```

- **`origin_file`**: The path to the original subtitle file.

  ```python
  origin_file = "Original File.srt"
  ```

- **`translated_file`**: The path where the translated subtitle file will be saved.

  ```python
  translated_file = "Translated File.srt"
  ```

- **`model_name`**: The name of the Google Generative AI model to use.

  ```python
  model_name = "gemini-1.5-flash-latest"
  ```

## Usage

1. **List Available Models**:

   Run `listmodels.py` to list available Google Generative AI models.

   ```sh
   python listmodels.py
   ```

2. **Translate Subtitles**:

   Run `translator.py` to translate the subtitle file.

   ```sh
   python translator.py
   ```
