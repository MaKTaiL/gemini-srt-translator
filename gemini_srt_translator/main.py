# gemini_srt_translator.py

import srt
import json
import time
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

class GeminiSRTTranslator:
    def __init__(self, gemini_api_key: str = None, target_language: str = None, input_file: str = None, output_file: str = None, model_name: str = "gemini-1.5-flash", batch_size: int = 30):
        self.gemini_api_key = gemini_api_key
        self.target_language = target_language
        self.input_file = input_file
        self.output_file = output_file
        self.model_name = model_name
        self.batch_size = batch_size

    def listmodels(self):
        """
        Lists available models from the Gemini API.
        """
        if not self.gemini_api_key:
            raise Exception("Please provide a valid Gemini API key.")

        genai.configure(api_key=self.gemini_api_key)
        models = genai.list_models()
        for model in models:
            if "generateContent" in model.supported_generation_methods:
                print(model.name.replace("models/", ""))

    def translate(self):
        """
        Translates a subtitle file using the Gemini API.
        """
        if not self.gemini_api_key:
            raise Exception("Please provide a valid Gemini API key.")
        
        if not self.target_language:
            raise Exception("Please provide a target language.")
        
        if not self.input_file:
            raise Exception("Please provide a subtitle file.")
        
        if not self.output_file:
            self.output_file = ".".join(self.input_file.split(".")[:-1]) + "_translated.srt"

        genai.configure(api_key=self.gemini_api_key)

        instruction = f"""You are an assistant that translates movies/series subtitles to {self.target_language}.
        User will send a json and you must return a copy of it with the dialogues translated. Do not change the numbering of the indices."""

        model = genai.GenerativeModel(
            model_name=self.model_name,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
            system_instruction=instruction,
            generation_config={"response_mime_type": "application/json"},
        )

        with open(self.input_file, "r", encoding="utf-8") as original_file, open(self.output_file, "w", encoding="utf-8") as translated_file:
            original_text = original_file.read()

            original_subtitle = list(srt.parse(original_text))
            translated_subtitle = original_subtitle.copy()

            chat = model.start_chat()

            i = 0
            total = len(original_subtitle)
            batch = {}

            print(f"Starting translation of {total} lines...")
            while i < total:
                if len(batch) < self.batch_size:
                    batch[str(i)] = original_subtitle[i].content
                    i += 1
                    continue
                else:
                    try:
                        response = chat.send_message(json.dumps(batch))
                        try:
                            translated_lines = json.loads(response.text)
                            if len(translated_lines) != len(batch):
                                raise Exception("Gemini has returned a different number of lines than the original, trying again...")
                            for x in translated_lines:
                                if x not in batch:
                                    raise Exception("Gemini has returned different indices than the original, trying again...")
                        except Exception as e:
                            print(e)
                            chat.rewind()
                            continue
                        for x in translated_lines:
                            translated_subtitle[int(x)].content = translated_lines[x]
                        batch = {}

                        print(f"Translated {i}/{total}")
                    except Exception as e:
                        e = str(e)
                        if "block" in e:
                            print(e)
                            batch = {}
                            break
                        elif "quota" in e:
                            print("Quota exceeded, waiting 1 minute...")
                            time.sleep(60)
                            continue
                        else:
                            print(e)
                            continue
            while len(batch) > 0:
                try:
                    response = chat.send_message(json.dumps(batch))
                    try:
                        translated_lines = json.loads(response.text)
                        if len(translated_lines) != len(batch):
                            raise Exception("Gemini has returned a different number of lines than the original, trying again...")
                        for x in translated_lines:
                            if x not in batch:
                                raise Exception("Gemini has returned different indices than the original, trying again...")
                    except Exception as e:
                        print(e)
                        chat.rewind()
                        continue
                    for x in translated_lines:
                        translated_subtitle[int(x)].content = translated_lines[x]
                    batch = {}

                    print(f"Translated {i}/{total}")
                except Exception as e:
                        e = str(e)
                        if "block" in e:
                            print(e)
                            batch = {}
                            break
                        elif "quota" in e:
                            print("Quota exceeded, waiting 1 minute...")
                            time.sleep(60)
                            continue
                        else:
                            print(e)
                            continue

            translated_file.write(srt.compose(translated_subtitle))
