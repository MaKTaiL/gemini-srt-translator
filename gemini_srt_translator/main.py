# gemini_srt_translator.py

import typing
import json
import time
import unicodedata as ud

import srt
from srt import Subtitle
from collections import Counter

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, ContentDict
from google.generativeai import GenerativeModel

class SubtitleObject(typing.TypedDict):
    index: str
    content: str
    _blank: str

class GeminiSRTTranslator:
    def __init__(self, gemini_api_key: str = None, gemini_api_key2: str = None, target_language: str = None, input_file: str = None, output_file: str = None, description: str = None, model_name: str = "gemini-1.5-flash", batch_size: int = 30, free_quota: bool = True):
        self.gemini_api_key = gemini_api_key
        self.gemini_api_key2 = gemini_api_key2  # İkinci API anahtarı
        self.target_language = target_language
        self.input_file = input_file
        self.output_file = output_file
        self.description = description
        self.model_name = model_name
        self.batch_size = batch_size
        self.free_quota = free_quota
        self.current_api_key = self.gemini_api_key  # Başlangıçta ilk API anahtarını kullan

    def listmodels(self):
        """
        Gemini API'sinden kullanılabilen modelleri listeler.
        """
        if not self.current_api_key:
            raise Exception("Lütfen geçerli bir Gemini API anahtarı sağlayın.")

        genai.configure(api_key=self.current_api_key)
        models = genai.list_models()
        for model in models:
            if "generateContent" in model.supported_generation_methods:
                print(model.name.replace("models/", ""))

    def translate(self):
        """
        Bir altyazı dosyasını Gemini API'sini kullanarak çevirir.
        """
        if not self.current_api_key:
            raise Exception("Lütfen geçerli bir Gemini API anahtarı sağlayın.")
        
        if not self.target_language:
            raise Exception("Lütfen bir hedef dil sağlayın.")
        
        if not self.input_file:
            raise Exception("Lütfen bir altyazı dosyası sağlayın.")
        
        if not self.output_file:
            self.output_file = ".".join(self.input_file.split(".")[:-1]) + "_translated.srt"

        genai.configure(api_key=self.current_api_key)

        instruction = f"""You are an assistant that translates subtitles to {self.target_language}.
You will receive the following JSON type:

class SubtitleObject(typing.TypedDict):
    index: str
    content: str

Request: list[SubtitleObject]
Response: list[SubtitleObject]

The 'index' key is the index of the subtitle line.
The 'content' key is the text to be translated.

The size of the list must remain the same as the one you received."""

        if self.description:
            instruction += "\nAdditional user instruction: '" + self.description + "'"

        model = genai.GenerativeModel(
            model_name=self.model_name,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
            system_instruction=instruction,
            generation_config=genai.GenerationConfig(response_mime_type="application/json", temperature=0)
        )

        with open(self.input_file, "r", encoding="utf-8") as original_file, open(self.output_file, "w", encoding="utf-8") as translated_file:
            original_text = original_file.read()

            original_subtitle = list(srt.parse(original_text))
            translated_subtitle = original_subtitle.copy()

            i = 0
            total = len(original_subtitle)
            batch = []
            previous_message = None
            reverted = 0
            delay = False

            if 'pro' in self.model_name and self.free_quota:
                delay = True
                print("Pro model and free user quota detected, enabling 30s delay between requests...")

            batch.append(SubtitleObject(index=str(i), content=original_subtitle[i].content))
            i += 1

            print(f"Starting translation of {total} lines...")

            while len(batch) > 0:
                if i < total and len(batch) < self.batch_size:
                    batch.append(SubtitleObject(index=str(i), content=original_subtitle[i].content))
                    i += 1
                    continue
                try:
                    start_time = time.time()
                    previous_message = self._process_batch(model, batch, previous_message, translated_subtitle)
                    end_time = time.time()
                    print(f"Translated {i}/{total}")
                    if delay and (end_time - start_time < 30):
                        time.sleep(30 - (end_time - start_time))
                    if reverted > 0:
                        self.batch_size += reverted
                        reverted = 0
                        print("Increasing batch size back to {}...".format(self.batch_size))
                    if i < total and len(batch) < self.batch_size:
                        batch.append(SubtitleObject(index=str(i), content=original_subtitle[i].content))
                        i += 1
                except Exception as e:
                    e = str(e)
                    if "block" in e:
                        print(e)
                        batch.clear()
                        break
                    elif "quota" in e:
                        if self.current_api_key == self.gemini_api_key and self.gemini_api_key2:
                            print("Quota exceeded for first API key, switching to second API key...")
                            self.current_api_key = self.gemini_api_key2
                            genai.configure(api_key=self.current_api_key)
                            model = genai.GenerativeModel(
                                model_name=self.model_name,
                                safety_settings={
                                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                                },
                                system_instruction=instruction,
                                generation_config=genai.GenerationConfig(response_mime_type="application/json", temperature=0)
                            )
                        else:
                            print("Quota exceeded, waiting 1 minute...")
                            time.sleep(60)
                    else:
                        if self.batch_size == 1:
                            raise Exception("Translation failed, aborting...")
                        if self.batch_size > 1:
                            decrement = min(10, self.batch_size - 1)
                            reverted += decrement
                            for _ in range(decrement):
                                i -= 1
                                batch.pop()
                            self.batch_size -= decrement
                        if "finish_reason" in e:
                            print("Gemini has blocked the translation for unknown reasons")
                        else:
                            print(e)
                        print("Decreasing batch size to {} and trying again...".format(self.batch_size))
            
            translated_file.write(srt.compose(translated_subtitle))

    def _process_batch(self, model: GenerativeModel, batch: list[SubtitleObject], previous_message: ContentDict, translated_subtitle: list[Subtitle]) -> ContentDict:
        """
        Bir grup altyazıyı işler.
        """
        if previous_message:
            messages = [previous_message] + [{"role": "user", "parts": json.dumps(batch)}]
        else:
            messages = [{"role": "user", "parts": json.dumps(batch)}]
        response = model.generate_content(messages)
        translated_lines: list[SubtitleObject] = json.loads(response.text)
        if len(translated_lines) != len(batch):
            raise Exception("Gemini has returned the wrong number of lines.")
        for line in translated_lines:
            if line["index"] not in [x["index"] for x in batch]:
                raise Exception("Gemini has returned different indices.")
            if self.dominant_strong_direction(line["content"]) == "rtl":
                translated_subtitle[int(line["index"])].content = f"\u202B{line['content']}\u202C"
            else:
                translated_subtitle[int(line["index"])].content = line["content"]
        batch.clear()
        return response.candidates[0].content

    def dominant_strong_direction(self, s: str) -> str:
        """
        Bir dizenin baskın güçlü yönünü belirler.
        """
        count = Counter([ud.bidirectional(c) for c in list(s)])
        rtl_count = count['R'] + count['AL'] + count['RLE'] + count["RLI"]
        ltr_count = count['L'] + count['LRE'] + count["LRI"]
        return "rtl" if rtl_count > ltr_count else "ltr"
