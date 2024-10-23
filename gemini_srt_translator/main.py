# gemini_srt_translator.py

import typing
import json
import time
import unicodedata as ud

import srt
from srt import Subtitle
from collections import Counter

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.generativeai import ChatSession

class SubtitleObject(typing.TypedDict):
    index: str
    content: str
    _blank: str

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

        instruction = f"""You are an assistant that translates subtitles to {self.target_language}.
        You will receive the following JSON type:

        class SubtitleObject(typing.TypedDict):
            index: str
            content: str

        Request: list[SubtitleObject]
        Response: list[SubtitleObject]
        
        The 'index' key is the index of the subtitle line.
        The 'content' key is the text to be translated.
        
        The size of the list must remain the same as the one you received.
        """

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

        chat = model.start_chat()

        with open(self.input_file, "r", encoding="utf-8") as original_file, open(self.output_file, "w", encoding="utf-8") as translated_file:
            original_text = original_file.read()

            original_subtitle = list(srt.parse(original_text))
            translated_subtitle = original_subtitle.copy()

            i = 0
            total = len(original_subtitle)
            batch = []
            reverted = 0

            batch.append(SubtitleObject(index=str(i), content=original_subtitle[i].content))
            i += 1

            print(f"Starting translation of {total} lines...")

            while len(batch) > 0:
                if i < total and len(batch) < self.batch_size:
                    batch.append(SubtitleObject(index=str(i), content=original_subtitle[i].content))
                    i += 1
                    continue
                try:
                    self._process_batch(chat, batch, translated_subtitle)
                    print(f"Translated {i}/{total}")
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
                        chat.rewind()
                        print(e)
                        print("Decreasing batch size to {} and trying again...".format(self.batch_size))
            
            translated_file.write(srt.compose(translated_subtitle))

    def _process_batch(self, chat: ChatSession, batch: list[SubtitleObject], translated_subtitle: list[Subtitle]) -> None:
        """
        Processes a batch of subtitles.
        """
        response = chat.send_message(json.dumps(batch))
        translated_lines: list[SubtitleObject] = json.loads(response.text)
        if len(translated_lines) != len(batch):
            raise Exception("Gemini has returned the wrong number of lines.")
        for line in translated_lines:
            if line["index"] not in [x["index"] for x in batch]:
                raise Exception("Gemini has returned different indices.")
            if self.dominant_strong_direction(line["content"]) == "rtl":
                translated_subtitle[int(line["index"])].content = f"\u202B{line["content"]}\u202C"
            else:
                translated_subtitle[int(line["index"])].content = line["content"]
        batch.clear()

    def dominant_strong_direction(self, s: str) -> str:
        """
        Determines the dominant strong direction of a string.
        """
        count = Counter([ud.bidirectional(c) for c in list(s)])
        rtl_count = count['R'] + count['AL'] + count['RLE'] + count["RLI"]
        ltr_count = count['L'] + count['LRE'] + count["LRI"]
        return "rtl" if rtl_count > ltr_count else "ltr"
