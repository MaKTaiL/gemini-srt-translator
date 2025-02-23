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

from pydub import AudioSegment, silence
from fs.memoryfs import MemoryFS
import pysrt
import random
from ffmpeg import FFmpeg

class SubtitleObject(typing.TypedDict):
    """
    TypedDict for subtitle objects used in translation
    """
    index: str
    content: str

class GeminiSRTTranslator:
    """
    A translator class that uses Gemini API to translate subtitles.
    """
    def __init__(self, gemini_api_key: str = None, gemini_api_key2: str = None, target_language: str = None, 
                 input_file: str = None, output_file: str = None, description: str = None, 
                 model_name: str = "gemini-2.0-flash", batch_size: int = 30, free_quota: bool = True,
                 is_input_audio: bool = False,
                 model_name_audio: str = "gemini-2.0-flash-thinking-exp",
                 extract_srt_from_media: bool = False):
        """
        Initialize the translator with necessary parameters.

        Args:
            gemini_api_key (str): Primary Gemini API key
            gemini_api_key2 (str): Secondary Gemini API key for additional quota
            target_language (str): Target language for translation
            input_file (str): Path to input subtitle file
            output_file (str): Path to output translated subtitle file
            description (str): Additional instructions for translation
            model_name (str): Gemini model to use
            batch_size (int): Number of subtitles to process in each batch
            free_quota (bool): Whether to use free quota (affects rate limiting)
            is_input_audio (bool): Whether input file is in an audio format
            model_name_audio (str): Gemini model to use for audio transcribing
            extract_srt_from_media (bool): Whether to extract input srt from input file
        """
        self.gemini_api_key = gemini_api_key
        self.gemini_api_key2 = gemini_api_key2
        self.current_api_key = gemini_api_key
        self.current_api_number = 1
        self.backup_api_number = 2
        self.target_language = target_language
        self.input_file = input_file
        self.output_file = output_file
        self.description = description
        self.model_name = model_name
        self.batch_size = batch_size
        self.free_quota = free_quota
        self.is_input_audio = is_input_audio
        self.model_name_audio = model_name_audio
        self.extract_srt_from_media = extract_srt_from_media

    def listmodels(self):
        """List available Gemini models that support content generation."""
        if not self.current_api_key:
            raise Exception("Please provide a valid Gemini API key.")

        genai.configure(api_key=self.current_api_key)
        models = genai.list_models()
        for model in models:
            if "generateContent" in model.supported_generation_methods:
                print(model.name.replace("models/", ""))


    def segment_audio(self, audio: AudioSegment, segment_length: int = 100000, current_pos = 0):
        total_length = len(audio)
        current_position = current_pos
        segments = []
        while current_position < total_length:
            # Calculate end position for current segment. At least segment_length and at most 2x.
            end_position = min(current_position + segment_length, total_length)
            if (end_position + segment_length) > total_length:
                end_position = total_length
            segments.append((current_position, end_position))
            current_position = end_position
        return segments


    def translate(self):
        """
        Main translation method. Reads the input subtitle file, translates it in batches,
        and writes the translated subtitles to the output file.
        """
        if not self.current_api_key:
            raise Exception("Please provide a valid Gemini API key.")
        
        if not self.target_language:
            raise Exception("Please provide a target language.")
        
        if not self.input_file:
            raise Exception("Please provide a subtitle file.")

        if not self.output_file:
            self.output_file = ".".join(self.input_file.split(".")[:-1]) + ".translated_" + self.target_language + ".srt"

        if self.is_input_audio:
            prompt = f"""
You are a professional transcriber proficient in all languages, specializing in creating subtitles for film and television. Your task is to transcribe the following audio input and generate subtitles in SRT format. Prioritize accuracy and synchronization above all else. The output must be suitable for direct use as subtitles.

**Example SRT Format:**
1
00:00:01,644 --> 00:00:02,900
subtitle segment 1 content
---
2
00:00:03,001 --> 00:00:05,123
subtitle segment 2 multiple line 1
subittle another line 2
---
3
00:00:06,001 --> 00:00:07,443
subtitle segment 3
---

**CRITICAL REQUIREMENTS:**
1. **Timestamp Accuracy:** All timestamps MUST be in valid SRT format: hours:minutes:seconds,milliseconds --> hours:minutes:seconds,milliseconds, representing the precise beginning and ending timecode aligned with the audio. Inaccurate timestamps will result in rejection.
2. **Segment Length:** Each subtitle segment MUST contain at most 1-2 lines of text and MUST have a maximum duration of 3 seconds. Break long sentences into shorter segments to ensure readability. Avoid segments shorter than 500 milliseconds unless absolutely necessary. Failure to adhere to this segment lenght will result in rejection.
3. **SRT Formatting:**  The output MUST be a valid SRT file, adhering to the example format above:
  * A sequential numerical index.
  * A correctly formatted timestamp line (as specified above).
  * 1-2 lines of transcribed text (the content).
  * Each subtitle entry is separated by '---'. Failure to adhere to this format will result in rejection. The transcribed text must be encoded in UTF-8.
4. **Complete Transcription:** Ensure timestamps are precisely synchronized with the audio input. The output MUST cover the entire input audio filewithout overlapped timestamps.
5. **Sound Effect Handling:** Do not transcribe non-speech audio such as screams, roars, or other sound effects unless they are integral to the dialogue or narrative and require specific textual representation (e.g., "(Screaming)" or a similar descriptive phrase).  Clearly indicate such sounds.
6. **Output Sorting:** The subtitle segments MUST be sorted by beginning timecode in ascending order.
            """

            print(prompt)
            model = self._get_model(prompt, model_name = self.model_name_audio)
            print(f"""Reading audio file: {self.input_file}""")
            audio = AudioSegment.from_file(self.input_file)
            print(f"""Segmenting audio of total len: {len(audio)}""")
            segments = self.segment_audio(audio)
            print(segments)
            mem_fs = MemoryFS()
            final_subs = pysrt.from_string("")
            error = 0

            while len(segments) > 0:
                try:
                    segment = segments[0]
                    start = segment[0]
                    end = segment[1]
                    with mem_fs.open('tmp-input.mp3', 'wb') as f:
                        audio[start:end].export(f, format='mp3')
                    with mem_fs.open('tmp-input.mp3', 'rb') as f:
                        print(f"""Transcribe from {start} to {end}""")
                        response = model.generate_content([
                            "",
                            {
                                "mime_type": "audio/mp3",
                                "data": f.readall()
                            }],
                            generation_config=genai.GenerationConfig(response_mime_type="text/plain", temperature=0),
                        )
                    subs = pysrt.from_string(response.text.strip().replace("---", "\n"))
                    subs.shift(milliseconds=start)
                    subs.clean_indexes()
                    if (len(subs) < 5):
                        raise Exception(f"""Too small number of subs returned: {len(subs)}""")
                    for idx, item in enumerate(subs):
                        if len(item.text) > 200:
                            raise Exception(f"""Subtitle item text's too long: {item.text}""")
                        if item.duration > "00:00:10,000":
                            raise Exception(f"""Subtitle item duration's too long: {item.duration}""")
                        if item.duration < "00:00:00,100":
                            raise Exception(f"""Subtitle item duration's too short: {item.duration}""")
                        if idx > 0:
                            gap = max(subs[idx].start - subs[idx-1].start, subs[idx].end - subs[idx-1].end)
                            if (gap > "00:02:00,000"):
                                raise Exception(f"""Gap between items is too far: {gap}""")
                    with mem_fs.open('tmp-srt.srt', 'w', encoding='utf-8') as f:
                        subs.write_into(f)
                except Exception as e:
                    print(f"""Error: {e}""")
                    error += 1
                    if error > 5:
                        raise Exception(f"""Too many errors: {error}""")
                    if "quota" in str(e):
                        print("Sleep for 1 min before re-trying")
                        time.sleep(60)
                    segment_length = random.randint(100000, 200000)
                    print(f"""Reading audio file: {self.input_file}""")
                    print(f"""Try to re-segment audio from {start} with segment length {segment_length}""")
                    segments = self.segment_audio(audio, segment_length, start)
                    print(segments)
                else:
                    error = 0
                    segments.pop(0)
                    with mem_fs.open('tmp-srt.srt', 'r', encoding='utf-8') as f:
                        sub_srt_text = f.read()
                    subs = pysrt.from_string(sub_srt_text)
                    final_subs.extend(subs)
                    final_subs.clean_indexes()
                    with mem_fs.open('tmp-final-srt.srt', 'w', encoding='utf-8') as f:
                        final_subs.write_into(f)
                    with mem_fs.open('tmp-final-srt.srt', 'r', encoding='utf-8') as f:
                        final_sub_srt_text = f.read()

            print(f"""Saving the final subtitles to {self.output_file}.tmp""")
            final_subs.clean_indexes()
            with open(f"""{self.output_file}.tmp""", 'w', encoding='utf-8') as f:
                final_subs.write_into(f)
            self.input_file = f"""{self.output_file}.tmp"""

        if self.extract_srt_from_media:
            try:
                ffmpeg = FFmpeg().input(self.input_file).output(f"""{self.input_file}.srt.tmp""", {"c:s" : "text", "f" : "srt"})
                ffmpeg.execute()
                self.input_file = f"""{self.input_file}.srt.tmp"""
            except Exception as e:
                print(f"""Error: extract srt from {self.input_file}""")
                print(e)
                raise Exception(e)


        instruction = f"""You are a professional translator proficient in all languages, specializing in translating subtitles for film and television.
        Your task is to translate the following subtitles into {self.target_language}.
        Your translation should be faithful to the original meaning and tone while also being culturally appropriate for {self.target_language}-speaking audiences.
        Pay particular attention to the accurate and nuanced translation of humor, idioms, and cultural references.
        The final translation should be suitable for direct use in subtitles.

        You will receive the following JSON type:

class SubtitleObject(typing.TypedDict):
    index: str
    content: str

Request: list[SubtitleObject]
Response: list[SubtitleObject]

The 'index' key is the index of the subtitle dialog.
The 'content' key is the dialog to be translated.

The indices must remain the same in the response as in the request.
Dialogs must be translated as they are without any changes.
"""

        if self.description:
            instruction += "\nAdditional user instruction: '" + self.description + "'"

        print(instruction)
        model = self._get_model(instruction, model_name = self.model_name)
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
            delay_time = 30

            if 'pro' in self.model_name and self.free_quota:
                delay = True
                if not self.gemini_api_key2:
                    print("Pro model and free user quota detected, enabling 30s delay between requests...")
                else:
                    delay_time = 15
                    print("Pro model and free user quota detected, using secondary API key for additional quota...")

            batch.append(SubtitleObject(index=str(i), content=original_subtitle[i].content))
            i += 1

            print(f"Starting translation of {total} lines...")

            if self.gemini_api_key2:
                print(f"Starting with API {self.current_api_number}:")

            while len(batch) > 0:
                if i < total and len(batch) < self.batch_size:
                    batch.append(SubtitleObject(index=str(i), content=original_subtitle[i].content))
                    i += 1
                    continue
                try:
                    start_time = time.time()
                    previous_message = self._process_batch(model, batch, previous_message, translated_subtitle)
                    end_time = time.time()
                    print(self.input_file)
                    print(f"Translated {i}/{total}")
                    if delay and (end_time - start_time < delay_time) and i < total:
                        time.sleep(30 - (end_time - start_time))
                    if reverted > 0:
                        self.batch_size += reverted
                        reverted = 0
                        print("Increasing batch size back to {}...".format(self.batch_size))
                    if i < total and len(batch) < self.batch_size:
                        batch.append(SubtitleObject(index=str(i), content=original_subtitle[i].content))
                        i += 1
                except Exception as e:
                    e_str = str(e)
                    
                    if "quota" in e_str:
                        if self._switch_api():
                            print(f"\n🔄 API {self.backup_api_number} quota exceeded! Switching to API {self.current_api_number}...")
                            model = self._get_model(instruction, model_name=self.model_name)
                        else:
                            print("\nAll API quotas exceeded, waiting 1 minute...")
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
                        if "Gemini" in e_str:
                            print(e_str)
                        else:
                            print("An unexpected error has occurred")
                            print(e_str)
                        print("Decreasing batch size to {} and trying again...".format(self.batch_size))
            
            translated_file.write(srt.compose(translated_subtitle))

    def _switch_api(self) -> bool:
        """
        Switch to the secondary API key if available.

        Returns:
            bool: True if switched successfully, False if no alternative API available
        """
        if self.current_api_number == 1 and self.gemini_api_key2:
            self.current_api_key = self.gemini_api_key2
            self.current_api_number = 2
            self.backup_api_number = 1
            return True
        if self.current_api_number == 2 and self.gemini_api_key:
            self.current_api_key = self.gemini_api_key
            self.current_api_number = 1
            self.backup_api_number = 2
            return True
        return False

    def _get_model(self, instruction: str, model_name: str) -> GenerativeModel:
        """
        Configure and return a Gemini model instance with current API key.

        Args:
            instruction (str): System instruction for the model

        Returns:
            GenerativeModel: Configured Gemini model instance
        """
        genai.configure(api_key=self.current_api_key)
        return genai.GenerativeModel(
            model_name=model_name,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
            system_instruction=instruction,
            generation_config=genai.GenerationConfig(response_mime_type="application/json", temperature=0)
        )

    def _process_batch(self, model: GenerativeModel, batch: list[SubtitleObject], previous_message: ContentDict, translated_subtitle: list[Subtitle]) -> ContentDict:
        """
        Process a batch of subtitles for translation.

        Args:
            model (GenerativeModel): The Gemini model instance
            batch (list[SubtitleObject]): Batch of subtitles to translate
            previous_message (ContentDict): Previous message for context
            translated_subtitle (list[Subtitle]): List to store translated subtitles

        Returns:
            ContentDict: The model's response for context in next batch
        """
        if previous_message:
            messages = [previous_message] + [{"role": "user", "parts": json.dumps(batch, ensure_ascii=False)}]
        else:
            messages = [{"role": "user", "parts": json.dumps(batch, ensure_ascii=False)}]

        response = model.generate_content(messages)
        translated_lines: list[SubtitleObject] = json.loads(response.text)

        if len(translated_lines) != len(batch):
            raise Exception("Gemini has returned the wrong number of lines.")
            
        for line in translated_lines:
            if line["index"] not in [x["index"] for x in batch]:
                raise Exception("Gemini has returned different indices.")
            if self._dominant_strong_direction(line["content"]) == "rtl":
                translated_subtitle[int(line["index"])].content = f"\u202B{line['content']}\u202C"
            else:
                translated_subtitle[int(line["index"])].content = line["content"]
        
        batch.clear()
        return response.candidates[0].content

    def _dominant_strong_direction(self, s: str) -> str:
        """
        Determine the dominant text direction (RTL or LTR) of a string.

        Args:
            s (str): Input string to analyze

        Returns:
            str: 'rtl' if right-to-left is dominant, 'ltr' otherwise
        """
        count = Counter([ud.bidirectional(c) for c in list(s)])
        rtl_count = count['R'] + count['AL'] + count['RLE'] + count["RLI"]
        ltr_count = count['L'] + count['LRE'] + count["LRI"]
        return "rtl" if rtl_count > ltr_count else "ltr"
