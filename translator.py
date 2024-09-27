import srt
import json
import time
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

################################
# Add your own values here
################################

origin_language = "English"
target_language = "Brazilian Portuguese"

origin_file = "Original File.srt"
translated_file = "Translated File.srt"

gemini_key = ""
model_name = "gemini-1.5-flash-latest"

################################
################################

genai.configure(api_key=gemini_key)

instruction = f"""You are an assistant that translates movies/series subtitles from {origin_language} to {target_language}.
Return a json in the same format you received, but with the dialogues translated. Do not remove any tags, if present.
Do not move dialogues or change the numbering of the indices. The number of dialogues must be the same as you received."""

model = genai.GenerativeModel(
    model_name=model_name,
    safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
    system_instruction=instruction,
    generation_config={"response_mime_type": "application/json"},
)

original_file = open(origin_file, "r", encoding="utf-8")
translated_file = open(translated_file, "w", encoding="utf-8")
original_text = original_file.read()

original_subtitle = list(srt.parse(original_text))
translated_subtitle = original_subtitle.copy()

chat = model.start_chat()

i = 0
total = len(original_subtitle)
batch = {}

while i < total:
    if len(batch) < 30:
        batch[str(i)] = original_subtitle[i].content
        i += 1
        continue
    else:
        try:
            response = chat.send_message(json.dumps(batch))
            try:
                translated_lines = json.loads(response.text)
            except Exception as e:
                print(i)
                print(e)
                print(response.text)
                continue
            for x in translated_lines:
                translated_subtitle[int(x)].content = translated_lines[x]
            batch = {}
            batch[i] = original_subtitle[i].content

            print(f"Traduziu {i}/{total}")

            i += 1
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
        except Exception as e:
            print(i)
            print(e)
            print(response.text)
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
