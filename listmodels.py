import google.generativeai as genai
import config

gemini_key = config.gemini_key

genai.configure(api_key=gemini_key)

for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name.replace("models/", ""))
