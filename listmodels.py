import google.generativeai as genai

gemini_key = ""

genai.configure(api_key=gemini_key)

for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)
