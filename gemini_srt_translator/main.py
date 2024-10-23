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

    def translate(self):
        """
        Translates a subtitle file using the Gemini API.
        """
        if not all([self.gemini_api_key, self.target_language, self.input_file]):
            raise Exception("Missing required parameters. Please check api_key, target_language and input_file.")
        
        if not self.output_file:
            self.output_file = ".".join(self.input_file.split(".")[:-1]) + "_translated.srt"

        genai.configure(api_key=self.gemini_api_key)

        model = genai.GenerativeModel(
            model_name=self.model_name,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
            generation_config={"temperature": 0}
        )

        with open(self.input_file, "r", encoding="utf-8") as original_file, open(self.output_file, "w", encoding="utf-8") as translated_file:
            original_text = original_file.read()
            original_subtitle = list(srt.parse(original_text))
            translated_subtitle = original_subtitle.copy()
            
            # Ana çeviri döngüsü
            self._translate_all(model, original_subtitle, translated_subtitle)
            
            translated_file.write(srt.compose(translated_subtitle))

    def _translate_all(self, model, original_subtitle, translated_subtitle):
        """
        Tüm altyazıları çevirir ve sorunları otomatik çözer.
        """
        total = len(original_subtitle)
        current_index = 0
        untranslated = {}  # çevrilmemiş altyazıları takip et
        
        print(f"Starting translation of {total} lines...")
        
        while current_index < total:
            batch = {}
            
            # Batch'i doldur
            while len(batch) < self.batch_size and current_index < total:
                batch[str(current_index)] = original_subtitle[current_index].content
                current_index += 1

            if not batch:
                continue

            # Çeviri dene
            translated_batch = self._translate_batch(model, batch)
            
            # Eksik çevirileri kontrol et
            missing = set(batch.keys()) - set(translated_batch.keys())
            
            # Başarılı çevirileri kaydet
            for idx, text in translated_batch.items():
                translated_subtitle[int(idx)].content = text
            
            # Eksik çevirileri tekil olarak dene
            if missing:
                for idx in missing:
                    single_result = self._translate_single(model, original_subtitle[int(idx)].content, idx)
                    if single_result:  # Başarılı çeviri
                        translated_subtitle[int(idx)].content = single_result
                    else:  # Başarısız çeviri, daha sonra tekrar dene
                        untranslated[idx] = original_subtitle[int(idx)].content
            
            # İlerlemeyi göster
            print(f"Translated {min(current_index, total)}/{total}")

        # Çevrilemeyen altyazıları farklı bir yaklaşımla tekrar dene
        if untranslated:
            print(f"\nRetrying {len(untranslated)} failed translations with alternative method...")
            for idx, content in untranslated.items():
                result = self._translate_with_alternative(model, content)
                if result:
                    translated_subtitle[int(idx)].content = result
                    print(f"Successfully translated line {idx} with alternative method")

    def _translate_batch(self, model, batch):
        """
        Bir batch'i çevirmeyi dener.
        """
        prompt = f"""
TASK: Translate these subtitles to {self.target_language}
INPUT: {json.dumps(batch, ensure_ascii=False)}
IMPORTANT: Return ONLY a JSON object containing all translations with their original indices.
"""
        max_attempts = 3
        current_attempt = 0
        
        while current_attempt < max_attempts:
            try:
                response = model.generate_content(prompt)
                response_text = response.text.strip()
                
                # JSON yanıtı temizle
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].strip()
                
                result = json.loads(response_text)
                return result

            except Exception as e:
                if "quota" in str(e).lower():
                    print("Quota exceeded, waiting 1 minute...")
                    time.sleep(60)
                current_attempt += 1
                if current_attempt == max_attempts:
                    return {}  # Boş sözlük döndür, eksik çeviriler tekil olarak denenecek
        
        return {}

    def _translate_single(self, model, text, idx):
        """
        Tek bir altyazıyı çevirmeyi dener.
        """
        prompt = f"""
TASK: Translate this single subtitle to {self.target_language}
INPUT: "{text}"
IMPORTANT: Return ONLY the translated text, nothing else.
"""
        try:
            response = model.generate_content(prompt)
            return response.text.strip().strip('"').strip("'")
        except Exception as e:
            if "quota" in str(e).lower():
                print("Quota exceeded, waiting 1 minute...")
                time.sleep(60)
            return None

    def _translate_with_alternative(self, model, text):
        """
        Alternatif çeviri yöntemi - son çare olarak kullanılır.
        """
        prompt = f"""
TASK: Simple translation
FROM: "{text}"
TO: {self.target_language}
RETURN: Only translated text
"""
        try:
            response = model.generate_content(prompt)
            return response.text.strip().strip('"').strip("'")
        except:
            return None
