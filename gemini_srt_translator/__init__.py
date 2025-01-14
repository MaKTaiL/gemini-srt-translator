from .main import GeminiSRTTranslator
gemini_api_key: str = None
gemini_api_key2: str = None
target_language: str = None
input_file: str = None
output_file: str = None
description: str = None
model_name: str = None
batch_size: int = None
free_quota: bool = None
def listmodels():
    translator = GeminiSRTTranslator(gemini_api_key)
    translator.listmodels()
def translate():
    params = {
        'gemini_api_key': gemini_api_key,
        'gemini_api_key2': gemini_api_key2,
        'target_language': target_language,
        'input_file': input_file,
        'output_file': output_file,
        'description': description,
        'model_name': model_name,
        'batch_size': batch_size,
        'free_quota': free_quota
    }
    filtered_params = {k: v for k, v in params.items() if v is not None}
    translator = GeminiSRTTranslator(**filtered_params)
    translator.translate()
