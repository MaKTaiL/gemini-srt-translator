from google.genai.types import HarmBlockThreshold, HarmCategory, SafetySetting


def get_instruction(language: str, description: str) -> str:
    """
    Get the instruction for the translation model based on the target language.
    """
    instruction = f"""You are an assistant that translates subtitles.
You will receive a list of objects of the following type:

class SubtitleObject(typing.TypedDict):
    index: str
    content: str

Request: list[SubtitleObject]
Response: list[SubtitleObject]

Translate the content of each object to {language} and return the list.
Keep all the formatting, including line breaks.
Do not under any circumstances merge content from different objects. 
The content of each object must stay true to the original.
"""
    if description:
        instruction += "\nAdditional user instruction: '" + description + "'"
    return instruction


def get_safety_settings() -> list[SafetySetting]:
    """
    Get the safety settings for the translation model.
    """
    return [
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=HarmBlockThreshold.BLOCK_NONE,
        ),
    ]
