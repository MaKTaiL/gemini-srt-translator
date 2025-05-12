from google.genai.types import HarmBlockThreshold, HarmCategory, SafetySetting


def get_check_instruction(language: str) -> str:
    return f"""User will provide you with two identical lists of subtitles of the following type:

class SubtitleObject(typing.TypedDict):
    index: str
    content: str

You job is to look at each index and check if the content in the first list has a suitable {language} translation in the second.
You don't have to be strict, just check if the content is similar enough to be considered a translation.
Return the number of the first index you think does not have a suitable match. If they all match, return -1.
Do not return any comments or explanations, just the result of the check.
"""


def get_instruction(language: str, description: str) -> str:
    """
    Get the instruction for the translation model based on the target language.
    """
    instruction = f"""You are an assistant that translates subtitles.
You will receive a list of the following type:

class SubtitleObject(typing.TypedDict):
    index: str
    content: str

Request: list[SubtitleObject]
Response: list[SubtitleObject]

Translate the content of each object to {language} and return the same list back.
Keep all the formatting, including line breaks.
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
