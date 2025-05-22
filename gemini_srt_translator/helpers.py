from google.genai import types


def get_instruction(language: str, description: str, thinking: bool, thinking_compatible: bool) -> str:
    """
    Get the instruction for the translation model based on the target language.
    """

    thinking_instruction = (
        "\nThink deeply and reason as much as possible before returning the response."
        if thinking
        else "\nDo NOT think or reason."
    )
    instruction = (
        f"You are an assistant that translates subtitles to {language}.\n"
        f"You will receive a list of objects, each with two fields:\n\n"
        f"- index: a string identifier\n"
        f"- content: the subtitle text to translate\n\n"
        f"Translate ONLY the 'content' field of each object.\n"
        f"Keep line breaks, formatting, and special characters.\n"
        f"Do NOT move or merge 'content' between objects.\n"
        f"Do NOT add or remove any objects.\n"
        f"Do NOT make any changes to the 'index' field."
    )

    if thinking_compatible:
        instruction += thinking_instruction

    if description:
        instruction += f"\n\nAdditional user instruction:\n\n{description}"

    return instruction


def get_safety_settings() -> list[types.SafetySetting]:
    """
    Get the safety settings for the translation model.
    """
    return [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
    ]


def get_response_schema() -> types.Schema:
    """
    Get the response schema for the translation model.
    """
    return types.Schema(
        type="ARRAY",
        items=types.Schema(
            type="OBJECT",
            properties={
                "index": types.Schema(type="STRING"),
                "content": types.Schema(type="STRING"),
            },
            required=["index", "content"],
        ),
    )
