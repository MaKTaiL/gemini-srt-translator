from google.genai import types


def get_translate_instruction(
    language: str, thinking: bool, thinking_compatible: bool, audio_file: str = None, description: str = None
) -> str:
    """
    Generates a clear and structured instruction prompt for the translation model.

    Args:
        language: The target language for translation.
        thinking: If True, instructs the model to reason before answering.
        thinking_compatible: If True, allows the thinking instruction to be added.
        audio_file: The path to an audio file, if available for context.
        description: Additional user-provided instructions.

    Returns:
        The complete instruction string for the AI translation task.
    """

    audio_structure = """"time_start": "01:56",
    "time_end": "02:00","""

    json_structure = f"""
```json
[
  {{
    "index": "1",
    "text": "This is the first subtitle line.\\nThis is the second line.",
    {audio_structure if audio_file else ""}
  }}
]
```"""

    base_instruction = f"""
# AI Subtitle Translation Task

Your primary goal is to translate the `text` of subtitle objects from their original language into **{language}**. You must preserve the structure and formatting of the input data precisely.

---
## 1. Input Structure

You will receive a JSON array of objects with the following structure:
{json_structure}

---
## 2. Core Translation Rules

These rules must be followed in all translations.

- **Translate Text Only**: Only translate the text inside the `text` field.
- **Preserve Formatting**: Keep all existing formatting, including HTML tags (like `<i>` or `<b>`) and line breaks (`\\n`).
- **Preserve punctuation**: Maintain the original punctuation and capitalization.
- **Handle Empty Text**: If a `text` field is empty or contains only whitespace, leave it as is.
- **Maintain Integrity**: 
  - **Do NOT** merge or move text from one object's `text` field to another.
  - **Do NOT** add, remove, or reorder any objects.
  - **Do NOT** alter any other values other than the `text` field.
"""

    audio_rules = f"""
---
## 3. Advanced Rules: Grammatical Gender

You can use the audio provided to apply correct grammatical gender and formality when gender can't be inferred from the text alone.
Do not add any information about this to the `text` field. Audio should only be used to determine the correct grammatical forms when necessary.

- **Analyze the Speaker's Voice**: Use the `time_start` and `time_end` of each segment to listen to the speaker in the audio.
  - Time will be provided in the format `MM:SS` (minutes:seconds).
  - If the speaker sounds **male**, use masculine forms for adjectives, pronouns, and verbs.
  - If the speaker sounds **female**, use feminine forms.
  - **Example (French):** "I am tired." -> `Je suis fatigué` (male) vs. `Je suis fatiguée` (female).

- **Analyze the Context**: When necessary, infer who the speaker is addressing.
  - You might need to analyze audio cues from before and/or after the current `time_start` and `time_end` segment to determine the context.
  - If addressing a **male**, use masculine forms.
  - If addressing a **female**, use feminine forms.
  - If addressing a **group**, use plural forms (and the correct group gender, if applicable).
  - **Example (Spanish):** "You all are tired." -> `Ustedes están cansados` (male/mixed group) vs. `Ustedes están cansadas` (female group).
"""

    # --- Section 3: Final Assembly ---
    # Start with the base instructions.
    instruction = base_instruction

    # Add audio rules if an audio file is present.
    if audio_file:
        instruction += audio_rules

    if thinking_compatible:
        thinking_instruction = (
            "\n**Think deeply and reason as much as possible before returning the response.**\n"
            if thinking
            else "\n**Do NOT think or reason.**\n"
        )
        instruction += thinking_instruction

    user_description_section = f"""
---
## 4. Additional User-Provided Context

Please use the following context to improve the accuracy and nuance of the translation. These notes do not override the core rules above:

{description}
"""
    if description:
        instruction += user_description_section

    # Return the final, assembled prompt, removing any leading/trailing whitespace.
    return instruction.strip()


def get_transcribe_instruction(thinking: bool, thinking_compatible: bool, description: str = None) -> str:
    """
    Generates and returns the complete instruction string for the AI subtitler,
    with optional instructions for reasoning.

    Args:
        thinking: If True, instructs the model to reason before answering.
        thinking_compatible: If True, allows the thinking instruction to be added.

    Returns:
        The complete instruction string.
    """

    # Define the required fields for the output JSON object. This is kept separate for clarity.
    output_fields = (
        "- `text`: (String) The transcribed text for the segment.\n"
        "- `time_start`: (String) The start time of the subtitle in `MM:SS` format.\n"
        "- `time_end`: (String) The end time of the subtitle in `MM:SS` format."
    )

    # Define the conditional instruction for the model's reasoning process.
    thinking_instruction = (
        "\nThink deeply and reason as much as possible before returning the response."
        if thinking
        else "\nDo NOT think or reason."
    )

    # The main instruction prompt, formatted using an f-string.
    instruction_prompt = f"""
# AI Subtitle Generation Task

Your primary goal is to transcribe audio into timed subtitles. You must return a JSON array of objects, where each object represents a single subtitle entry.

---
## 1. Output Structure

The output must be a JSON array. Each object in the array must contain these four fields:

{output_fields}

**Example Object:**
```json
{{
  "text": "This is the first line of the subtitle.\\nThis is the second line.",
  "time_start": "00:06",
  "time_end": "00:09",
}}
```

---
## 2. Core Formatting Rules

### Timing & Duration
- **Minimum Duration**: Each subtitle must be at least **1 second** long.
- **Maximum Duration**: Each subtitle must be no more than **7 seconds** long.

### Content and Line Breaks
- **Character Limit**: Maximum of **42 characters** per line.
- **Line Limit**: Maximum of **2 lines** per subtitle. Use a newline character (`\\n`) to separate lines.
- **Prefer Two Lines**: When possible, format subtitles to use two lines instead of one.
- **Line Breaking Logic**:
  - **Break lines** after punctuation (commas, periods) or before conjunctions (and, but) and prepositions (to, from, in).
  - **Do NOT break lines** if it separates closely related words, such as:
    - An article from its noun (`a car`, not `a\\ncar`).
    - A noun from its adjective (`a red\\ncar`).
    - A first name from a last name (`John\\nDoe`).
    - A verb from its subject pronoun (`He\\nis`).
    - A verb from its auxiliary or negation (`I would\\nnot`).

---
## 3. Dialogue and Continuity

- **Multiple Speakers**: If two people speak within one subtitle, use a hyphen.
  - `- Are you coming?\\n- In a minute.`
- **Interruptions**: Indicate an abrupt cutoff with two hyphens (`--`).
  - `- What are you--\\n- Be quiet!`
- **Continuous Sentences (No Pause)**: If a sentence continues across subtitles without a significant pause (< 2 seconds), do not use any special punctuation.
  - **Subtitle 1:** `I always knew`
  - **Subtitle 2:** `that you would agree with me.`
- **Significant Pauses (2+ seconds)**: If a sentence is interrupted by a long pause, end the first part with an ellipsis (`...`). Do not start the next subtitle with an ellipsis.
  - **Subtitle 1:** `Had I known...`
  - **(Pause in audio)**
  - **Subtitle 2:** `I wouldn’t have called you.`
- **Starting Mid-Sentence**: If a subtitle begins in the middle of a sentence (e.g., the audio is joined in progress), start it with an ellipsis without a preceding space (`...have signed an agreement.`).

---
## 4. Text and Number Formatting

### Numbers
- **0-10**: Spell out (zero, one, ten).
- **11 and above**: Use digits (11, 42, 101).
- **Sentence Start**: Always spell out a number if it begins a sentence.
- **Decades**: Use digits (`1950s`, `'50s`).
- **Centuries**: Use digits with suffixes (`20th century`).
- **Ages**: Spell out decades (`He's in his fifties`).

### Dates & Times
- **Dates**: Transcribe as spoken, but omit "the" and "of" (e.g., `March 6th` or `6th March`).
- **Time of Day**: Use numerals for specific times (`9:30 a.m.`, `eleven o'clock`).

### Quotes
- Use double quotes (`"`) for dialogue and single quotes (`'`) for quotes within quotes.
- Punctuation like periods and commas go *inside* the closing quote.
  - `He told me, "Come back tomorrow."`
  - `He said, "'Singing in the Rain' is my favorite song."`

---
## 5. Italics and Songs

### When to Use Italics
Wrap text in `<i></i>` tags **only** for the following:
- **Narration**, off-screen voices, or inner thoughts.
- **Dialogue from electronic media** (phones, TVs, AI assistants).
- **Sung lyrics**.
- **Titles** of books, movies, albums, etc.
- **Unfamiliar foreign words**.
- **Vocal emphasis** that cannot be conveyed otherwise (`It <i>was</i> you.`).

### Formatting Songs
- Italicize all lyrics using `<i></i>`.
- Enclose each subtitle containing lyrics with a music note and a space (`♪ <i>lyrics go here</i> ♪`).
- If the song title is known and there is space, identify it in square brackets: `["Forever Your Girl" playing]`
"""

    # Append the thinking instruction if the model is compatible.
    if thinking_compatible:
        instruction_prompt += thinking_instruction

    user_description_section = f"""
---
## 4. Additional User-Provided Context

Please use the following context to improve the accuracy and nuance of the transcription. These notes do not override the core rules above:

{description}
"""
    if description:
        instruction += user_description_section

    return instruction_prompt.strip()


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


def get_translate_response_schema() -> types.Schema:
    """
    Get the response schema for the translation model.
    """
    return types.Schema(
        type="ARRAY",
        items=types.Schema(
            type="OBJECT",
            properties={
                "index": types.Schema(type="STRING"),
                "text": types.Schema(type="STRING"),
            },
            required=["index", "text"],
        ),
    )


def get_transcribe_response_schema() -> types.Schema:
    """
    Get the response schema for the transcription model.
    """
    return types.Schema(
        type="ARRAY",
        items=types.Schema(
            type="OBJECT",
            properties={
                "text": types.Schema(type="STRING"),
                "time_start": types.Schema(type="STRING"),
                "time_end": types.Schema(type="STRING"),
            },
            required=["text", "time_start", "time_end"],
        ),
    )


if __name__ == "__main__":
    ### Example usage of the get_instruction function

    # result = get_translate_instruction(
    #     language="French",
    #     description="Translate the subtitles accurately.",
    #     thinking=True,
    #     audio_file="example_audio.mp3",
    #     thinking_compatible=True,
    # )
    # print(result)

    result = get_transcribe_instruction(thinking=True, thinking_compatible=True)
    print(result)
