from typing import List, Optional

from google.genai import types

# ==============================================================================
# INSTRUCTION GENERATION FOR TRANSLATION
# ==============================================================================


def get_translate_instruction(
    language: str,
    thinking: bool,
    thinking_compatible: bool,
    audio_file: Optional[str] = None,
    description: Optional[str] = None,
    include_timestamps: bool = False,
) -> str:
    """
    Generates a structured instruction prompt for a subtitle translation task.

    Args:
        language: The target language for the translation (e.g., "French").
        thinking: If True, instructs the model to reason step-by-step.
        thinking_compatible: If True, the "Reasoning Protocol" section is included.
        audio_file: The name of an audio file provided for context.
        description: Additional user-provided instructions for context.

    Returns:
        The complete, formatted instruction string for the AI translation task.
    """
    prompt_parts = []
    section_number = 1

    # --- Section 1: Persona and Primary Goal ---
    prompt_parts.append(
        f"# INSTRUCTION: Translate Subtitles to {language}\n\n"
        "You are an expert AI linguist specializing in subtitle translation. Your goal is to translate the `text` field of each item into "
        f"**{language}**."
    )

    # --- Section 2: Input/Output Structure ---
    section_number += 1
    audio_structure = '"time_start": "01:56",\n    "time_end": "02:00",'
    json_structure = f"""
```json
[
  {{
    "index": "1",
    "text": "This is the first subtitle line.\\nThis is the second line.",
{'    ' + audio_structure if audio_file or include_timestamps else '...'}
  }}
]
```"""
    prompt_parts.append(
        f"## {section_number}. Data Structure\nYou will receive and must return a list of items with this exact structure:\n{json_structure}"
    )

    # --- Section 3: Core Translation Rules ---
    section_number += 1
    core_rules = """
## {section_number}. Core Translation Rules
- **Translate Text Only**: Only translate the value of the `text` field.
- **Preserve Formatting**: Keep all existing formatting, including HTML tags (`<i>`, `<b>`) and line breaks (`\\n`).
- **Handle Empty Text**: If a `text` field is empty or contains only whitespace, keep it unchanged.
- **Maintain Integrity**:
  - Number of items in the output must match the input.
  - **Do NOT** alter any fields other than `text`.
  - **Do NOT** add, remove, or reorder any items on the list.
  - **Do NOT** merge text between different items. Original and translation must match.
""".format(
        section_number=section_number, language=language
    )
    prompt_parts.append(core_rules)

    # --- Section 4: Advanced Rules (Conditional) ---
    if audio_file:
        section_number += 1
        audio_rules = f"""
## {section_number}. Advanced Rules: Grammatical Gender from Audio
If the text is ambiguous, use the provided audio to infer the correct grammatical gender and formality.
- **Analyze Speaker's Voice**: Use the `time_start` and `time_end` (`MM:SS` format) to listen to the speaker.
  - If the speaker sounds **male**, use masculine grammatical forms.
  - If the speaker sounds **female**, use feminine forms.
  - **Example (French):** "I am tired." -> `Je suis fatigué` (male) vs. `Je suis fatiguée` (female).
- **Analyze Context**: If necessary, listen to the audio before or after the current segment to determine who is being addressed and apply the correct gender and pluralization.
  - **Example (Spanish):** "You all are tired." -> `Ustedes están cansados` (male/mixed) vs. `Ustedes están cansadas` (female).
"""
        prompt_parts.append(audio_rules)

    # --- Section 5: User Context (Conditional) ---
    if description:
        section_number += 1
        user_description_section = f"""
## {section_number}. Additional User-Provided Context
Use this context to improve translation accuracy. These notes do not override core rules.
- {description}
"""
        prompt_parts.append(user_description_section)

    # --- Section 6: Reasoning Protocol (Conditional) ---
    if thinking_compatible:
        section_number += 1
        reasoning_instruction = (
            "**Think step-by-step**: Before generating the final JSON, reason through your choices to ensure all rules are strictly followed."
            if thinking
            else "**Respond directly**: Generate the JSON output immediately without providing a step-by-step reasoning process."
        )
        reasoning_protocol_section = f"""
## {section_number}. Reasoning Protocol
{reasoning_instruction}"""
        prompt_parts.append(reasoning_protocol_section)

    # --- Section 6: include timestamps ---
    if include_timestamps and description and not audio_file:
        section_number += 1
        timestamp_context_section = f"""
        ## {section_number}. Using Timestamps for Context Matching
        Each subtitle includes `time_start` and `time_end` fields (in `MM:SS` format). Use these timestamps to match each subtitle with the transcription/context provided below.
        - **Identify speakers**: Determine WHO is speaking and TO WHOM based on the context at that timestamp.
        - **Apply correct gender**: Use appropriate grammatical gender for verbs and adjectives based on the speaker's gender (e.g., Polish: "zrobiłem" for male vs "zrobiłam" for female).
        - **Scene context**: Understand the situation, location, and emotional tone of the scene to choose appropriate vocabulary and register.
        """
        prompt_parts.append(timestamp_context_section)

    return "\n---\n".join(prompt_parts).strip()


# ==============================================================================
# INSTRUCTION GENERATION FOR TRANSCRIPTION
# ==============================================================================


def get_transcribe_instruction(thinking: bool, thinking_compatible: bool, description: Optional[str] = None) -> str:
    """
    Generates a complete instruction prompt for an AI audio transcription task.

    Args:
        thinking: If True, instructs the model to reason step-by-step.
        thinking_compatible: If True, the "Reasoning Protocol" section is included.
        description: Additional user-provided instructions for context.

    Returns:
        The complete, formatted instruction string for the AI transcription task.
    """
    prompt_parts = []
    section_number = 0

    # --- Section 1: Persona and Primary Goal ---
    prompt_parts.append(
        "# INSTRUCTION: Transcribe Audio to Subtitles\n\n"
        "You are an expert AI transcriber specializing in broadcast-quality subtitles. Your goal is to transcribe audio into a timed JSON array of subtitle objects."
    )

    # --- Section 2: Output Structure ---
    section_number += 1
    output_structure = f"""
## {section_number}. Output Structure
The output must be a JSON array. Each object must contain `text`, `time_start`, and `time_end`.
```json
{{
  "text": "This is the first line of the subtitle.\\nThis is the second line.",
  "time_start": "00:06",
  "time_end": "00:09"
}}
```"""
    prompt_parts.append(output_structure)

    # --- Section 3: Core Rules ---
    section_number += 1
    core_rules = f"""
## {section_number}. Core Formatting Rules
- **Timing**: Subtitles must be between **1-7 seconds** in duration.
- **Content**: Max **42 characters** per line, max **2 lines** per subtitle (use `\\n`).
- **Line Breaking**: Break lines logically (e.g., after punctuation, before conjunctions). Do not separate closely related words (e.g., article and noun).
- **Omit Non-Verbal Sounds**: Do not transcribe sounds like `[coughs]` or `[laughs]`. Use italics for song lyrics as described below."""
    prompt_parts.append(core_rules)

    # --- Section 4: Dialogue and Continuity ---
    section_number += 1
    dialogue_rules = f"""
## {section_number}. Dialogue and Continuity
- **Multiple Speakers**: Use a hyphen: `- Are you coming?\\n- In a minute.`
- **Interruptions**: Use two hyphens: `- What are you--\\n- Be quiet!`
- **Significant Pauses (2+ sec)**: End the preceding subtitle with an ellipsis (`...`).
- **Starting Mid-Sentence**: Begin the subtitle with an ellipsis (`...like this.`)."""
    prompt_parts.append(dialogue_rules)

    # --- Section 5: Text and Number Formatting ---
    section_number += 1
    formatting_rules = f"""
## {section_number}. Text and Number Formatting
- **Numbers**: Spell out numbers zero through ten; use digits for 11 and up. Always spell out a number that begins a sentence.
- **Dates & Times**: Transcribe as spoken but omit "the" and "of" (e.g., `March 6th`). Use numerals for specific times (`9:30 a.m.`).
- **Quotes**: Use double quotes (`"`) for dialogue and single quotes (`'`) for quotes within quotes. Punctuation goes *inside* the closing quote."""
    prompt_parts.append(formatting_rules)

    # --- Section 6: Italics and Songs ---
    section_number += 1
    italics_rules = f"""
## {section_number}. Italics and Songs
Use `<i></i>` tags **only** for:
- Narration, off-screen voices, or inner thoughts.
- Dialogue from electronic media (phones, TVs).
- Sung lyrics.
- Titles of books, movies, etc.
- Unfamiliar foreign words.
- Clear vocal emphasis (`It <i>was</i> you.`).

For songs, italicize lyrics and wrap the subtitle with music notes: `♪ <i>lyrics go here</i> ♪`."""
    prompt_parts.append(italics_rules)

    # --- Section 7: User Context (Conditional) ---
    if description:
        section_number += 1
        user_description_section = f"""
## {section_number}. Additional User-Provided Context
Use this context to improve transcription accuracy. These notes do not override core rules.
- {description}
"""
        prompt_parts.append(user_description_section)

    # --- Section 8: Reasoning Protocol (Conditional) ---
    if thinking_compatible:
        section_number += 1
        reasoning_instruction = (
            "**Think step-by-step**: Before generating the final JSON, reason through your choices to ensure all rules are strictly followed."
            if thinking
            else "**Respond directly**: Generate the JSON output immediately without providing a step-by-step reasoning process."
        )
        reasoning_protocol_section = f"""
## {section_number}. Reasoning Protocol
{reasoning_instruction}"""
        prompt_parts.append(reasoning_protocol_section)

    return "\n---\n".join(prompt_parts).strip()


# ==============================================================================
# API CONFIGURATION
# ==============================================================================


def get_safety_settings() -> List[types.SafetySetting]:
    """
    Configures the safety settings for the generative model.

    This function disables all safety filters (Hate Speech, Dangerous Content, etc.).
    This is necessary for tasks like subtitle translation or transcription of media,
    which may contain sensitive content that should be faithfully processed, not blocked.
    Use with caution and ensure compliance with content policies.

    Returns:
        A list of SafetySetting objects with all thresholds set to BLOCK_NONE.
    """
    return [
        types.SafetySetting(
            category=category,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        )
        for category in [
            types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        ]
    ]


def get_translate_response_schema() -> types.Schema:
    """
    Defines the expected JSON schema for the translation model's response.
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
    Defines the expected JSON schema for the transcription model's response.
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


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    print("----------- TRANSLATE INSTRUCTION (THINKING ENABLED) -----------")
    translate_instruction = get_translate_instruction(
        language="French",
        description="The setting is a formal business meeting. Please use formal address.",
        thinking=True,
        thinking_compatible=True,
        audio_file="meeting_audio.mp3",
    )
    print(translate_instruction)

    print("\n\n----------- TRANSCRIBE INSTRUCTION (THINKING DISABLED) -----------")
    transcribe_instruction = get_transcribe_instruction(
        thinking=False,
        thinking_compatible=True,
        description="The main speaker has a slight lisp.",
    )
    print(transcribe_instruction)

    print("\n\n----------- TRANSCRIBE INSTRUCTION (REASONING NOT COMPATIBLE) -----------")
    transcribe_instruction_no_reason = get_transcribe_instruction(
        thinking=True,  # This argument is ignored
        thinking_compatible=False,
    )
    print(transcribe_instruction_no_reason)
