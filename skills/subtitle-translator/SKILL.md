---
name: subtitle-translator
description: Translate subtitle files (SRT, ASS) and video/media with embedded subtitles into any target language with high linguistic quality, sliding context window, timestamp alignment, and formatting preservation. Use when the user asks to translate a subtitle file (.srt, .ass) or video subtitles.
---

# Subtitle Translator Skill

This skill allows the agent to act as the translation engine while leveraging the mature subtitle processing pipeline of **Gemini SRT Translator** (subtitle parsing, line counting, timestamp slicing, sliding context window, timestamp preservation, JSON repair, progress tracking, and atomic file saving).

## When to Use

- When the user asks to translate a subtitle file (`.srt` or `.ass`) into another language.
- When translating multi-part dialogue requiring context awareness and consistent character tone/gender agreement.
- When working with video files (`.mp4`, `.mkv`, etc.) that contain extractable subtitle streams.

---

## Subtitle Translation Protocol

### 1. Start a Translation Session

```bash
gst agent translate start <INPUT_FILE> -l "<TARGET_LANGUAGE>" [--batch-size N] [--description "<OPTIONAL_CONTEXT>"]
```

> **Optimal Batch Size Guidance (`-b` / `--batch-size` is optional, defaults to 100):**
> As an agent, select the batch size you find most optimal for your model capabilities and the file length:
>
> - **Recommended Default (80–120 lines):** Optimal balance between narrative context, translation accuracy, and fast validation.
> - **High-Capacity Models (Claude 3.7 / GPT-4o / Gemini Pro & Flash):** Feel free to use **100–150 lines** to translate full scenes in fewer turns.
> - **Short Files / Anime Episodes (< 300 lines):** 60–80 lines provides 3–4 quick, responsive turns.
> - **Constrained Output / Local Models:** Use **40–60 lines** to guarantee the full JSON array fits comfortably within the model's output generation limits.

### 2. Commit Translated Batch

```bash
gst agent translate commit <INPUT_FILE> --data '<TRANSLATED_JSON>'
# or from file
gst agent translate commit <INPUT_FILE> --data-file batch_1_translated.json
```

**Commit Data Format:**

```json
[
  { "index": "0", "text": "Bonjour le monde !" },
  { "index": "1", "text": "Comment vas-tu aujourd'hui ?" }
]
```

### 3. Status, Next & Reset

```bash
gst agent translate status <INPUT_FILE>
gst agent translate next <INPUT_FILE> -l "<TARGET_LANGUAGE>"
gst agent translate reset <INPUT_FILE>
```

---

## Translation Rules

1. **Translation Item Parity:** The output JSON array must contain the exact same number of items with identical indices (`index`).
2. **Formatting Preservation:** Preserve all newlines (`\n`), italic tags (`<i>...</i>`), and ASS styling tags (`{\an8}`, `{\pos(...)}`, etc.).
3. **Punctuation & Tone:** Maintain dialogue flow, character voice, and natural target language phrasing without altering timing/structural markers.

---

## Python Programmatic API

```python
from gemini_srt_translator import SubtitleSession

session = SubtitleSession(input_file="input.srt", target_language="Spanish", batch_size=100)
while not session.is_complete():
    batch = session.get_next_batch()
    if not batch:
        break
    # Translate batch["batch"] with your agent / LLM
    # translated = ...
    session.commit_batch(translated)
```
