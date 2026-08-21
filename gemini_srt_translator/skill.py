"""
Skill Management Module for Gemini SRT Translator
Handles exporting and installing the Subtitle Translator Skill (SKILL.md)
for AI coding agents (Antigravity, Claude Code, Cursor, Cline, Roo-Code, etc.).
"""

import os
import sys
from typing import List, Optional


def get_skill_path() -> str:
    """Return the absolute path of the packaged SKILL.md."""
    try:
        if sys.version_info >= (3, 9):
            import importlib.resources as pkg_resources

            ref = pkg_resources.files("gemini_srt_translator").joinpath("SKILL.md")
            if ref.is_file():
                return str(ref)
    except Exception:
        pass

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_skill = os.path.join(pkg_dir, "SKILL.md")
    if os.path.exists(pkg_skill):
        return pkg_skill

    repo_skill = os.path.join(pkg_dir, "..", "skills", "subtitle-translator", "SKILL.md")
    if os.path.exists(repo_skill):
        return os.path.abspath(repo_skill)

    return pkg_skill


def get_skill_content() -> str:
    """Read the content of SKILL.md."""
    path = get_skill_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            pass

    return """---
name: subtitle-translator
description: Translate subtitle files (SRT, ASS) and video/media with embedded subtitles into any target language with high linguistic quality, sliding context window, timestamp alignment, and formatting preservation. Use when the user asks to translate a subtitle file (.srt, .ass) or video subtitles.
---

# Subtitle Translator Skill

This skill allows the agent to act as the translation engine while leveraging the mature subtitle processing pipeline of **Gemini SRT Translator**.

## Subtitle Translation Protocol

### 1. Start a Translation Session
```bash
gst agent translate start <INPUT_FILE> -l "<TARGET_LANGUAGE>" [--batch-size N]
```

### 2. Commit Translated Batch
```bash
gst agent translate commit <INPUT_FILE> --data '<TRANSLATED_JSON>'
```

### 3. Status, Next & Reset
```bash
gst agent translate status <INPUT_FILE>
gst agent translate next <INPUT_FILE> -l "<TARGET_LANGUAGE>"
gst agent translate reset <INPUT_FILE>
```
"""


def install_skill(
    target: str = "antigravity",
    is_global: bool = False,
    custom_dir: Optional[str] = None,
    cwd: Optional[str] = None,
) -> List[str]:
    """
    Install SKILL.md for the requested agent platform(s).
    Returns list of paths where SKILL.md was installed.
    """
    content = get_skill_content()
    installed_paths: List[str] = []
    base_dir = os.path.abspath(cwd or os.getcwd())
    home_dir = os.path.expanduser("~")

    raw_targets = [t.strip().lower() for t in target.split(",") if t.strip()]
    if "all" in raw_targets:
        targets = ["antigravity", "claude", "cursor", "agent"]
    else:
        targets = raw_targets or ["antigravity"]

    dest_paths: List[str] = []

    if custom_dir:
        abs_custom = os.path.abspath(custom_dir)
        if abs_custom.lower().endswith("skill.md"):
            dest_paths.append(abs_custom)
        else:
            dest_paths.append(os.path.join(abs_custom, "subtitle-translator", "SKILL.md"))
    else:
        for t in targets:
            if t in ("antigravity", "gemini", "agy"):
                if is_global:
                    dest_paths.append(
                        os.path.join(
                            home_dir,
                            ".gemini",
                            "antigravity",
                            "skills",
                            "subtitle-translator",
                            "SKILL.md",
                        )
                    )
                else:
                    dest_paths.append(
                        os.path.join(
                            base_dir,
                            ".gemini",
                            "skills",
                            "subtitle-translator",
                            "SKILL.md",
                        )
                    )
            elif t in ("claude", "claudecode"):
                if is_global:
                    dest_paths.append(
                        os.path.join(
                            home_dir,
                            ".claude",
                            "skills",
                            "subtitle-translator",
                            "SKILL.md",
                        )
                    )
                else:
                    dest_paths.append(
                        os.path.join(
                            base_dir,
                            ".claude",
                            "skills",
                            "subtitle-translator",
                            "SKILL.md",
                        )
                    )
            elif t in ("cursor", "agent", "agents", "cline", "roo"):
                if is_global:
                    dest_paths.append(
                        os.path.join(
                            home_dir,
                            ".agent",
                            "skills",
                            "subtitle-translator",
                            "SKILL.md",
                        )
                    )
                else:
                    dest_paths.append(
                        os.path.join(
                            base_dir,
                            ".agent",
                            "skills",
                            "subtitle-translator",
                            "SKILL.md",
                        )
                    )

    # De-duplicate paths
    dest_paths = list(dict.fromkeys(dest_paths))

    for dest in dest_paths:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(content)
        installed_paths.append(dest)

    return installed_paths
