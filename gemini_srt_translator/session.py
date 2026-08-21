import json
import os
import stat
import tempfile
import unicodedata as ud
from collections import Counter
from datetime import timedelta
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union

import json_repair
import pysubs2

from .ffmpeg_utils import (
    check_ffmpeg_installation,
    extract_audio_from_video,
    extract_subtitle_from_video,
)
from .helpers import get_transcribe_instruction, get_translate_instruction
from .logger import info, warning
from .utils import convert_timedelta_to_timestamp, convert_timestamp_to_timedelta


class Subtitle:
    def __init__(self, index: int, start: timedelta, end: timedelta, content: str):
        self.index = index
        self.start = start
        self.end = end
        self.content = content


class SubtitleObject(TypedDict, total=False):
    index: str
    text: str
    time_start: Optional[str]
    time_end: Optional[str]


class BatchPayload(TypedDict, total=False):
    batch_number: int
    total_batches: int
    start_line: int
    end_line: int
    total_lines: int
    progress_percent: float
    system_prompt: str
    batch: List[SubtitleObject]
    context: List[SubtitleObject]
    audio_chunk_path: Optional[str]
    audio_bytes: Optional[bytes]
    is_complete: bool


class TranscribeItem(TypedDict, total=False):
    text: str
    time_start: str
    time_end: str


class TranscribeChunkPayload(TypedDict, total=False):
    chunk_number: int
    total_chunks: int
    start_seconds: int
    end_seconds: int
    total_seconds: int
    progress_percent: float
    system_prompt: str
    audio_chunk_path: Optional[str]
    audio_bytes: Optional[bytes]
    is_complete: bool


class SubtitleSession:
    """
    Core subtitle translation session engine.
    Manages parsing, batching, sliding context window, validation, atomic saving,
    and progress tracking independently from the LLM translation backend.
    """

    def __init__(
        self,
        input_file: Optional[str] = None,
        target_language: str = "English",
        output_file: Optional[str] = None,
        video_file: Optional[str] = None,
        audio_file: Optional[str] = None,
        batch_size: int = 100,
        start_line: Optional[int] = None,
        resume_context_size: int = 20,
        description: Optional[str] = None,
        audio_chunk_size: int = 300,
        extract_audio: bool = False,
        isolate_voice: bool = True,
        resume: Optional[bool] = None,
        thinking: bool = False,
    ):
        self.target_language = target_language
        self.video_file = video_file
        self.input_file = input_file
        self.audio_file = audio_file
        self.batch_size = max(1, batch_size)
        self.resume_context_size = max(0, int(resume_context_size or 0))
        self.description = description
        self.audio_chunk_size = audio_chunk_size
        self.extract_audio = extract_audio
        self.isolate_voice = isolate_voice
        self.resume = resume
        self.thinking = thinking

        # Check if input_file is actually a video file
        video_extensions = (".mkv", ".mp4", ".avi", ".mov", ".flv", ".wmv", ".webm", ".m4v", ".ts")
        if self.input_file and self.input_file.lower().endswith(video_extensions):
            if not self.video_file:
                self.video_file = self.input_file
            self.input_file = None

        # Video / Audio extraction preprocessing
        self.subtitle_extracted = False
        self.audio_extracted = False
        self._prepare_media_inputs()

        if not self.input_file or not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Subtitle or video input file not found: {self.input_file}")

        # Resolve output file and progress file (prefer video_file base name without _extracted)
        base_file = self.video_file or self.input_file or self.audio_file
        base_name = os.path.splitext(os.path.basename(base_file))[0] if base_file else "subtitle"
        if base_name.endswith("_extracted"):
            base_name = base_name[:-10]
        dir_path = os.path.dirname(base_file) if base_file else ""

        if output_file:
            self.output_file = output_file
        else:
            ext = os.path.splitext(self.input_file)[1].lower()
            suffix = "_translated.ass" if ext == ".ass" else "_translated.srt"
            self.output_file = os.path.join(dir_path, f"{base_name}{suffix}") if dir_path else f"{base_name}{suffix}"

        self.progress_file = os.path.join(dir_path, f"{base_name}.progress") if dir_path else f"{base_name}.progress"

        # Audio instance for slicing
        self.audio = None
        self._init_audio()

        # Parse original subtitles
        self.original_subtitles: List[Subtitle] = self._parse_subtitle_file(self.input_file)
        self.total_lines = len(self.original_subtitles)

        # Load or initialize translated subtitles
        self.translated_subtitles: List[Subtitle] = []
        self.current_line = 1  # 1-indexed
        self.batch_number = 1
        self._init_translation_state(start_line)

    def _prepare_media_inputs(self):
        """Extract subtitle or audio from video if needed."""
        ffmpeg_ok = check_ffmpeg_installation()

        if self.video_file and not self.input_file:
            if not ffmpeg_ok:
                raise RuntimeError("FFmpeg is required to extract subtitles from video.")
            self.input_file = extract_subtitle_from_video(self.video_file)
            self.subtitle_extracted = True
        elif self.input_file and (
            self.input_file.endswith("_extracted.ass") or self.input_file.endswith("_extracted.srt")
        ):
            if self.video_file:
                self.subtitle_extracted = True

        if self.video_file and self.extract_audio and not self.audio_file:
            if not ffmpeg_ok:
                raise RuntimeError("FFmpeg is required to extract audio from video.")
            self.audio_file = extract_audio_from_video(self.video_file, isolate_voice=self.isolate_voice)
            self.audio_extracted = True
        elif self.audio_file and (
            self.audio_file.endswith("_extracted.mp3") or self.audio_file.endswith("_isolated_voice.mp3")
        ):
            if self.video_file:
                self.audio_extracted = True

    def _init_audio(self):
        """Initialize pydub AudioSegment if audio file is available."""
        if self.audio_file and os.path.exists(self.audio_file):
            try:
                from pydub import AudioSegment

                self.audio = AudioSegment.from_file(self.audio_file)
            except Exception as e:
                warning(f"Could not load audio for slicing: {e}")
                self.audio = None

    def _init_translation_state(self, explicit_start_line: Optional[int]):
        """Initialize translated subtitles and resume pointer."""
        saved_line = self._read_saved_progress()

        if explicit_start_line is not None:
            self.current_line = max(1, min(explicit_start_line, self.total_lines))
        elif saved_line is not None and (self.resume is True or self.resume is None):
            self.current_line = max(1, min(saved_line, self.total_lines))

        # Check existing translated output file
        if os.path.exists(self.output_file):
            try:
                existing = self._parse_subtitle_file(self.output_file)
                if len(existing) == self.total_lines:
                    self.translated_subtitles = existing
                else:
                    self.translated_subtitles = self._copy_subtitles(self.original_subtitles)
            except Exception:
                self.translated_subtitles = self._copy_subtitles(self.original_subtitles)
        else:
            self.translated_subtitles = self._copy_subtitles(self.original_subtitles)

        # Update batch number estimate
        if self.current_line > 1:
            self.batch_number = ((self.current_line - 1) // self.batch_size) + 1

    def _copy_subtitles(self, subs: List[Subtitle]) -> List[Subtitle]:
        return [Subtitle(s.index, s.start, s.end, s.content) for s in subs]

    def _read_saved_progress(self) -> Optional[int]:
        """Read saved line from .progress file."""
        if not self.progress_file or not os.path.exists(self.progress_file):
            return None
        try:
            with open(self.progress_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("input_file") == self.input_file:
                return data.get("line")
        except Exception:
            pass
        return None

    def _save_progress(self, line: int):
        """Save progress atomically."""
        if not self.progress_file:
            return
        try:
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump({"line": line, "input_file": self.input_file}, f)
        except Exception as e:
            warning(f"Failed to save progress: {e}")

    @staticmethod
    def _parse_subtitle_file(file_path: str) -> List[Subtitle]:
        """Parse subtitle file using pysubs2."""
        subs = pysubs2.load(file_path, encoding="utf-8", keep_html_tags=True)
        result = []
        for i, ev in enumerate(subs):
            text = f"{ev.text.replace(chr(92) + 'N', chr(10))}"
            result.append(
                Subtitle(
                    index=i + 1,
                    start=timedelta(milliseconds=ev.start),
                    end=timedelta(milliseconds=ev.end),
                    content=text,
                )
            )
        return result

    @staticmethod
    def _save_subtitle_file(input_file: str, translated_subtitle: List[Subtitle], output_file: str):
        """Save translated subtitles to destination file using atomic replace."""
        subs = pysubs2.load(input_file, encoding="utf-8")
        for sub in translated_subtitle:
            idx = sub.index - 1
            if 0 <= idx < len(subs):
                subs[idx].text = f"{sub.content.replace(chr(10), chr(92) + 'N')}"

        temp_dir = os.path.dirname(os.path.abspath(output_file)) or "."
        temp_out = os.path.join(temp_dir, f".tmp_{os.path.basename(output_file)}")
        subs.save(temp_out, encoding="utf-8")
        os.replace(temp_out, output_file)

    @staticmethod
    def _dominant_strong_direction(s: str) -> str:
        count = Counter([ud.bidirectional(c) for c in list(s)])
        rtl_count = count["R"] + count["AL"] + count["RLE"] + count["RLI"]
        ltr_count = count["L"] + count["LRE"] + count["LRI"]
        return "rtl" if rtl_count > ltr_count else "ltr"

    @staticmethod
    def _flatten_repaired_json(data: Any) -> List[dict]:
        result = []
        if not isinstance(data, list):
            return result
        for item in data:
            if isinstance(item, dict):
                result.append(item)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, dict):
                        result.append(sub)
        return result

    def is_complete(self) -> bool:
        """Check if all lines have been translated."""
        return self.current_line > self.total_lines

    def get_status(self) -> Dict[str, Any]:
        """Return current session status and progress percentage."""
        completed_lines = min(self.total_lines, max(0, self.current_line - 1))
        percent = (completed_lines / self.total_lines * 100.0) if self.total_lines > 0 else 100.0
        return {
            "input_file": self.input_file,
            "output_file": self.output_file,
            "target_language": self.target_language,
            "total_lines": self.total_lines,
            "completed_lines": completed_lines,
            "current_line": self.current_line,
            "remaining_lines": max(0, self.total_lines - completed_lines),
            "progress_percent": round(percent, 2),
            "is_complete": self.is_complete(),
            "batch_number": self.batch_number,
        }

    def get_next_batch(self, batch_size: Optional[int] = None) -> Optional[BatchPayload]:
        """
        Generate the next batch payload containing batch items, sliding context,
        instruction prompt, and optional audio slice.
        """
        if self.is_complete():
            return None

        effective_batch_size = batch_size or self.batch_size
        start_idx = self.current_line - 1
        end_idx = min(self.total_lines, start_idx + effective_batch_size)

        batch_items: List[SubtitleObject] = []
        audio_chunk_path = None
        audio_bytes = None

        offset = self.original_subtitles[start_idx].start.seconds if start_idx < self.total_lines else 0
        offset_end = offset

        for idx in range(start_idx, end_idx):
            sub = self.original_subtitles[idx]
            if self.audio_file and batch_items and (sub.end.seconds - offset > self.audio_chunk_size):
                break

            item: SubtitleObject = {
                "index": str(idx),
                "text": sub.content,
            }
            if self.audio_file:
                item["time_start"] = convert_timedelta_to_timestamp(sub.start, offset=offset)
                item["time_end"] = convert_timedelta_to_timestamp(sub.end, offset=offset)
                offset_end = sub.end.seconds
            batch_items.append(item)

        # Slice audio chunk if audio is loaded
        if self.audio and self.audio_file and offset_end > offset:
            try:
                audio_bytes = self.audio[offset * 1000 : offset_end * 1000].export(format="mp3").read()
                temp_audio = tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp3", prefix=f"gst_batch_{self.batch_number}_"
                )
                temp_audio.write(audio_bytes)
                temp_audio.close()
                audio_chunk_path = temp_audio.name
            except Exception as e:
                warning(f"Failed to slice audio chunk: {e}")

        # Context items (preceding translated lines)
        context_items: List[SubtitleObject] = []
        if self.resume_context_size > 0 and start_idx > 0:
            context_start = max(0, start_idx - self.resume_context_size)
            for c_idx in range(context_start, start_idx):
                context_items.append(
                    {
                        "index": str(c_idx),
                        "text": self.translated_subtitles[c_idx].content,
                    }
                )

        # System prompt with translation instructions
        system_prompt = get_translate_instruction(
            language=self.target_language,
            thinking=self.thinking,
            thinking_compatible=self.thinking,
            audio_file=self.audio_file,
            description=self.description,
        )

        total_batches = max(1, ((self.total_lines - 1) // self.batch_size) + 1)

        payload: BatchPayload = {
            "batch_number": self.batch_number,
            "total_batches": total_batches,
            "start_line": start_idx + 1,
            "end_line": end_idx,
            "total_lines": self.total_lines,
            "progress_percent": round((start_idx / self.total_lines) * 100.0, 2),
            "system_prompt": system_prompt,
            "batch": batch_items,
            "context": context_items,
            "audio_chunk_path": audio_chunk_path,
            "audio_bytes": audio_bytes,
            "is_complete": False,
        }
        return payload

    def commit_batch(self, data: Union[str, List[dict], dict]) -> Dict[str, Any]:
        """
        Validate and commit translated batch items.
        Updates translated subtitles, writes atomically to output file, and updates progress.
        """
        if self.is_complete():
            return {
                "success": True,
                "is_complete": True,
                "message": "Translation is already complete.",
                "status": self.get_status(),
            }

        # Parse input data
        parsed_items: List[dict] = []
        if isinstance(data, str):
            try:
                loaded = json_repair.loads(data)
                if isinstance(loaded, dict) and "batch" in loaded:
                    parsed_items = loaded["batch"]
                elif isinstance(loaded, dict) and "items" in loaded:
                    parsed_items = loaded["items"]
                elif isinstance(loaded, list):
                    parsed_items = loaded
                else:
                    parsed_items = [loaded] if isinstance(loaded, dict) else []
            except Exception as e:
                return {"success": False, "error": f"JSON parse error: {e}"}
        elif isinstance(data, dict):
            if "batch" in data and isinstance(data["batch"], list):
                parsed_items = data["batch"]
            elif "items" in data and isinstance(data["items"], list):
                parsed_items = data["items"]
            else:
                parsed_items = [data]
        elif isinstance(data, list):
            parsed_items = data

        parsed_items = self._flatten_repaired_json(parsed_items)

        if not parsed_items:
            return {"success": False, "error": "No valid subtitle items found in translation data."}

        start_idx = self.current_line - 1
        expected_count = min(self.batch_size, self.total_lines - start_idx)

        # Validation: check item count
        if len(parsed_items) != expected_count:
            return {
                "success": False,
                "error": f"Item count mismatch: expected {expected_count} items, but received {len(parsed_items)} items.",
                "expected_count": expected_count,
                "received_count": len(parsed_items),
            }

        # Apply translations
        for i, item in enumerate(parsed_items):
            target_idx = start_idx + i
            if target_idx >= self.total_lines:
                break

            text = item.get("text", "") if isinstance(item, dict) else str(item)
            if self._dominant_strong_direction(text) == "rtl":
                formatted_text = f"\u202b{text}\u202c"
            else:
                formatted_text = text

            self.translated_subtitles[target_idx].content = formatted_text

        # Advance line pointer
        next_line = start_idx + len(parsed_items) + 1
        self.current_line = next_line
        self.batch_number += 1

        # Write output file and save progress
        SubtitleSession._save_subtitle_file(self.input_file, self.translated_subtitles, self.output_file)
        self._save_progress(self.current_line)

        is_done = self.is_complete()
        if is_done:
            self.cleanup()

        return {
            "success": True,
            "is_complete": is_done,
            "committed_lines": len(parsed_items),
            "current_line": self.current_line,
            "output_file": self.output_file,
            "status": self.get_status(),
        }

    def cleanup(self):
        """Remove temporary extracted audio, subtitle, and progress files."""
        if self.audio_extracted and self.audio_file and os.path.exists(self.audio_file):
            try:
                os.remove(self.audio_file)
            except Exception:
                pass
        if self.subtitle_extracted and self.input_file and os.path.exists(self.input_file):
            try:
                os.remove(self.input_file)
            except Exception:
                pass
        if self.progress_file and os.path.exists(self.progress_file):
            try:
                os.remove(self.progress_file)
            except Exception:
                pass

    def reset_progress(self):
        """Reset progress and start from line 1."""
        if self.progress_file and os.path.exists(self.progress_file):
            try:
                os.remove(self.progress_file)
            except Exception:
                pass
        self.current_line = 1
        self.batch_number = 1
        self.translated_subtitles = self._copy_subtitles(self.original_subtitles)


class TranscriptionSession:
    """
    Core audio/video subtitle transcription session engine.
    Manages audio extraction, timestamped slicing, prompt formatting,
    transcription parsing, timestamp offset calculation, validation,
    atomic saving, and progress tracking.
    """

    def __init__(
        self,
        audio_file: Optional[str] = None,
        video_file: Optional[str] = None,
        output_file: Optional[str] = None,
        audio_chunk_size: int = 600,
        start_time: Optional[int] = None,
        extract_audio: bool = True,
        isolate_voice: bool = False,
        resume: Optional[bool] = None,
        description: Optional[str] = None,
        thinking: bool = True,
    ):
        self.audio_file = audio_file
        self.video_file = video_file
        self.audio_chunk_size = max(10, int(audio_chunk_size))
        self.extract_audio = extract_audio
        self.isolate_voice = isolate_voice
        self.resume = resume
        self.description = description
        self.thinking = thinking
        self.audio_extracted = False

        # Check if audio_file passed is a video file
        video_extensions = (".mkv", ".mp4", ".avi", ".mov", ".flv", ".wmv", ".webm", ".m4v", ".ts")
        if self.audio_file and self.audio_file.lower().endswith(video_extensions):
            if not self.video_file:
                self.video_file = self.audio_file
            self.audio_file = None

        # Extract audio from video if needed
        self._prepare_media_inputs()

        if not self.audio_file or not os.path.exists(self.audio_file):
            raise FileNotFoundError(f"Audio file not found: {self.audio_file}")

        # Resolve output and progress file paths
        base_source = self.video_file or self.audio_file or "audio"
        base_name = os.path.splitext(os.path.basename(base_source))[0]
        if base_name.endswith("_extracted"):
            base_name = base_name[:-10]
        dir_path = os.path.dirname(base_source)

        if output_file:
            self.output_file = output_file
        else:
            self.output_file = os.path.join(dir_path, f"{base_name}.srt") if dir_path else f"{base_name}.srt"

        self.progress_file = os.path.join(dir_path, f"{base_name}.progress") if dir_path else f"{base_name}.progress"

        # Load audio and determine duration
        self.audio = None
        self._init_audio()

        # State tracking
        self.current_seconds: int = 0
        self.chunk_number: int = 1
        self.transcribed_subtitles: List[Subtitle] = []
        self._temp_chunk_files: List[str] = []
        self._init_transcription_state(start_time)

    def _prepare_media_inputs(self):
        """Extract audio from video file if video is provided without audio."""
        if self.video_file and not self.audio_file:
            if not os.path.exists(self.video_file):
                raise FileNotFoundError(f"Video file not found: {self.video_file}")
            if not check_ffmpeg_installation():
                raise RuntimeError("FFmpeg is required to extract audio from video.")
            self.audio_file = extract_audio_from_video(self.video_file, isolate_voice=self.isolate_voice)
            self.audio_extracted = True
        elif self.audio_file and (
            self.audio_file.endswith("_extracted.mp3") or self.audio_file.endswith("_isolated_voice.mp3")
        ):
            if self.video_file:
                self.audio_extracted = True

    def _init_audio(self):
        """Load AudioSegment and calculate total duration in seconds."""
        from pydub import AudioSegment

        self.audio = AudioSegment.from_file(self.audio_file)
        self.total_seconds = int(len(self.audio) / 1000)

    def _read_saved_progress(self) -> Optional[int]:
        """Read saved progress time from .progress file."""
        if not self.progress_file or not os.path.exists(self.progress_file):
            return None
        try:
            with open(self.progress_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            source_file = self.audio_file if self.audio_file else self.video_file
            if data.get("input_file") == source_file and "time" in data:
                return int(data["time"])
        except Exception:
            pass
        return None

    def _save_progress(self, time_in_seconds: int):
        """Save progress atomically."""
        if not self.progress_file:
            return
        try:
            source_file = self.audio_file if self.audio_file else self.video_file
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump({"time": time_in_seconds, "input_file": source_file}, f)
        except Exception as e:
            warning(f"Failed to save progress: {e}")

    def _init_transcription_state(self, explicit_start_time: Optional[int]):
        """Initialize start pointer and load existing subtitles if resuming."""
        saved_time = self._read_saved_progress()

        if explicit_start_time is not None:
            self.current_seconds = max(0, min(explicit_start_time, self.total_seconds))
        elif saved_time is not None and (self.resume is True or self.resume is None):
            self.current_seconds = max(0, min(saved_time, self.total_seconds))

        if self.current_seconds > 0 and os.path.exists(self.output_file):
            try:
                self.transcribed_subtitles = SubtitleSession._parse_subtitle_file(self.output_file)
            except Exception:
                self.transcribed_subtitles = []

        if self.current_seconds > 0:
            self.chunk_number = (self.current_seconds // self.audio_chunk_size) + 1

    def is_complete(self) -> bool:
        """Check if all audio chunks have been processed."""
        return self.current_seconds >= self.total_seconds

    def get_status(self) -> Dict[str, Any]:
        """Get machine-parseable progress status."""
        percent = (self.current_seconds / self.total_seconds) * 100.0 if self.total_seconds > 0 else 100.0
        return {
            "current_seconds": self.current_seconds,
            "total_seconds": self.total_seconds,
            "progress_percent": round(percent, 2),
            "chunk_number": self.chunk_number,
            "subtitle_count": len(self.transcribed_subtitles),
            "is_complete": self.is_complete(),
            "output_file": self.output_file,
            "audio_file": self.audio_file,
        }

    def get_next_chunk(self) -> Optional[TranscribeChunkPayload]:
        """
        Generate the next audio slice and instructions payload for transcription.
        """
        if self.is_complete():
            return None

        chunk_end = min(self.current_seconds + self.audio_chunk_size, self.total_seconds)
        audio_bytes = self.audio[self.current_seconds * 1000 : chunk_end * 1000].export(format="mp3").read()

        temp_audio = tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp3", prefix=f"gst_transcribe_chunk_{self.chunk_number}_"
        )
        temp_audio.write(audio_bytes)
        temp_audio.close()
        self._temp_chunk_files.append(temp_audio.name)

        system_prompt = get_transcribe_instruction(
            thinking=self.thinking,
            thinking_compatible=self.thinking,
            description=self.description,
        )

        total_chunks = max(1, ((self.total_seconds - 1) // self.audio_chunk_size) + 1)
        percent = (self.current_seconds / self.total_seconds) * 100.0 if self.total_seconds > 0 else 0.0

        return {
            "chunk_number": self.chunk_number,
            "total_chunks": total_chunks,
            "start_seconds": self.current_seconds,
            "end_seconds": chunk_end,
            "total_seconds": self.total_seconds,
            "progress_percent": round(percent, 2),
            "system_prompt": system_prompt,
            "audio_chunk_path": temp_audio.name,
            "audio_bytes": audio_bytes,
            "is_complete": False,
        }

    @staticmethod
    def _normalize_timestamp(ts_str: str) -> str:
        """Converts 'HH:MM:SS' back to 'MM:SS' for timestamp parsing."""
        parts = ts_str.split(":")
        if len(parts) == 3:
            try:
                h, m, s = map(int, parts)
                total_minutes = (h * 60) + m
                return f"{total_minutes:02}:{s:02}"
            except (ValueError, TypeError):
                return ts_str
        return ts_str

    @staticmethod
    def _save_transcribed_subtitles(subtitles: List[Subtitle], output_file: str):
        """Save transcribed subtitle list using pysubs2 atomic replace."""
        subs = pysubs2.SSAFile()
        for sub in subtitles:
            ev = pysubs2.SSAEvent(
                start=int(sub.start.total_seconds() * 1000),
                end=int(sub.end.total_seconds() * 1000),
                text=sub.content.replace("\n", "\\N") if output_file.lower().endswith(".ass") else sub.content,
            )
            subs.append(ev)

        temp_dir = os.path.dirname(os.path.abspath(output_file)) or "."
        temp_out = os.path.join(temp_dir, f".tmp_{os.path.basename(output_file)}")
        subs.save(temp_out, encoding="utf-8")
        os.replace(temp_out, output_file)

    def commit_chunk(self, data: Union[str, List[dict], dict]) -> Dict[str, Any]:
        """
        Validate and commit transcribed items for the current audio chunk.
        """
        if self.is_complete():
            return {
                "success": True,
                "is_complete": True,
                "added_subtitles": 0,
                "current_seconds": self.current_seconds,
                "total_seconds": self.total_seconds,
                "output_file": self.output_file,
                "status": self.get_status(),
            }

        # Parse data with json_repair
        parsed_items: List[dict] = []
        if isinstance(data, str):
            try:
                raw = json_repair.loads(data)
                if isinstance(raw, list):
                    parsed_items = raw
                elif isinstance(raw, dict):
                    parsed_items = [raw]
            except Exception as e:
                return {"success": False, "error": f"JSON parse error: {e}"}
        elif isinstance(data, list):
            parsed_items = data
        elif isinstance(data, dict):
            parsed_items = [data]
        else:
            return {"success": False, "error": f"Invalid data type: {type(data)}"}

        if not all(isinstance(item, dict) for item in parsed_items):
            parsed_items = SubtitleSession._flatten_repaired_json(parsed_items)

        current_offset = self.current_seconds
        added_count = 0

        for item in parsed_items:
            if not isinstance(item, dict) or "text" not in item or "time_start" not in item or "time_end" not in item:
                continue

            text = item["text"]
            start_ts = self._normalize_timestamp(str(item["time_start"]))
            end_ts = self._normalize_timestamp(str(item["time_end"]))

            try:
                start_td = convert_timestamp_to_timedelta(start_ts, offset=current_offset)
                end_td = convert_timestamp_to_timedelta(end_ts, offset=current_offset)
            except Exception as e:
                warning(f"Error parsing timestamp ({start_ts} - {end_ts}): {e}")
                continue

            if SubtitleSession._dominant_strong_direction(text) == "rtl":
                text = f"\u202b{text}\u202c"

            idx = len(self.transcribed_subtitles) + 1
            self.transcribed_subtitles.append(Subtitle(index=idx, start=start_td, end=end_td, content=text))
            added_count += 1

        # Advance audio pointer
        self.current_seconds = min(self.current_seconds + self.audio_chunk_size, self.total_seconds)
        self.chunk_number += 1

        # Save output and progress
        self._save_transcribed_subtitles(self.transcribed_subtitles, self.output_file)
        self._save_progress(self.current_seconds)

        is_done = self.is_complete()
        if is_done:
            self.cleanup()

        return {
            "success": True,
            "is_complete": is_done,
            "added_subtitles": added_count,
            "current_seconds": self.current_seconds,
            "total_seconds": self.total_seconds,
            "output_file": self.output_file,
            "status": self.get_status(),
        }

    def cleanup(self):
        """Remove temporary extracted audio, chunk slices, and progress files."""
        if self.audio_extracted and self.audio_file and os.path.exists(self.audio_file):
            try:
                os.remove(self.audio_file)
            except Exception:
                pass
        if hasattr(self, "_temp_chunk_files"):
            for chunk_path in self._temp_chunk_files:
                if chunk_path and os.path.exists(chunk_path):
                    try:
                        os.remove(chunk_path)
                    except Exception:
                        pass
        if self.progress_file and os.path.exists(self.progress_file):
            try:
                os.remove(self.progress_file)
            except Exception:
                pass

    def reset_progress(self):
        """Reset transcription progress to beginning."""
        if self.progress_file and os.path.exists(self.progress_file):
            try:
                os.remove(self.progress_file)
            except Exception:
                pass
        self.current_seconds = 0
        self.chunk_number = 1
        self.transcribed_subtitles = []
