#!/usr/bin/env python3
"""Antigravity (agy) adapter: Gemini, signed in through the agy CLI, reads local media.

No API key. agy runs headless with its default permissions and never with
--dangerously-skip-permissions, so shell and MCP tools are auto-denied and
Gemini can only open the files in the work directory it is pointed at.
Its file viewer samples a video's frames but not its soundtrack, so audio is
always handed over as a separate file.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from runtime import diagnostic, run_text
from transcribe import Segments

CHUNK_SECONDS = 600.0  # Shorter audio keeps Gemini's timestamps tight and its output in budget.
_HINTS = {
    'missing': 'agy is not installed or not on PATH; install Antigravity, run `agy` once to sign in, '
               'or set WATCH_AGY_BIN to its absolute path',
    'failed': 'agy exited with an error; run `agy` interactively once to check sign-in and quota',
    'denied': 'Gemini tried to use a tool that headless watch never grants',
    'empty': 'agy returned no answer text',
    'response': 'agy printed something this version cannot read',
    'transcript': 'Gemini returned no transcript lines in the expected format',
}
DEFAULT_QUESTION = ('Give a thorough summary of this video: what is shown, any on-screen text or code, '
                    'and what is said, in chronological order.')
_TOOLS_NOTE = ('Use only your file viewer to open these files; shell commands, MCP tools, and the web '
               'are unavailable, so do not try them.')
_LINE = re.compile(r'^\s*\[\s*([0-9:.]+)\s*[-–]\s*([0-9:.]+)\s*\]\s*(.+?)\s*$')


def _fail(category: str, detail: str = '') -> SystemExit:
    detail = diagnostic(detail, 600)
    return SystemExit(f'agy {category}: {_HINTS[category]}.' + (f' Detail: {detail}.' if detail else ''))


def find_binary(configured: str = '') -> str | None:
    if configured:
        return configured if Path(configured).is_absolute() and Path(configured).is_file() else None
    found = shutil.which('agy')
    if found:
        return found
    fallback = Path.home() / '.local' / 'bin' / 'agy'
    return str(fallback) if fallback.is_file() else None


def command(binary: str, prompt: str, *, model: str, add_dirs, timeout: float) -> list[str]:
    cmd = [binary, '--model', model, '--output-format', 'json', '--disable-slash-commands',
           '--print-timeout', f'{int(timeout)}s']
    for directory in add_dirs:
        cmd += ['--add-dir', str(Path(directory).resolve())]
    return cmd + [f'--print={prompt}']  # Attached, so a prompt can never be read as a flag.


def parse_output(stdout: str) -> dict:
    """agy prints one JSON object; diagnostics may precede it on their own lines."""
    for line in reversed((stdout or '').splitlines()):
        line = line.strip()
        if line.startswith('{'):
            try:
                data = json.loads(line)
            except ValueError:
                continue
            if isinstance(data, dict):
                return data
    raise _fail('response', stdout[-300:] if stdout else 'no output')


def run(prompt: str, *, add_dirs, model: str, timeout: float, binary: str | None, cwd: Path) -> dict:
    if not binary:
        raise _fail('missing')
    result = run_text(command(binary, prompt, model=model, add_dirs=add_dirs, timeout=timeout),
                      cwd=str(cwd), timeout=timeout + 60)
    try:
        data = parse_output(result.stdout)
    except SystemExit:
        if result.returncode:
            raise _fail('failed', f'exit {result.returncode}: {result.stderr or result.stdout}') from None
        raise
    text = str(data.get('response') or '').strip()
    if data.get('status') != 'SUCCESS' and not text:
        raise _fail('failed', f"status {data.get('status')}: {result.stderr}")
    if not text:
        denied = [str(item.get('display_name') or item.get('action')) for item in data.get('denied_actions') or []
                  if isinstance(item, dict)]
        raise _fail('denied', ', '.join(denied)) if denied else _fail('empty')
    usage = data.get('usage') if isinstance(data.get('usage'), dict) else {}
    tokens = usage.get('total_tokens')
    return {'text': text, 'model': model, 'total_tokens': tokens if isinstance(tokens, int) else None}


def _seconds(stamp: str) -> float:
    return sum(float(part) * 60 ** i for i, part in enumerate(reversed(stamp.split(':'))))


def parse_transcript(text: str, duration: float | None = None) -> Segments:
    """Read `[START-END] text` lines into segments; anything else is ignored."""
    if text.strip() == 'NO_SPEECH':
        return Segments(no_speech=True)
    segments = []
    for line in text.splitlines():
        match = _LINE.match(line)
        if not match:
            continue
        try:
            start, end = _seconds(match.group(1)), _seconds(match.group(2))
        except ValueError:
            continue
        if duration and start > duration + 2:  # Past the end of the audio: invented, not heard.
            continue
        segments.append({'start': round(start, 2), 'end': round(max(start, end), 2), 'text': match.group(3)})
    if not segments:
        raise _fail('transcript', text[:200])
    segments.sort(key=lambda seg: seg['start'])
    return Segments(segments)


def transcribe_file(audio: Path, *, model: str, timeout: float, binary: str | None,
                    duration: float | None = None) -> Segments:
    audio = Path(audio).resolve()
    prompt = (f'Open the audio file {audio} and transcribe all speech verbatim in its original language. '
              f'{_TOOLS_NOTE}\nOutput ONLY transcript lines, one per sentence or short phrase, in exactly '
              'this format with times in seconds from the start of the file:\n[START-END] text\n'
              'Example: [12.4-15.9] and that is how it works.\n'
              'No headings, commentary, or markdown. If there is no speech, output exactly: NO_SPEECH')
    result = run(prompt, add_dirs=[audio.parent], model=model, timeout=timeout, binary=binary, cwd=audio.parent)
    return parse_transcript(result['text'], duration)


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f'{hours}:{minutes:02d}:{secs:02d}' if hours else f'{minutes:02d}:{secs:02d}'


def build_prompt(question: str | None, *, video: Path, audio: Path | None, transcript: Path | None,
                 offset: float = 0.0) -> str:
    files = [f'- {video} — the video. Your viewer shows its frames but not its sound.']
    if audio:
        files.append(f'- {audio} — the soundtrack of that video, on the same timeline. Listen to it.')
    if transcript:
        files.append(f'- {transcript} — the platform captions, timestamps in source-video time. '
                     'Use them to check what you hear.')
    timing = ''
    if offset:
        timing = (f'\nThe video and audio files are a clip that starts at {_clock(offset)} in the source video. '
                  f'Add {_clock(offset)} to every position you observe so all cited timestamps are source-video time.')
    return ('Answer the question about a video using these local files. ' + _TOOLS_NOTE + '\n'
            + '\n'.join(files) + timing
            + f'\n\nQuestion: {(question or "").strip() or DEFAULT_QUESTION}\n\n'
            'Cite an MM:SS (or H:MM:SS) timestamp for every claim about a specific moment. Separate what is '
            'seen from what is heard when it matters. If the video does not show or say something needed '
            'to answer, say so plainly instead of guessing.')


def ask(question: str | None, *, video: Path, audio: Path | None, transcript: Path | None, offset: float,
        model: str, timeout: float, binary: str | None, work: Path) -> dict:
    prompt = build_prompt(question, video=Path(video).resolve(), audio=Path(audio).resolve() if audio else None,
                          transcript=Path(transcript).resolve() if transcript else None, offset=offset)
    return run(prompt, add_dirs=[work], model=model, timeout=timeout, binary=binary, cwd=work)
