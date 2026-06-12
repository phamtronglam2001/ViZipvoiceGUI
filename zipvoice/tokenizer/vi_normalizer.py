"""Pluggable Vietnamese text normalization pipeline for ViZipVoice TTS."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

NormalizerFn = Callable[[str], str]

STEP_LABELS: dict[str, str] = {
    "none": "— Không —",
    "soe_vinorm": "soe-vinorm (số, ngày, đơn vị, viết tắt)",
    "spacing": "Dọn khoảng trắng quanh dấu câu",
    "vieneu": "VieNeu (gộp khoảng trắng thừa)",
    "lowercase": "Chuyển chữ thường",
    "strip_quotes": "Bỏ dấu ngoặc kép",
    "dot_newline": "Chấm + space → xuống dòng",
    "period_break": "Cấu trúc TTS (ngoặc/số+chấm → xuống dòng)",
    "newline_sentence": "Xuống dòng → ranh giới câu (thêm chấm)",
    "join_soft_breaks": "Gộp xuống dòng PDF (dòng ngắn, viết thường)",
    "sea_g2p": "sea-g2p NSW (cần pip install sea-g2p)",
}

DEFAULT_PIPELINE: list[str] = ["soe_vinorm", "spacing"]

# ZipVoice-Vietnamese-ONNX-GUI audiobook pipeline (sea-g2p → … → VieNeu).
AUDIOBOOK_PRESET_PIPELINE: list[str] = [
    "sea_g2p",
    "dot_newline",
    "period_break",
    "newline_sentence",
    "join_soft_breaks",
    "vieneu",
]

PUNCTUATION_NO_SPACE_BEFORE = r",.;:!?…%"
OPENING_QUOTES_AND_BRACKETS = r"\(\[\{«“‘"
CLOSING_QUOTES_AND_BRACKETS = r"\)\]\}»”’"

_RE_MULTI_SPACE = re.compile(r" {2,}")
_RE_BRACKET = re.compile(r"[\(\)\[\]\{]")
_RE_SPACES_AROUND_NL = re.compile(r"[ \t]*\n[ \t]*")
_RE_MULTI_NL = re.compile(r"\n{2,}")
_RE_DIGIT_PERIOD = re.compile(r"(\d{1,4})\.(?!\d)\s+")
_RE_DOT_SPACE = re.compile(r"(?<![0-9])\.\s+")
_VI_NUM_WORDS = (
    "một|hai|ba|bốn|tư|năm|sáu|bảy|tám|chín|"
    "mười|mười một|mười hai|mười ba|mười bốn|mười lăm|"
    "mười sáu|mười bảy|mười tám|mười chín|"
    "hai mươi|ba mươi|bốn mươi|năm mươi"
)
_RE_WORD_PERIOD = re.compile(
    rf"\b({_VI_NUM_WORDS})\s*\.\s+",
    re.IGNORECASE | re.UNICODE,
)
_TERMINAL_PUNCT = re.compile(r'[.!?…]["\'""»)\]]*\s*$')
_RE_LOWERCASE_START = re.compile(r"^[a-zà-ỹ0-9]", re.UNICODE)
_MAX_SOFT_JOIN_LINE = 120

_sea_g2p_normalizer = None


def cleanup_vietnamese_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(
        rf"\s+([{re.escape(PUNCTUATION_NO_SPACE_BEFORE)}])",
        r"\1",
        text,
    )
    text = re.sub(
        rf"\s+([{CLOSING_QUOTES_AND_BRACKETS}])",
        r"\1",
        text,
    )
    text = re.sub(
        rf"([{OPENING_QUOTES_AND_BRACKETS}])\s+",
        r"\1",
        text,
    )
    text = re.sub(
        rf"([{re.escape(PUNCTUATION_NO_SPACE_BEFORE)}])"
        rf"([^\s{CLOSING_QUOTES_AND_BRACKETS}])",
        r"\1 \2",
        text,
    )
    return text.strip()


def _normalize_soe_vinorm(text: str) -> str:
    try:
        from soe_vinorm import normalize_text as soe_normalize
    except ImportError as exc:
        raise RuntimeError(
            "Cần cài soe-vinorm: uv pip install soe-vinorm"
        ) from exc
    return soe_normalize(text)


def _normalize_vieneu(text: str) -> str:
    if "\n" not in text:
        return _RE_MULTI_SPACE.sub(" ", text).strip()
    lines = []
    for line in text.split("\n"):
        lines.append(_RE_MULTI_SPACE.sub(" ", line).strip() if line.strip() else "")
    return "\n".join(lines)


def _normalize_lowercase(text: str) -> str:
    return text.lower()


def _normalize_strip_quotes(text: str) -> str:
    return text.replace('"', "").replace("'", "")


def _normalize_dot_newline(text: str) -> str:
    if not text:
        return text
    return _RE_DOT_SPACE.sub(".\n", text)


def _normalize_period_break(text: str) -> str:
    if not text or not text.strip():
        return text
    out = _RE_BRACKET.sub("\n", text)
    out = _RE_SPACES_AROUND_NL.sub("\n", out)
    out = _RE_MULTI_NL.sub("\n", out)
    out = out.strip()
    out = _RE_DIGIT_PERIOD.sub(r"\1.\n", out)
    out = _RE_WORD_PERIOD.sub(lambda m: f"{m.group(1)}.\n", out)
    return out


def _normalize_newline_sentence(text: str) -> str:
    if not text or not text.strip():
        return text
    lines = text.split("\n")
    out: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.rstrip()
        if not stripped:
            out.append("")
            continue
        if i < len(lines) - 1 and not _TERMINAL_PUNCT.search(stripped):
            stripped = stripped.rstrip(".") + "."
        out.append(stripped)
    return "\n".join(out)


def _should_join_soft_break(prev: str, nxt: str) -> bool:
    if _TERMINAL_PUNCT.search(prev):
        return False
    if len(prev) > _MAX_SOFT_JOIN_LINE or len(nxt) > _MAX_SOFT_JOIN_LINE:
        return False
    if not _RE_LOWERCASE_START.match(nxt):
        return False
    return True


def _normalize_join_soft_breaks(text: str) -> str:
    if not text or "\n" not in text:
        return text
    lines = text.split("\n")
    merged: list[str] = []
    buf = ""
    for line in lines:
        if not line.strip():
            if buf:
                merged.append(buf)
                buf = ""
            merged.append("")
            continue
        piece = line.strip()
        if not buf:
            buf = piece
            continue
        if _should_join_soft_break(buf, piece):
            buf = f"{buf} {piece}"
        else:
            merged.append(buf)
            buf = piece
    if buf:
        merged.append(buf)
    return "\n".join(merged)


def _strip_sea_g2p_en_tags(text: str) -> str:
    return re.sub(r"</?en>", "", text)


def _normalize_sea_g2p(text: str) -> str:
    global _sea_g2p_normalizer
    if _sea_g2p_normalizer is None:
        from sea_g2p import Normalizer

        _sea_g2p_normalizer = Normalizer()

    def _run_line(line: str) -> str:
        out = _sea_g2p_normalizer.normalize(line)
        if isinstance(out, list):
            out = out[0] if out else line
        return _strip_sea_g2p_en_tags(str(out))

    if "\n" not in text:
        return _run_line(text)
    return "\n".join(_run_line(line) if line.strip() else "" for line in text.split("\n"))


NORMALIZERS: dict[str, NormalizerFn] = {
    "soe_vinorm": _normalize_soe_vinorm,
    "spacing": cleanup_vietnamese_spacing,
    "vieneu": _normalize_vieneu,
    "lowercase": _normalize_lowercase,
    "strip_quotes": _normalize_strip_quotes,
    "dot_newline": _normalize_dot_newline,
    "period_break": _normalize_period_break,
    "newline_sentence": _normalize_newline_sentence,
    "join_soft_breaks": _normalize_join_soft_breaks,
    "sea_g2p": _normalize_sea_g2p,
}


def build_normalize_pipeline(steps: list[str] | str | None) -> list[str]:
    if steps is None:
        return list(DEFAULT_PIPELINE)
    if isinstance(steps, str):
        key = (steps or "none").strip().lower()
        if key in ("none", "off", ""):
            return []
        if key not in NORMALIZERS:
            raise ValueError(f"Bước chuẩn hóa không hợp lệ: {steps!r}")
        return [key]

    pipeline: list[str] = []
    seen: set[str] = set()
    for raw in steps:
        key = (raw or "none").strip().lower()
        if key in ("none", "off", ""):
            continue
        if key not in NORMALIZERS:
            raise ValueError(f"Bước chuẩn hóa không hợp lệ: {raw!r}")
        if key in seen:
            continue
        seen.add(key)
        pipeline.append(key)
    return pipeline


def format_pipeline_label(pipeline: list[str]) -> str:
    if not pipeline:
        return STEP_LABELS["none"]
    return " → ".join(STEP_LABELS.get(step, step) for step in pipeline)


@dataclass(frozen=True)
class NormalizePreview:
    original: str
    normalized: str
    pipeline: list[str]
    steps: list[tuple[str, str, str]]


def normalize_text_pipeline(
    text: str,
    pipeline: list[str] | str | None = None,
    *,
    enabled: bool = True,
) -> str:
    if not enabled:
        return text.strip()
    steps = build_normalize_pipeline(pipeline)
    result = text
    for step in steps:
        result = NORMALIZERS[step](result)
    return result.strip()


def preview_normalize(
    text: str,
    pipeline: list[str] | str | None = None,
    *,
    enabled: bool = True,
    max_chars: int = 12_000,
) -> NormalizePreview:
    raw = (text or "")[:max_chars]
    if not enabled:
        return NormalizePreview(
            original=raw,
            normalized=raw.strip(),
            pipeline=[],
            steps=[],
        )

    steps = build_normalize_pipeline(pipeline)
    trace: list[tuple[str, str, str]] = []
    current = raw
    for step in steps:
        nxt = NORMALIZERS[step](current)
        trace.append((step, current, nxt))
        current = nxt

    return NormalizePreview(
        original=raw,
        normalized=current.strip(),
        pipeline=steps,
        steps=trace,
    )


def format_preview_markdown(preview: NormalizePreview) -> str:
    lines = [
        f"**Pipeline:** {format_pipeline_label(preview.pipeline)}",
        "",
        "### Kết quả",
        "```",
        preview.normalized,
        "```",
    ]
    if preview.steps:
        lines.extend(["", "### Từng bước"])
        for index, (step, before, after) in enumerate(preview.steps, start=1):
            label = STEP_LABELS.get(step, step)
            lines.append(f"**{index}. {label}**")
            if before != after:
                lines.append(f"- Trước: `{before[:200]}{'…' if len(before) > 200 else ''}`")
                lines.append(f"- Sau: `{after[:200]}{'…' if len(after) > 200 else ''}`")
            else:
                lines.append("- _(không đổi)_")
    return "\n".join(lines)
