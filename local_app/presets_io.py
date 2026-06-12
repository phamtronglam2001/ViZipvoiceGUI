"""Built-in and user-saved Advanced TTS presets for local Gradio GUIs."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from collections.abc import Callable
from typing import Any

import gradio as gr

from zipvoice.tokenizer.vi_normalizer import (
    AUDIOBOOK_PRESET_PIPELINE,
    DEFAULT_PIPELINE,
    build_normalize_pipeline,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILTIN_PRESETS_DIR = REPO_ROOT / "assets" / "presets"
SAVED_PRESETS_FILE = Path(__file__).resolve().parent / "saved_presets.json"

PRESET_SCHEMA = 1
BUILTIN_DEFAULT_KEY = "default"
BUILTIN_AUDIOBOOK_KEY = "audiobook"


@dataclass
class AdvancedPreset:
    name: str
    description: str = ""
    norm_pipeline: list[str] = field(default_factory=lambda: list(DEFAULT_PIPELINE))
    normalize_vietnamese: bool = True
    split_sentences: bool = True
    remove_long_sil: bool = False
    crossfade_ms: int = 80
    silence_ms: int = 180
    fade_in_ms: int = 20
    fade_out_ms: int = 80
    num_step: int = 16
    guidance_scale: float = 1.0
    speed: float = 1.0
    t_shift: float = 0.5
    max_duration: int = 100
    seed: int = 666

    def pipeline_str(self) -> str:
        return ", ".join(self.norm_pipeline)


def _slug_name(name: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", name.strip().lower(), flags=re.UNICODE)
    return slug.strip("_") or "preset"


def default_preset() -> AdvancedPreset:
    return AdvancedPreset(
        name="Default",
        description="soe-vinorm → spacing (mặc định ViZipVoice)",
        norm_pipeline=list(DEFAULT_PIPELINE),
    )


def audiobook_preset() -> AdvancedPreset:
    return AdvancedPreset(
        name="Audiobook",
        description="sea-g2p → … → VieNeu (ZipVoice-Vietnamese-ONNX-GUI)",
        norm_pipeline=list(AUDIOBOOK_PRESET_PIPELINE),
        silence_ms=350,
        split_sentences=True,
        num_step=16,
        guidance_scale=1.0,
        speed=1.0,
        t_shift=0.5,
    )


_BUILTIN_FACTORIES: dict[str, Callable[[], AdvancedPreset]] = {
    BUILTIN_DEFAULT_KEY: default_preset,
    BUILTIN_AUDIOBOOK_KEY: audiobook_preset,
}


def _preset_to_dict(preset: AdvancedPreset) -> dict[str, Any]:
    data = asdict(preset)
    data["schema_version"] = PRESET_SCHEMA
    return data


def _preset_from_dict(data: dict[str, Any]) -> AdvancedPreset:
    pipeline = build_normalize_pipeline(data.get("norm_pipeline") or DEFAULT_PIPELINE)
    return AdvancedPreset(
        name=str(data.get("name") or "Preset"),
        description=str(data.get("description") or ""),
        norm_pipeline=pipeline,
        normalize_vietnamese=bool(data.get("normalize_vietnamese", True)),
        split_sentences=bool(data.get("split_sentences", True)),
        remove_long_sil=bool(data.get("remove_long_sil", False)),
        crossfade_ms=int(data.get("crossfade_ms", 80)),
        silence_ms=int(data.get("silence_ms", 180)),
        fade_in_ms=int(data.get("fade_in_ms", 20)),
        fade_out_ms=int(data.get("fade_out_ms", 80)),
        num_step=int(data.get("num_step", 16)),
        guidance_scale=float(data.get("guidance_scale", 1.0)),
        speed=float(data.get("speed", 1.0)),
        t_shift=float(data.get("t_shift", 0.5)),
        max_duration=int(data.get("max_duration", 100)),
        seed=int(data.get("seed", 666)),
    )


def ensure_builtin_preset_files() -> None:
    BUILTIN_PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    for key, factory in _BUILTIN_FACTORIES.items():
        path = BUILTIN_PRESETS_DIR / f"{key}.json"
        if path.is_file():
            continue
        path.write_text(
            json.dumps(_preset_to_dict(factory()), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def load_builtin_preset(key: str) -> AdvancedPreset:
    ensure_builtin_preset_files()
    path = BUILTIN_PRESETS_DIR / f"{key}.json"
    if path.is_file():
        return _preset_from_dict(json.loads(path.read_text(encoding="utf-8")))
    factory = _BUILTIN_FACTORIES.get(key)
    if factory is None:
        raise KeyError(key)
    return factory()


def _load_saved_presets() -> dict[str, AdvancedPreset]:
    if not SAVED_PRESETS_FILE.is_file():
        return {}
    raw = json.loads(SAVED_PRESETS_FILE.read_text(encoding="utf-8"))
    items = raw.get("presets", raw) if isinstance(raw, dict) else {}
    out: dict[str, AdvancedPreset] = {}
    for key, data in items.items():
        if isinstance(data, dict):
            preset = _preset_from_dict(data)
            if not preset.name:
                preset = AdvancedPreset(**{**asdict(preset), "name": key})
            out[key] = preset
    return out


def _write_saved_presets(presets: dict[str, AdvancedPreset]) -> None:
    payload = {
        "schema_version": PRESET_SCHEMA,
        "presets": {key: _preset_to_dict(p) for key, p in sorted(presets.items())},
    }
    SAVED_PRESETS_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def preset_dropdown_choices() -> list[tuple[str, str]]:
    ensure_builtin_preset_files()
    choices: list[tuple[str, str]] = [
        ("Default", BUILTIN_DEFAULT_KEY),
        ("Audiobook", BUILTIN_AUDIOBOOK_KEY),
    ]
    for key, preset in sorted(_load_saved_presets().items(), key=lambda x: x[0].lower()):
        label = preset.name or key
        choices.append((f"{label} (saved)", f"saved:{key}"))
    return choices


def load_preset_by_key(key: str) -> AdvancedPreset:
    if key == BUILTIN_DEFAULT_KEY:
        return load_builtin_preset(BUILTIN_DEFAULT_KEY)
    if key == BUILTIN_AUDIOBOOK_KEY:
        return load_builtin_preset(BUILTIN_AUDIOBOOK_KEY)
    if key.startswith("saved:"):
        saved_key = key[6:]
        saved = _load_saved_presets()
        if saved_key not in saved:
            raise FileNotFoundError(f"Saved preset not found: {saved_key}")
        return saved[saved_key]
    raise KeyError(key)


def save_user_preset(save_name: str, state: dict[str, Any]) -> tuple[str, str]:
    name = (save_name or "").strip()
    if not name:
        raise ValueError("Nhập tên preset trước khi lưu.")
    key = _slug_name(name)
    preset = _preset_from_dict({**state, "name": name})
    saved = _load_saved_presets()
    saved[key] = preset
    _write_saved_presets(saved)
    return key, str(SAVED_PRESETS_FILE)


def collect_advanced_state(
    *,
    num_step: int,
    guidance_scale: float,
    speed: float,
    t_shift: float,
    max_duration: int,
    seed: int,
    normalize_vietnamese: bool,
    norm_pipeline_raw: str,
    split_sentences: bool,
    remove_long_sil: bool,
    crossfade_ms: int,
    silence_ms: int,
    fade_in_ms: int,
    fade_out_ms: int,
) -> dict[str, Any]:
    if not norm_pipeline_raw or not norm_pipeline_raw.strip():
        pipeline = list(DEFAULT_PIPELINE)
    else:
        pipeline = build_normalize_pipeline(
            [part.strip() for part in norm_pipeline_raw.split(",") if part.strip()]
        )

    return {
        "norm_pipeline": pipeline,
        "normalize_vietnamese": bool(normalize_vietnamese),
        "split_sentences": bool(split_sentences),
        "remove_long_sil": bool(remove_long_sil),
        "crossfade_ms": int(crossfade_ms),
        "silence_ms": int(silence_ms),
        "fade_in_ms": int(fade_in_ms),
        "fade_out_ms": int(fade_out_ms),
        "num_step": int(num_step),
        "guidance_scale": float(guidance_scale),
        "speed": float(speed),
        "t_shift": float(t_shift),
        "max_duration": int(max_duration),
        "seed": int(seed),
    }


def on_load_preset(preset_key: str | None) -> tuple:
    if not preset_key:
        preset = default_preset()
    else:
        preset = load_preset_by_key(preset_key)
    return apply_preset_to_gui(preset)


def on_save_preset(
    save_name: str,
    num_step: int,
    guidance_scale: float,
    speed: float,
    t_shift: float,
    max_duration: int,
    seed: int,
    normalize_vietnamese: bool,
    norm_pipeline_raw: str,
    split_sentences: bool,
    remove_long_sil: bool,
    crossfade_ms: int,
    silence_ms: int,
    fade_in_ms: int,
    fade_out_ms: int,
) -> tuple:
    state = collect_advanced_state(
        num_step=num_step,
        guidance_scale=guidance_scale,
        speed=speed,
        t_shift=t_shift,
        max_duration=max_duration,
        seed=seed,
        normalize_vietnamese=normalize_vietnamese,
        norm_pipeline_raw=norm_pipeline_raw,
        split_sentences=split_sentences,
        remove_long_sil=remove_long_sil,
        crossfade_ms=crossfade_ms,
        silence_ms=silence_ms,
        fade_in_ms=fade_in_ms,
        fade_out_ms=fade_out_ms,
    )
    _, path = save_user_preset(save_name, state)
    choices = preset_dropdown_choices()
    key = _slug_name(save_name.strip())
    return (
        gr.update(choices=choices, value=f"saved:{key}"),
        f"Đã lưu preset **{save_name.strip()}** → `{path}`",
    )


def apply_preset_to_gui(preset: AdvancedPreset) -> tuple:
    """Gradio updates for all Advanced accordion controls + status markdown."""
    status = (
        f"Đã tải preset **{preset.name}**"
        + (f" — {preset.description}" if preset.description else "")
    )
    return (
        gr.update(value=int(preset.num_step)),
        gr.update(value=float(preset.guidance_scale)),
        gr.update(value=float(preset.speed)),
        gr.update(value=float(preset.t_shift)),
        gr.update(value=int(preset.max_duration)),
        gr.update(value=int(preset.seed)),
        gr.update(value=bool(preset.normalize_vietnamese)),
        gr.update(value=preset.pipeline_str()),
        gr.update(value=bool(preset.split_sentences)),
        gr.update(value=bool(preset.remove_long_sil)),
        gr.update(value=int(preset.crossfade_ms)),
        gr.update(value=int(preset.silence_ms)),
        gr.update(value=int(preset.fade_in_ms)),
        gr.update(value=int(preset.fade_out_ms)),
        status,
    )
