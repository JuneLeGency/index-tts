#!/usr/bin/env python3
"""IndexTTS2 FastAPI server — zero-shot TTS with emotion control.

Supports:
  - Preset voices (no reference audio needed): /v1/tts-preset
  - Custom voice cloning from reference audio: /v1/tts
  - Emotion control via emo_text, emo_audio, or emo_alpha
  - Automatic text chunking for long texts
  - NDJSON streaming mode (stream=true) for chunk-by-chunk delivery
"""

import base64
import io
import json
import logging
import os
import re
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

logger = logging.getLogger("indextts-server")

tts_model = None

# --- Preset voices (WAV files in presets/ directory) ---
PRESETS_DIR = Path(__file__).parent / "presets"
_PRESET_VOICES: dict[str, str] = {}  # name -> abs path, populated at startup

# --- Chunking config ---
MAX_CHUNK_CHARS = 150
SILENCE_MS = 300


def _load_presets() -> dict[str, str]:
    """Scan presets/ directory for WAV files."""
    presets = {}
    if PRESETS_DIR.is_dir():
        for f in sorted(PRESETS_DIR.glob("*.wav")):
            name = f.stem  # e.g. "female_zh", "male_en", "default"
            presets[name] = str(f.resolve())
            logger.info("  Preset voice: %s -> %s", name, f.name)
    return presets


def split_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text at sentence boundaries, each chunk <= max_chars."""
    sentences = re.split(r"(?<=[。！？.!?\n])", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [text] if text.strip() else []

    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if len(sent) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(sent), max_chars):
                chunks.append(sent[i : i + max_chars])
            continue
        if len(current) + len(sent) > max_chars and current:
            chunks.append(current)
            current = sent
        else:
            current += sent
    if current:
        chunks.append(current)
    return chunks


def concat_audio(arrays: list[np.ndarray], sr: int, silence_ms: int = SILENCE_MS) -> np.ndarray:
    """Concatenate audio arrays with silence padding between them."""
    if len(arrays) == 1:
        return arrays[0]
    silence = np.zeros(int(sr * silence_ms / 1000), dtype=arrays[0].dtype)
    parts: list[np.ndarray] = []
    for i, arr in enumerate(arrays):
        parts.append(arr)
        if i < len(arrays) - 1:
            parts.append(silence)
    return np.concatenate(parts)


def _do_infer(ref_path: str, chunks: list[str], emo_text: str | None,
              emo_audio_path: str | None, emo_alpha: float) -> tuple[list[np.ndarray], int, float]:
    """Core inference loop shared by both endpoints. Returns (audio_arrays, sample_rate, total_time)."""
    audio_arrays: list[np.ndarray] = []
    sample_rate = None
    total_inference = 0.0
    out_paths: list[str] = []

    try:
        for i, chunk in enumerate(chunks):
            logger.info("  Chunk %d/%d (%d chars): %s...",
                        i + 1, len(chunks), len(chunk), chunk[:40])
            out_path = tempfile.mktemp(suffix=".wav")
            out_paths.append(out_path)

            t0 = time.time()
            kwargs = {}
            if emo_text:
                kwargs["use_emo_text"] = True
                kwargs["emo_text"] = emo_text
            if emo_audio_path:
                kwargs["emo_audio_prompt"] = emo_audio_path
            if emo_alpha != 1.0:
                kwargs["emo_alpha"] = emo_alpha

            tts_model.infer(
                spk_audio_prompt=ref_path,
                text=chunk,
                output_path=out_path,
                **kwargs,
            )
            total_inference += time.time() - t0

            data, sr = sf.read(out_path)
            audio_arrays.append(data)
            sample_rate = sr
    finally:
        for p in out_paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    return audio_arrays, sample_rate, total_inference


def _do_infer_streaming(ref_path: str, chunks: list[str], emo_text: str | None,
                        emo_audio_path: str | None, emo_alpha: float):
    """Generator that yields (index, audio_data, sample_rate, inference_time) per chunk."""
    for i, chunk in enumerate(chunks):
        logger.info("  [Stream] Chunk %d/%d (%d chars): %s...",
                    i + 1, len(chunks), len(chunk), chunk[:40])
        out_path = tempfile.mktemp(suffix=".wav")

        try:
            t0 = time.time()
            kwargs = {}
            if emo_text:
                kwargs["use_emo_text"] = True
                kwargs["emo_text"] = emo_text
            if emo_audio_path:
                kwargs["emo_audio_prompt"] = emo_audio_path
            if emo_alpha != 1.0:
                kwargs["emo_alpha"] = emo_alpha

            tts_model.infer(
                spk_audio_prompt=ref_path,
                text=chunk,
                output_path=out_path,
                **kwargs,
            )
            elapsed = time.time() - t0

            data, sr = sf.read(out_path)
            yield i, data, sr, elapsed
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass


def _chunk_to_wav_base64(audio_data: np.ndarray, sample_rate: int) -> str:
    """Encode a single audio chunk as a complete, playable WAV in base64."""
    buf = io.BytesIO()
    sf.write(buf, audio_data, sample_rate, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


def _ndjson_stream(ref_path: str, chunks: list[str], emo_text: str | None,
                   emo_audio_path: str | None, emo_alpha: float):
    """Generator yielding NDJSON lines: one per chunk + a final done event."""
    total_chunks = len(chunks)
    total_duration = 0.0
    last_sr = 0

    for idx, audio_data, sr, _elapsed in _do_infer_streaming(
        ref_path, chunks, emo_text, emo_audio_path, emo_alpha
    ):
        duration = len(audio_data) / sr
        total_duration += duration
        last_sr = sr
        line = json.dumps({
            "event": "chunk",
            "index": idx,
            "total": total_chunks,
            "audio_base64": _chunk_to_wav_base64(audio_data, sr),
            "sample_rate": sr,
            "duration_seconds": round(duration, 3),
        })
        yield line + "\n"

    done_line = json.dumps({
        "event": "done",
        "total_chunks": total_chunks,
        "total_duration_seconds": round(total_duration, 3),
        "sample_rate": last_sr,
    })
    yield done_line + "\n"


def _build_response(audio_arrays: list[np.ndarray], sample_rate: int,
                    total_inference: float, n_chunks: int) -> dict:
    """Build the standard response dict."""
    full_wav = concat_audio(audio_arrays, sample_rate)
    duration = len(full_wav) / sample_rate
    buf = io.BytesIO()
    sf.write(buf, full_wav, sample_rate, format="WAV")
    return {
        "audio_base64": base64.b64encode(buf.getvalue()).decode(),
        "sample_rate": sample_rate,
        "format": "wav",
        "duration_seconds": round(duration, 2),
        "inference_time": round(total_inference, 2),
        "chunks": n_chunks,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_model, _PRESET_VOICES
    logger.info("Loading IndexTTS2...")
    t0 = time.time()
    from indextts.infer_v2 import IndexTTS2
    tts_model = IndexTTS2(
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        device="cuda:0",
    )
    logger.info("IndexTTS2 loaded in %.1fs", time.time() - t0)
    logger.info("Loading preset voices from %s ...", PRESETS_DIR)
    _PRESET_VOICES = _load_presets()
    logger.info("Loaded %d preset voices", len(_PRESET_VOICES))
    yield
    tts_model = None


app = FastAPI(title="IndexTTS2 API", lifespan=lifespan)


# ─── Request/Response Models ────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    voice_audio_base64: str = Field(..., description="Base64-encoded reference audio for timbre (WAV)")
    emo_text: Optional[str] = Field(default=None, description="Emotion description text")
    emo_audio_base64: Optional[str] = Field(default=None, description="Base64-encoded emotion reference audio")
    emo_alpha: float = Field(default=1.0, ge=0.0, le=2.0, description="Emotion intensity")
    format: str = Field(default="wav", description="Output format")
    stream: bool = Field(default=False, description="Stream chunks as NDJSON lines")


class PresetTTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    voice: str = Field(default="default", description="Preset voice name (default, female_zh, male_zh, female_en, male_en)")
    emo_text: Optional[str] = Field(default=None, description="Emotion description text")
    emo_alpha: float = Field(default=1.0, ge=0.0, le=2.0, description="Emotion intensity")
    stream: bool = Field(default=False, description="Stream chunks as NDJSON lines")


class TTSResponse(BaseModel):
    audio_base64: str
    sample_rate: int
    format: str
    duration_seconds: float
    inference_time: float
    chunks: int = Field(default=1)


# ─── Health & Info ──────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "IndexTTS2",
        "loaded": tts_model is not None,
        "preset_voices": list(_PRESET_VOICES.keys()),
    }


@app.get("/presets")
async def list_presets():
    return {"presets": list(_PRESET_VOICES.keys())}


# ─── Endpoint 1: Preset Voice TTS (no reference audio needed) ──────

@app.post("/v1/tts-preset", response_model=TTSResponse)
async def synthesize_preset(req: PresetTTSRequest):
    """Synthesize speech using a preset voice — no reference audio needed."""
    if tts_model is None:
        raise HTTPException(503, "Model not loaded")

    ref_path = _PRESET_VOICES.get(req.voice)
    if not ref_path:
        raise HTTPException(400,
            f"Unknown preset voice {req.voice}. "
            f"Available: {list(_PRESET_VOICES.keys())}")

    chunks = split_text(req.text)
    if not chunks:
        raise HTTPException(400, "Empty text")

    logger.info("[Preset] voice=%s, %d chunk(s), %d chars, emo_text=%s, stream=%s",
                req.voice, len(chunks), len(req.text),
                f"{req.emo_text}" if req.emo_text else "None",
                req.stream)

    try:
        if req.stream:
            return StreamingResponse(
                _ndjson_stream(ref_path, chunks, req.emo_text, None, req.emo_alpha),
                media_type="application/x-ndjson",
            )
        arrays, sr, t = _do_infer(ref_path, chunks, req.emo_text, None, req.emo_alpha)
        return _build_response(arrays, sr, t, len(chunks))
    except Exception as e:
        logger.exception("IndexTTS2 preset inference failed")
        raise HTTPException(500, f"Inference failed: {e}")


# ─── Endpoint 2: Custom Voice TTS (reference audio required) ───────

@app.post("/v1/tts", response_model=TTSResponse)
async def synthesize(req: TTSRequest):
    """Synthesize speech by cloning a voice from reference audio."""
    if tts_model is None:
        raise HTTPException(503, "Model not loaded")

    try:
        ref_bytes = base64.b64decode(req.voice_audio_base64)
    except Exception:
        raise HTTPException(400, "Invalid base64 voice audio data")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_f:
        ref_f.write(ref_bytes)
        ref_path = ref_f.name

    emo_audio_path = None
    if req.emo_audio_base64:
        try:
            emo_bytes = base64.b64decode(req.emo_audio_base64)
        except Exception:
            os.unlink(ref_path)
            raise HTTPException(400, "Invalid base64 emotion audio data")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as emo_f:
            emo_f.write(emo_bytes)
            emo_audio_path = emo_f.name

    chunks = split_text(req.text)
    if not chunks:
        os.unlink(ref_path)
        if emo_audio_path:
            os.unlink(emo_audio_path)
        raise HTTPException(400, "Empty text")

    logger.info("[Clone] %d chunk(s), %d chars, emo_text=%s, emo_audio=%s, emo_alpha=%.1f, stream=%s",
                len(chunks), len(req.text),
                f"{req.emo_text}" if req.emo_text else "None",
                "yes" if emo_audio_path else "no",
                req.emo_alpha,
                req.stream)

    try:
        if req.stream:
            def _streaming_with_cleanup():
                try:
                    yield from _ndjson_stream(
                        ref_path, chunks, req.emo_text, emo_audio_path, req.emo_alpha
                    )
                finally:
                    for p in [ref_path] + ([emo_audio_path] if emo_audio_path else []):
                        try:
                            os.unlink(p)
                        except OSError:
                            pass

            return StreamingResponse(
                _streaming_with_cleanup(),
                media_type="application/x-ndjson",
            )

        arrays, sr, t = _do_infer(ref_path, chunks, req.emo_text, emo_audio_path, req.emo_alpha)
        return _build_response(arrays, sr, t, len(chunks))
    except Exception as e:
        logger.exception("IndexTTS2 inference failed")
        raise HTTPException(500, f"Inference failed: {e}")
    finally:
        if not req.stream:
            for p in [ref_path] + ([emo_audio_path] if emo_audio_path else []):
                try:
                    os.unlink(p)
                except OSError:
                    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=9093)
