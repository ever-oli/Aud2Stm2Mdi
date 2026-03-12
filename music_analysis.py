from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch


def _scalarize(value):
    if isinstance(value, np.ndarray):
        return value.item() if value.size == 1 else value
    if isinstance(value, np.generic):
        return value.item()
    return value


def to_float(value, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(_scalarize(value)), digits)


def _root_mean_square(data: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(data))))


def _normalized(data: np.ndarray) -> np.ndarray:
    peak = float(np.amax(np.abs(data)))
    if peak == 0:
        return data.astype(np.float32)
    return data.astype(np.float32) / peak


_NEG80POINT8DB = 0.00009120108393559096
_BIT_DEPTH = 16
_DEFAULT_SILENCE_THRESHOLD = (_NEG80POINT8DB * (2 ** (_BIT_DEPTH - 1))) * 4


def _start_of(data: np.ndarray, threshold=_DEFAULT_SILENCE_THRESHOLD, samples_before: int = 1) -> int:
    if int(threshold) != threshold:
        threshold = threshold * float(2 ** (_BIT_DEPTH - 1))
    index = int(np.argmax(np.absolute(data) > threshold))
    if index > (samples_before - 1):
        return index - samples_before
    return 0


def _end_of(data: np.ndarray, threshold=_DEFAULT_SILENCE_THRESHOLD, samples_after: int = 1) -> int:
    if int(threshold) != threshold:
        threshold = threshold * float(2 ** (_BIT_DEPTH - 1))
    rev_index = int(np.argmax(np.flipud(np.absolute(data)) > threshold))
    if rev_index > (samples_after - 1):
        return len(data) - (rev_index - samples_after)
    return len(data)


def _trim_data(
    data: np.ndarray,
    start_threshold=_DEFAULT_SILENCE_THRESHOLD,
    end_threshold=_DEFAULT_SILENCE_THRESHOLD,
) -> np.ndarray:
    start = _start_of(data, start_threshold)
    end = _end_of(data, end_threshold)
    return data[start:end]


def _load_and_trim(file_path: str) -> tuple[np.ndarray, int]:
    y, rate = librosa.load(file_path, mono=True)
    y = _normalized(y)
    trimmed = _trim_data(y)
    return trimmed, rate


def _get_volume(file_path: str) -> tuple[list[float] | None, float | None, float | None]:
    try:
        audio, _ = _load_and_trim(file_path)
        volume = librosa.feature.rms(y=audio)[0]
        avg_volume = float(np.mean(volume))
        loudness = _root_mean_square(audio)
        return volume.tolist(), avg_volume, loudness
    except Exception as exc:  # pragma: no cover - defensive
        sys.stderr.write(f"Failed to get volume and loudness on {file_path}: {exc}\n")
        return None, None, None


def _get_key(freq: float) -> str:
    if freq <= 0:
        return "A0"
    a4 = 440.0
    c0 = a4 * math.pow(2, -4.75)
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    h = round(12 * math.log2(freq / c0))
    octave = h // 12
    n = h % 12
    return names[n] + str(octave)


def _get_average_pitch(pitch_frames: list[list[float]]) -> tuple[float | None, str | None]:
    pitches: list[float] = []
    for timestamp, frequency, confidence in pitch_frames:
        del timestamp
        if confidence > 0.8:
            pitches.append(frequency)
    if not pitches:
        return None, None
    average_frequency = float(np.array(pitches).mean())
    return average_frequency, _get_key(average_frequency)


def _get_intensity(y: np.ndarray, sr: int, beats: np.ndarray) -> np.ndarray:
    cqt = librosa.cqt(y=y, sr=sr, fmin=librosa.note_to_hz("A1"))
    freqs = librosa.cqt_frequencies(cqt.shape[0], fmin=librosa.note_to_hz("A1"))
    perceptual_cqt = librosa.perceptual_weighting(cqt**2, freqs, ref=np.max)
    return librosa.util.sync(perceptual_cqt, beats, aggregate=np.median)


def _get_pitch(y_harmonic: np.ndarray, sr: int, beats: np.ndarray) -> np.ndarray:
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    return librosa.util.sync(chroma, beats, aggregate=np.median)


def _get_timbre(y: np.ndarray, sr: int, beats: np.ndarray) -> np.ndarray:
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_spectrogram, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    stacked = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
    return librosa.util.sync(stacked, beats)


def summarize_sections(
    boundaries: list[float] | np.ndarray | None,
    labels: list[str] | np.ndarray | None,
) -> list[dict[str, object]]:
    if boundaries is None:
        boundaries_list: list[float] = []
    elif hasattr(boundaries, "tolist"):
        boundaries_list = boundaries.tolist()
    else:
        boundaries_list = list(boundaries)

    if labels is None:
        labels_list: list[str] = []
    elif hasattr(labels, "tolist"):
        labels_list = labels.tolist()
    else:
        labels_list = list(labels)

    sections = []
    for idx, label in enumerate(labels_list):
        start = boundaries_list[idx] if idx < len(boundaries_list) else None
        end = boundaries_list[idx + 1] if idx + 1 < len(boundaries_list) else None
        sections.append(
            {
                "index": idx,
                "label": str(label),
                "start": to_float(start),
                "end": to_float(end),
            }
        )
    return sections


def summarize_structure_labels(sections: list[dict[str, object]]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for section in sections:
        label = section.get("label")
        if label is None:
            continue
        label_text = str(label)
        if label_text not in seen:
            labels.append(label_text)
            seen.add(label_text)
    return labels


def canonicalize_section_labels(sections: list[dict[str, object]]) -> tuple[list[str], dict[str, str]]:
    labels = summarize_structure_labels(sections)
    mapping: dict[str, str] = {}
    canonical_labels: list[str] = []
    for idx, label in enumerate(labels):
        lowered = str(label).lower()
        if lowered in {"0", "0.0", "a"}:
            canonical = "section_a"
        elif lowered in {"1", "1.0", "b"}:
            canonical = "section_b"
        elif lowered in {"2", "2.0", "c"}:
            canonical = "section_c"
        else:
            canonical = f"section_{idx + 1}"
        mapping[str(label)] = canonical
        canonical_labels.append(canonical)
    return canonical_labels, mapping


def infer_mood_tags(features: dict[str, object]) -> list[str]:
    tags: list[str] = []
    tempo = features.get("tempo")
    intensity = features.get("intensity")
    timbre = features.get("timbre")

    if isinstance(tempo, (int, float)):
        if tempo < 90:
            tags.append("slow")
        elif tempo < 120:
            tags.append("midtempo")
        else:
            tags.append("upbeat")

    if isinstance(intensity, (int, float)):
        if intensity < 0.25:
            tags.append("soft")
        elif intensity > 0.6:
            tags.append("driving")

    if isinstance(timbre, (int, float)) and timbre > 0:
        tags.append("bright")

    return tags


def _get_pitch_dnn(audio_file: str) -> list[list[float]]:
    try:
        import torchcrepe
    except ImportError as exc:
        raise RuntimeError(
            "torchcrepe is not installed. Install `requirements-context-backends.txt` "
            "to enable pitch/key analysis."
        ) from exc

    sample_rate = 16000
    hop_length = 160
    audio, _ = librosa.load(audio_file, sr=sample_rate, mono=True)
    if audio.size == 0:
        return []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_tensor = torch.tensor(audio, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.inference_mode():
        frequency, confidence = torchcrepe.predict(
            audio_tensor,
            sample_rate,
            hop_length,
            fmin=32.7,
            fmax=1975.5,
            model="tiny",
            batch_size=512,
            device=device,
            return_periodicity=True,
        )

    frequency = frequency.squeeze(0).detach().cpu().numpy()
    confidence = confidence.squeeze(0).detach().cpu().numpy()
    timestamps = np.arange(frequency.shape[0], dtype=np.float32) * (hop_length / sample_rate)

    valid = np.isfinite(frequency) & np.isfinite(confidence) & (frequency > 0)
    if not np.any(valid):
        return []

    return np.column_stack([timestamps[valid], frequency[valid], confidence[valid]]).tolist()


def _get_segments(audio_file: str) -> tuple[list[float] | np.ndarray, list[str] | np.ndarray]:
    try:
        from sf_segmenter.segmenter import Segmenter
    except ImportError as exc:
        raise RuntimeError(
            "sf_segmenter is not installed. Install `requirements-context-backends.txt` "
            "to enable section analysis."
        ) from exc

    segmenter = Segmenter()
    return segmenter.proc_audio(audio_file)


def analyze_audio_file(
    audio_path: str,
    *,
    include_sections: bool = True,
    include_pitch_key: bool = True,
) -> dict[str, object]:
    audio_path = str(Path(audio_path).expanduser().resolve())
    result: dict[str, object] = {
        "status": "ok",
        "audio_path": audio_path,
        "analysis_version": "0.1.0",
        "tempo": None,
        "duration": None,
        "frequency": None,
        "key": None,
        "timbre": None,
        "pitch": None,
        "intensity": None,
        "avg_volume": None,
        "loudness": None,
        "segment_count": 0,
        "sections": [],
        "structure_labels": [],
        "section_label_map": {},
        "mood_tags": [],
        "backends": {
            "librosa": "ok",
            "torchcrepe": "disabled" if not include_pitch_key else "pending",
            "sf_segmenter": "disabled" if not include_sections else "pending",
        },
        "issues": [],
    }

    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        result["duration"] = to_float(librosa.get_duration(y=y, sr=sr), digits=3)

        y_harmonic, y_percussive = librosa.effects.hpss(y=y)
        tempo, beats = librosa.beat.beat_track(
            sr=sr,
            onset_envelope=librosa.onset.onset_strength(y=y_percussive, sr=sr),
            trim=False,
        )
        result["tempo"] = to_float(tempo, digits=2)

        beats = np.asarray(beats)
        if beats.size > 0:
            intensity_frames = np.matrix(_get_intensity(y, sr, beats)).getT()
            pitch_frames = np.matrix(_get_pitch(y_harmonic, sr, beats)).getT()
            timbre_frames = np.matrix(_get_timbre(y, sr, beats)).getT()
            result["intensity"] = to_float(np.mean(intensity_frames))
            result["pitch"] = to_float(np.mean(pitch_frames))
            result["timbre"] = to_float(np.mean(timbre_frames))
        else:
            result["issues"].append("No beats detected for beat-synchronous timbre/pitch/intensity summary.")

        _, avg_volume, loudness = _get_volume(audio_path)
        result["avg_volume"] = to_float(avg_volume, digits=6)
        result["loudness"] = to_float(loudness, digits=6)
    except Exception as exc:
        result["status"] = "error"
        result["issues"].append(str(exc))
        return result

    if include_pitch_key:
        try:
            frequency_frames = _get_pitch_dnn(audio_path)
            average_frequency, average_key = _get_average_pitch(frequency_frames)
            result["frequency"] = to_float(average_frequency, digits=2)
            result["key"] = average_key
            result["backends"]["torchcrepe"] = "ok"
        except Exception as exc:
            result["backends"]["torchcrepe"] = "unavailable"
            result["issues"].append(str(exc))

    if include_sections:
        try:
            boundaries, labels = _get_segments(audio_path)
            sections = summarize_sections(boundaries, labels)
            structure_labels, section_label_map = canonicalize_section_labels(sections)
            result["sections"] = sections
            result["segment_count"] = len(sections)
            result["structure_labels"] = structure_labels
            result["section_label_map"] = section_label_map
            result["backends"]["sf_segmenter"] = "ok"
        except Exception as exc:
            result["backends"]["sf_segmenter"] = "unavailable"
            result["issues"].append(str(exc))

    result["mood_tags"] = infer_mood_tags(result)
    if result["issues"] and result["status"] == "ok":
        result["status"] = "partial"
    return result


def build_retrieval_hints(
    *,
    item_id: str,
    analysis: dict[str, object] | None,
    source_audio_path: str | None,
    lyrics_excerpt: str | None = None,
) -> dict[str, object]:
    analysis = analysis or {}
    text_queries: list[str] = []
    if analysis.get("key") and analysis.get("tempo") is not None:
        text_queries.append(f"{analysis['key']} tonal music around {analysis['tempo']} bpm")
    structure_labels = analysis.get("structure_labels") or []
    if structure_labels:
        text_queries.append("structured song with sections " + ", ".join(structure_labels[:4]))
    if lyrics_excerpt:
        text_queries.append(lyrics_excerpt[:160])
    return {
        "item_id": item_id,
        "audio_reference_paths": [path for path in [source_audio_path] if path and os.path.isfile(path)],
        "text_queries": text_queries,
        "mood_tags": analysis.get("mood_tags") or [],
        "structure_labels": structure_labels,
    }
