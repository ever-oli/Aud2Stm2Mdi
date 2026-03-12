import os
import uuid
import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

# Keep caches and temporary compilation artifacts in writable paths so the app
# works reliably in sandboxes and hosted runtimes.
RUNTIME_ROOT = Path(os.environ.get("AUD2STM2MDI_RUNTIME_ROOT", "/tmp/audio_processor"))
RUNTIME_ENV_DIR = RUNTIME_ROOT / ".runtime"
RUNTIME_CACHE_DIR = RUNTIME_ENV_DIR / "cache"
RUNTIME_MPLCONFIG_DIR = RUNTIME_ENV_DIR / "mplconfig"
RUNTIME_TMP_DIR = RUNTIME_ENV_DIR / "tmp"
RUNTIME_TORCH_HOME = RUNTIME_CACHE_DIR / "torch"

for runtime_dir in (
    RUNTIME_ROOT,
    RUNTIME_ENV_DIR,
    RUNTIME_CACHE_DIR,
    RUNTIME_MPLCONFIG_DIR,
    RUNTIME_TMP_DIR,
    RUNTIME_TORCH_HOME,
):
    runtime_dir.mkdir(parents=True, exist_ok=True)


def _configure_runtime_path(var_name: str, fallback: Path, *, force: bool = False) -> Path:
    current = os.environ.get(var_name)
    current_path = Path(current) if current else None
    if force or current_path is None or not current_path.exists() or not os.access(current_path, os.W_OK):
        os.environ[var_name] = str(fallback)
        return fallback
    return current_path


RUNTIME_CACHE_DIR = _configure_runtime_path("XDG_CACHE_HOME", RUNTIME_CACHE_DIR)
RUNTIME_MPLCONFIG_DIR = _configure_runtime_path("MPLCONFIGDIR", RUNTIME_MPLCONFIG_DIR)
RUNTIME_TMP_DIR = _configure_runtime_path("TMPDIR", RUNTIME_TMP_DIR, force=True)
RUNTIME_TORCH_HOME = _configure_runtime_path("TORCH_HOME", RUNTIME_CACHE_DIR / "torch")
RUNTIME_TORCH_HOME.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(RUNTIME_TMP_DIR)

# Non-interactive Matplotlib backend — must be set before pyplot is imported
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import librosa
import pyrubberband as pyrb

import pretty_midi
import gradio as gr

# ── Environment variables ────────────────────────────────────────────────────
# Suppress verbose TF / Metal logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Allow PyTorch MPS to fall back to CPU for any unsupported ops instead of
# raising an error.  Must be set before torch is imported (which happens
# inside demucs_handler).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# torchaudio is imported inside the handlers; audio loading is done via
# soundfile directly (TorchCodec is not available on Apple Silicon).

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from validators import AudioValidator          # noqa: E402
from separator_backends import create_separator_processor  # noqa: E402
from separator_registry import (  # noqa: E402
    DEFAULT_SEPARATOR_MODEL,
    get_default_stem_for_model,
    get_separator_model_sources,
    get_separator_model_spec,
    list_separator_dropdown_choices,
)
from amt_backends import create_amt_processor  # noqa: E402
from amt_registry import (  # noqa: E402
    DEFAULT_AMT_MODEL,
    get_amt_model_help_text,
    get_amt_model_spec,
    list_amt_dropdown_choices,
)

# ── Output directory ─────────────────────────────────────────────────────────
OUTPUT_DIR = RUNTIME_ROOT
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Singleton model instances (loaded once, reused across requests) ──────────
_processors: dict[str, object] = {}
_amt_processors: dict[str, object] = {}


def get_processor(model_name: str = DEFAULT_SEPARATOR_MODEL):
    processor = _processors.get(model_name)
    if processor is None:
        processor = create_separator_processor(model_name)
        _processors[model_name] = processor
    return processor


def get_amt_processor(model_name: str = DEFAULT_AMT_MODEL):
    processor = _amt_processors.get(model_name)
    if processor is None:
        processor = create_amt_processor(model_name)
        _amt_processors[model_name] = processor
    return processor


def get_stem_choices(model_name: str) -> tuple[list[str], str]:
    stems = get_separator_model_sources(model_name)
    return stems, get_default_stem_for_model(model_name)


def update_stem_dropdown(model_name: str):
    stems, default_stem = get_stem_choices(model_name)
    return gr.update(choices=stems, value=default_stem)


def update_amt_help_text(model_name: str):
    return gr.update(value=get_amt_model_help_text(model_name))


# ── Piano roll renderer ───────────────────────────────────────────────────────
_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_BLACK_KEYS = {1, 3, 6, 8, 10}


def render_piano_roll(midi_path: str) -> np.ndarray:
    """
    Render a dark-themed piano-roll image from a MIDI file.

    Returns an (H, W, 3) uint8 RGB array suitable for gr.Image(type='numpy').
    """
    midi = pretty_midi.PrettyMIDI(midi_path)
    all_notes = [note for inst in midi.instruments for note in inst.notes]

    fig, ax = plt.subplots(figsize=(18, 6), dpi=100, facecolor="#0d1117")
    ax.set_facecolor("#161b22")

    if not all_notes:
        ax.text(
            0.5, 0.5,
            "No notes detected — try lowering Onset or Frame threshold",
            transform=ax.transAxes,
            color="#8b949e", ha="center", va="center", fontsize=12,
        )
    else:
        t_max = max(n.end for n in all_notes)
        p_min = max(0,   min(n.pitch for n in all_notes) - 3)
        p_max = min(127, max(n.pitch for n in all_notes) + 3)

        # ── Black-key shading ────────────────────────────────────────────
        for p in range(p_min, p_max + 1):
            if p % 12 in _BLACK_KEYS:
                ax.axhspan(p - 0.5, p + 0.5, alpha=0.08, color="white", linewidth=0)

        # ── Octave separator lines ────────────────────────────────────────
        for p in range(p_min, p_max + 1):
            if p % 12 == 0:
                ax.axhline(p - 0.5, color="#21262d", linewidth=0.8, zorder=1)

        # ── Notes, colour-coded by instrument track ───────────────────────
        n_inst = max(1, len(midi.instruments))
        cmap   = plt.cm.cool
        for i, inst in enumerate(midi.instruments):
            colour = cmap(i / n_inst)
            for note in inst.notes:
                dur   = max(note.end - note.start, 0.015)   # minimum visual width
                alpha = 0.45 + 0.55 * (note.velocity / 127.0)
                ax.add_patch(
                    patches.FancyBboxPatch(
                        (note.start, note.pitch - 0.45), dur, 0.90,
                        boxstyle="round,pad=0.01",
                        linewidth=0,
                        facecolor=colour,
                        alpha=alpha,
                        zorder=2,
                    )
                )

        ax.set_xlim(0, t_max)
        ax.set_ylim(p_min - 1, p_max + 1)

        # Y-axis: label only C notes (one per octave)
        c_ticks = [p for p in range(p_min, p_max + 1) if p % 12 == 0]
        ax.set_yticks(c_ticks)
        ax.set_yticklabels(
            [f"{_NOTE_NAMES[p % 12]}{p // 12 - 1}" for p in c_ticks],
            color="#8b949e", fontsize=8,
        )
        ax.tick_params(axis="x", colors="#8b949e", labelsize=8)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Time (s)", color="#8b949e", fontsize=9)

        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

        n = len(all_notes)
        ax.set_title(
            f"Piano Roll  ·  {n} note{'s' if n != 1 else ''}  ·  {t_max:.1f} s",
            color="#e6edf3", fontsize=11, pad=8,
        )

    plt.tight_layout(pad=0.4)
    fig.canvas.draw()
    # buffer_rgba() → RGBA array; drop alpha channel for gr.Image
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return img


# ── Core processing function ──────────────────────────────────────────────────
def process_audio(
    audio_file,
    stem_type: str,
    target_bpm: float,
    convert_midi: bool,
    onset_threshold: float,
    frame_threshold: float,
    min_note_length: float,
    multiple_pitch_bends: bool,
    separator_model: str = DEFAULT_SEPARATOR_MODEL,
    amt_model: str = DEFAULT_AMT_MODEL,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Gradio Blocks handler.

    Inputs (must match the order in run_btn.click(inputs=[...])):
        audio_file, stem_type, target_bpm, convert_midi,
        onset_threshold, frame_threshold, min_note_length,
        multiple_pitch_bends, separator_model, amt_model

    Outputs  →  [stem_audio, midi_file, piano_roll]
    """
    # ── Validate input ────────────────────────────────────────────────────
    if audio_file is None:
        raise gr.Error("Upload an audio file first.")

    # gr.File may return a string path or an object with a .name attribute
    file_path = audio_file.name if hasattr(audio_file, "name") else str(audio_file)

    valid, msg = AudioValidator.validate_audio_file(file_path)
    if not valid:
        raise gr.Error(f"File validation failed: {msg}")

    # ── Work directory (UUID prevents collisions) ─────────────────────────
    work_dir = OUTPUT_DIR / uuid.uuid4().hex
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── Stage 1: stem separation ──────────────────────────────────────
        spec = get_separator_model_spec(separator_model)
        progress(0.05, desc=f"Loading separator '{spec.display_name}'…")
        processor = get_processor(separator_model)

        progress(0.10, desc="Separating stems (this takes ~30-90 s on first run)…")
        stem_paths = processor.separate_to_dir(file_path, str(work_dir))

        if stem_type not in stem_paths:
            available = ", ".join(processor.sources)
            raise gr.Error(
                f"Stem '{stem_type}' is not available for separator '{spec.display_name}'. "
                f"Available stems: {available}"
            )
        stem_path = Path(stem_paths[stem_type])

        progress(0.55, desc="Stem extracted.")

        # Read stem back for processing and preview (mono int16)
        y, orig_sr = librosa.load(str(stem_path), sr=None)
        
        # ── Polymath Integration: BPM Quantization ───────────────────────
        # If Target BPM > 0, we time-stretch the audio to perfectly align 
        # to that exact grid before MIDI conversion. This ensures the output
        # MIDI notes lock onto the piano roll.
        if target_bpm > 0:
            progress(0.56, desc=f"Quantizing stem to {target_bpm} BPM…")
            
            # Extract harmonic/percussive and find beats
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            tempo, beats = librosa.beat.beat_track(
                sr=orig_sr, 
                onset_envelope=librosa.onset.onset_strength(y=y_percussive, sr=orig_sr), 
                trim=False
            )
            beat_frames = librosa.frames_to_samples(beats)
            
            # Generate target metronome map
            fixed_beat_times = [i * 120 / target_bpm for i in range(len(beat_frames))]
            fixed_beat_frames = librosa.time_to_samples(fixed_beat_times)
            
            # Construct time map for pyrubberband
            time_map = list(zip(beat_frames, fixed_beat_frames))
            
            # Handle the ending clip length
            if len(beat_frames) > 0 and len(y) > beat_frames[-1]:
                orig_end_diff = len(y) - beat_frames[-1]
                # tempo is an ndarray, so we extract the scalar float for math
                tempo_val = tempo[0] if isinstance(tempo, np.ndarray) else tempo
                new_ending = int(round(fixed_beat_frames[-1] + orig_end_diff * (tempo_val / target_bpm)))
                time_map.append((len(y), new_ending))
            
            # Time-stretch
            y = pyrb.timemap_stretch(y, orig_sr, time_map)
            # Re-save the stretched stem to use for Basic Pitch
            sf.write(str(stem_path), y, orig_sr)
            progress(0.59, desc="Quantization complete.")

        # Preview Audio formatting
        if y.ndim > 1:
            y = y.mean(axis=1)
        audio_out = (orig_sr, (y * 32767).astype(np.int16))

        # ── Early exit if MIDI not requested ─────────────────────────────
        if not convert_midi:
            progress(1.0, desc="Done.")
            return audio_out, None, gr.update(value=None, visible=False)

        # ── Stage 2: MIDI conversion ──────────────────────────────────────
        amt_spec = get_amt_model_spec(amt_model)
        progress(0.60, desc=f"Running {amt_spec.display_name}…")
        amt_processor = get_amt_processor(amt_model)

        midi_path = work_dir / f"{stem_type}_{amt_model}.mid"
        amt_processor.convert_to_midi(
            str(stem_path),
            str(midi_path),
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=min_note_length,
            multiple_pitch_bends=multiple_pitch_bends,
        )

        # ── Stage 3: piano roll render ────────────────────────────────────
        progress(0.90, desc="Rendering piano roll…")
        roll_img = render_piano_roll(str(midi_path))

        progress(1.0, desc="Done.")
        return audio_out, str(midi_path), gr.update(value=roll_img, visible=True)

    except gr.Error:
        raise
    except Exception as exc:
        logger.exception("Processing failed")
        raise gr.Error(str(exc)) from exc


# ── Direct-path processing (used by Quick Test UI) ───────────────────────────
def process_audio_path(
    file_path: str,
    stem_type: str,
    target_bpm: float,
    convert_midi: bool,
    onset_threshold: float,
    frame_threshold: float,
    min_note_length: float,
    multiple_pitch_bends: bool,
    separator_model: str = DEFAULT_SEPARATOR_MODEL,
    amt_model: str = DEFAULT_AMT_MODEL,
    progress=gr.Progress(track_tqdm=True),
):
    """Same as process_audio but accepts a plain file-system path string."""

    class _FakePath:
        def __init__(self, p):
            self.name = p

    return process_audio(
        _FakePath(file_path),
        stem_type, target_bpm, convert_midi,
        onset_threshold, frame_threshold, min_note_length, multiple_pitch_bends,
        separator_model,
        amt_model,
        progress,
    )


# Discover any audio files in the repo's mp3/ folder for the Quick Test picker
_MP3_DIR = Path(__file__).parent / "mp3"

def get_test_files():
    if not _MP3_DIR.is_dir():
        return []
    return sorted(
        str(p) for p in _MP3_DIR.glob("*")
        if p.suffix.lower() in (".mp3", ".wav", ".flac")
    )

# ── Gradio Blocks UI ──────────────────────────────────────────────────────────
def build_interface() -> gr.Blocks:
    default_stem_choices, default_stem = get_stem_choices(DEFAULT_SEPARATOR_MODEL)
    model_choices = list_separator_dropdown_choices()
    amt_model_choices = list_amt_dropdown_choices()
    default_amt_help = get_amt_model_help_text(DEFAULT_AMT_MODEL)

    with gr.Blocks(
        title="Aud2Stm2Mdi",
        theme=gr.themes.Base(primary_hue="indigo", neutral_hue="slate"),
    ) as demo:

        gr.Markdown(
            "## Aud2Stm2Mdi\n"
            "Separate audio into stems with **Demucs** plus experimental "
            "**RoFormer / SCNet / MDX23C** backends, "
            "then transcribe to **MIDI** with **Basic Pitch** or **MT3**."
        )

        with gr.Row():

            # ── Left column: controls ─────────────────────────────────────
            with gr.Column(scale=1, min_width=300):

                audio_input = gr.File(
                    label="Audio File  (.mp3 / .wav / .flac)",
                    file_types=[".mp3", ".wav", ".flac"],
                )
                model_dd = gr.Dropdown(
                    choices=model_choices,
                    value=DEFAULT_SEPARATOR_MODEL,
                    label="Separator model",
                    info="Demucs checkpoints plus selected MSST RoFormer / SCNet / MDX23C models.",
                )
                stem_dd = gr.Dropdown(
                    choices=default_stem_choices,
                    value=default_stem,
                    label="Stem to extract",
                )
                midi_cb = gr.Checkbox(label="Convert to MIDI", value=True)
                amt_model_dd = gr.Dropdown(
                    choices=amt_model_choices,
                    value=DEFAULT_AMT_MODEL,
                    label="MIDI transcription model",
                    info="Basic Pitch is lightweight. MT3 is heavier but better suited to multi-instrument transcription.",
                )
                amt_help = gr.Markdown(default_amt_help)

                with gr.Accordion("BPM Quantization (Polymath Core)", open=False):
                    bpm_sl = gr.Slider(
                        0, 200, value=0, step=1,
                        label="Target BPM",
                        info="Time-stretches the stem so MIDI falls perfectly on the beat grid. Set to 0 to disable."
                    )

                with gr.Accordion("Basic Pitch Parameters", open=False):
                    onset_sl = gr.Slider(
                        0.10, 0.95, value=0.50, step=0.05,
                        label="Onset Threshold",
                        info="Higher → fewer but more confident note onsets",
                    )
                    frame_sl = gr.Slider(
                        0.10, 0.95, value=0.40, step=0.05,
                        label="Frame Threshold",
                        info="Higher → shorter notes, less legato smear",
                    )
                    minlen_sl = gr.Slider(
                        50, 500, value=150, step=10,
                        label="Min Note Length (ms)",
                        info="Increase to filter ghost / glitch notes",
                    )
                    bends_cb = gr.Checkbox(
                        label="Multiple Pitch Bends",
                        value=False,
                        info="Keep OFF for cleaner Ableton import",
                    )

                run_btn = gr.Button("Process", variant="primary", size="lg", elem_id="run_btn")

                test_files = get_test_files()
                with gr.Accordion("🧪 Quick Test (pre-loaded files)", open=bool(test_files), elem_id="quick_test", visible=bool(test_files)):
                    test_dd = gr.Dropdown(
                        choices=test_files if test_files else ["No files found"],
                        value=test_files[0] if test_files else None,
                        label="Select test file",
                        elem_id="test_file_dd",
                    )
                    test_btn = gr.Button(
                        "Run Quick Test", variant="secondary", size="sm",
                        elem_id="test_btn",
                    )

            # ── Right column: results ─────────────────────────────────────
            with gr.Column(scale=2):
                stem_audio = gr.Audio(label="Separated Stem", type="numpy")
                midi_file  = gr.File(label="MIDI Download")
                piano_roll = gr.Image(
                    label="Piano Roll Preview",
                    type="numpy",
                    visible=False,   # hidden until MIDI is produced
                )

        run_btn.click(
            fn=process_audio,
            inputs=[
                audio_input, stem_dd, bpm_sl, midi_cb,
                onset_sl, frame_sl, minlen_sl, bends_cb,
                model_dd,
                amt_model_dd,
            ],
            outputs=[stem_audio, midi_file, piano_roll],
        )

        test_btn.click(
            fn=process_audio_path,
            inputs=[
                test_dd, stem_dd, bpm_sl, midi_cb,
                onset_sl, frame_sl, minlen_sl, bends_cb,
                model_dd,
                amt_model_dd,
            ],
            outputs=[stem_audio, midi_file, piano_roll],
        )

        model_dd.change(
            fn=update_stem_dropdown,
            inputs=[model_dd],
            outputs=[stem_dd],
        )

        amt_model_dd.change(
            fn=update_amt_help_text,
            inputs=[amt_model_dd],
            outputs=[amt_help],
        )

    return demo


    # ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load the default separator and MIDI model eagerly so the first request
    # doesn't pay the full model-load penalty.
    print("Loading models at startup…")
    get_processor(DEFAULT_SEPARATOR_MODEL)
    get_amt_processor(DEFAULT_AMT_MODEL)
    print("Models ready — launching server.")

    build_interface().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        allowed_paths=[str(OUTPUT_DIR)],
    )
