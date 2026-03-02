import os
import uuid
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

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
from demucs_handler import DemucsProcessor    # noqa: E402
from basic_pitch_handler import BasicPitchConverter  # noqa: E402

# ── Output directory ─────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/tmp/audio_processor")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Singleton model instances (loaded once, reused across requests) ──────────
_processor: Optional[DemucsProcessor] = None
_converter: Optional[BasicPitchConverter] = None


def get_processor() -> DemucsProcessor:
    global _processor
    if _processor is None:
        _processor = DemucsProcessor()
    return _processor


def get_converter() -> BasicPitchConverter:
    global _converter
    if _converter is None:
        _converter = BasicPitchConverter()
    return _converter


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
    progress=gr.Progress(track_tqdm=True),
):
    """
    Gradio Blocks handler.

    Inputs (must match the order in run_btn.click(inputs=[...])):
        audio_file, stem_type, target_bpm, convert_midi,
        onset_threshold, frame_threshold, min_note_length, multiple_pitch_bends

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
        progress(0.05, desc="Loading Demucs model…")
        processor = get_processor()

        progress(0.10, desc="Separating stems (this takes ~30-90 s on first run)…")
        sources, sr = processor.separate_stems(file_path)

        # Use model.sources for robust stem index lookup
        stem_index    = processor.model.sources.index(stem_type)
        selected_stem = sources[0, stem_index]  # shape: (2, time)

        processor.save_stem(selected_stem, stem_type, str(work_dir))
        stem_path = work_dir / f"{stem_type}.wav"

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
        progress(0.60, desc="Running Basic Pitch (TFLite inference)…")
        converter = get_converter()
        converter.set_process_options(
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=min_note_length,
            multiple_pitch_bends=multiple_pitch_bends,
        )

        midi_path = work_dir / f"{stem_type}.mid"
        converter.convert_to_midi(str(stem_path), str(midi_path))

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
    with gr.Blocks(
        title="Aud2Stm2Mdi",
        theme=gr.themes.Base(primary_hue="indigo", neutral_hue="slate"),
    ) as demo:

        gr.Markdown(
            "## Aud2Stm2Mdi\n"
            "Separate audio into stems with **Demucs** `htdemucs`, "
            "then transcribe to **MIDI** with **Basic Pitch**."
        )

        with gr.Row():

            # ── Left column: controls ─────────────────────────────────────
            with gr.Column(scale=1, min_width=300):

                audio_input = gr.File(
                    label="Audio File  (.mp3 / .wav / .flac)",
                    file_types=[".mp3", ".wav", ".flac"],
                )
                stem_dd = gr.Dropdown(
                    choices=["vocals", "drums", "bass", "other"],
                    value="vocals",
                    label="Stem to extract",
                )
                midi_cb = gr.Checkbox(label="Convert to MIDI", value=True)

                with gr.Accordion("BPM Quantization (Polymath Core)", open=False):
                    bpm_sl = gr.Slider(
                        0, 200, value=0, step=1,
                        label="Target BPM",
                        info="Time-stretches the stem so MIDI falls perfectly on the beat grid. Set to 0 to disable."
                    )

                with gr.Accordion("MIDI Parameters", open=False):
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
            ],
            outputs=[stem_audio, midi_file, piano_roll],
        )

        test_btn.click(
            fn=process_audio_path,
            inputs=[
                test_dd, stem_dd, bpm_sl, midi_cb,
                onset_sl, frame_sl, minlen_sl, bends_cb,
            ],
            outputs=[stem_audio, midi_file, piano_roll],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load both models eagerly at startup so the first request doesn't pay
    # the full model-load penalty.
    print("Loading models at startup…")
    get_processor()
    get_converter()
    print("Models ready — launching server.")

    build_interface().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        allowed_paths=[str(OUTPUT_DIR)],
    )
