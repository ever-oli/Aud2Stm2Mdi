import logging
from basic_pitch.inference import predict
import pretty_midi
from typing import Optional

logger = logging.getLogger(__name__)


class BasicPitchConverter:
    """
    Thin wrapper around Spotify's Basic Pitch for audio → MIDI transcription.

    Default parameter choices are tuned for clean import into a DAW (Ableton etc.):
    - ``multiple_pitch_bends=False``  →  one pitch-bend channel per instrument track,
      which DAWs handle predictably.  Set True only if you need polyphonic bend data.
    - ``frame_threshold=0.4``         →  slightly higher than the library default (0.3)
      to reduce "legato smear" where sustained frames bleed into adjacent notes.
    - ``minimum_note_length=150 ms``  →  filters ghost / glitch notes below 150 ms.
    """

    def __init__(self):
        self.process_options: dict = {
            "onset_threshold":    0.5,
            "frame_threshold":    0.4,    # 0.3 default → raised to reduce smear
            "minimum_note_length": 150.0, # ms — was 127.7 ms
            "minimum_frequency":  32.7,   # C1
            "maximum_frequency":  2093.0, # C7
            "multiple_pitch_bends": False, # True → messy in most DAWs
            "melodia_trick":      True,
            "midi_tempo":         120.0,
        }
        print("[BasicPitch] converter ready")

    # ------------------------------------------------------------------
    def convert_to_midi(self, audio_path: str, output_path: str) -> str:
        """
        Run Basic Pitch inference on *audio_path* and write a MIDI file to
        *output_path*.  Returns *output_path* on success.
        """
        print(f"[BasicPitch] converting  {audio_path}")

        # basic_pitch.inference.predict returns (model_output, PrettyMIDI, note_events)
        _, midi_data, _ = predict(
            audio_path=audio_path,
            onset_threshold=self.process_options["onset_threshold"],
            frame_threshold=self.process_options["frame_threshold"],
            minimum_note_length=self.process_options["minimum_note_length"],
            minimum_frequency=self.process_options["minimum_frequency"],
            maximum_frequency=self.process_options["maximum_frequency"],
            multiple_pitch_bends=self.process_options["multiple_pitch_bends"],
            melodia_trick=self.process_options["melodia_trick"],
            midi_tempo=self.process_options["midi_tempo"],
        )

        if not isinstance(midi_data, pretty_midi.PrettyMIDI):
            raise ValueError("Basic Pitch returned unexpected MIDI data type")

        midi_data.write(output_path)
        n_notes = sum(len(i.notes) for i in midi_data.instruments)
        print(f"[BasicPitch] saved {n_notes} notes → {output_path}")
        return output_path

    # ------------------------------------------------------------------
    def set_process_options(self, **kwargs) -> None:
        """Override any processing parameter at runtime (called from the UI)."""
        self.process_options.update(kwargs)
