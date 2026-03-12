import os
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

import app


class _FakeProcessor:
    def __init__(self):
        self.sources = ("drums", "bass", "other", "vocals")

    def separate_to_dir(self, audio_path: str, output_dir: str):
        audio_np, sample_rate = sf.read(audio_path, always_2d=True)
        waveform = torch.from_numpy(audio_np.T.astype(np.float32))
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        stems = {
            "drums": waveform * 0.1,
            "bass": waveform * 0.2,
            "other": waveform * 0.3,
            "vocals": waveform * 0.4,
        }
        stem_paths = {}
        for stem_name, stem in stems.items():
            output_path = Path(output_dir) / f"{stem_name}.wav"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(
                output_path,
                stem.detach().cpu().numpy().T.astype("float32"),
                sample_rate,
            )
            stem_paths[stem_name] = output_path
        return stem_paths


class _FakeAmtProcessor:
    def __init__(self):
        self.calls = []

    def convert_to_midi(self, audio_path: str, output_path: str, **kwargs):
        self.calls.append((audio_path, output_path, kwargs))
        Path(output_path).write_bytes(b"MThd")
        return output_path


class _ProgressRecorder:
    def __init__(self):
        self.events = []

    def __call__(self, value, desc=None):
        self.events.append((value, desc))


def _write_sample_wav(directory: str) -> str:
    sample_rate = 44100
    seconds = 0.25
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    audio = 0.2 * np.sin(2 * np.pi * 440 * t)
    path = Path(directory) / "sample.wav"
    sf.write(path, audio.astype(np.float32), sample_rate)
    return str(path)


class AppProcessAudioTests(unittest.TestCase):
    def test_runtime_directories_are_configured(self):
        self.assertTrue(Path(app.RUNTIME_ROOT).is_dir())
        self.assertTrue(Path(app.RUNTIME_CACHE_DIR).is_dir())
        self.assertTrue(Path(app.RUNTIME_MPLCONFIG_DIR).is_dir())
        self.assertTrue(Path(app.RUNTIME_TMP_DIR).is_dir())
        self.assertEqual(os.environ["TMPDIR"], str(app.RUNTIME_TMP_DIR))
        self.assertEqual(os.environ["MPLCONFIGDIR"], str(app.RUNTIME_MPLCONFIG_DIR))

    def test_get_stem_choices_expands_for_six_stem_model(self):
        stems, default_stem = app.get_stem_choices("htdemucs_6s")

        self.assertEqual(default_stem, "vocals")
        self.assertEqual(
            stems,
            ["vocals", "drums", "bass", "other", "guitar", "piano"],
        )

    def test_get_stem_choices_for_external_model(self):
        stems, default_stem = app.get_stem_choices("msst_scnet")

        self.assertEqual(default_stem, "vocals")
        self.assertEqual(stems, ["drums", "bass", "other", "vocals"])

    def test_amt_help_text_reflects_selected_model(self):
        basic_pitch_help = app.get_amt_model_help_text("basic_pitch")
        mt3_help = app.get_amt_model_help_text("mt3")
        yourmt3_help = app.get_amt_model_help_text("yourmt3")

        self.assertIn("Basic Pitch", basic_pitch_help)
        self.assertIn("sliders below are active", basic_pitch_help)
        self.assertIn("MT3", mt3_help)
        self.assertIn("ignored", mt3_help)
        self.assertIn("YourMT3", yourmt3_help)

    def test_process_audio_without_midi_returns_audio_and_hides_roll(self):
        fake_processor = _FakeProcessor()
        progress = _ProgressRecorder()

        with TemporaryDirectory() as tmpdir:
            sample_path = _write_sample_wav(tmpdir)
            with patch.object(app, "OUTPUT_DIR", Path(tmpdir)), patch.object(
                app, "get_processor", return_value=fake_processor
            ) as get_processor:
                audio_out, midi_path, manifest_path, lyrics_path, piano_roll = app.process_audio_path(
                    sample_path,
                    stem_type="other",
                    target_bpm=0,
                    convert_midi=False,
                    onset_threshold=0.5,
                    frame_threshold=0.4,
                    min_note_length=150,
                    multiple_pitch_bends=False,
                    separator_model="mdx",
                    progress=progress,
                )
                manifest_exists = Path(manifest_path).is_file()

        get_processor.assert_called_once_with("mdx")

        sample_rate, samples = audio_out
        self.assertEqual(sample_rate, 44100)
        self.assertEqual(samples.dtype, np.int16)
        self.assertIsNone(midi_path)
        self.assertTrue(manifest_exists)
        self.assertIsNone(lyrics_path)
        self.assertFalse(piano_roll["visible"])
        self.assertEqual(progress.events[-1][0], 1.0)

    def test_process_audio_with_invalid_stem_for_model_raises_error(self):
        fake_processor = _FakeProcessor()
        progress = _ProgressRecorder()

        with TemporaryDirectory() as tmpdir:
            sample_path = _write_sample_wav(tmpdir)
            with patch.object(app, "OUTPUT_DIR", Path(tmpdir)), patch.object(
                app, "get_processor", return_value=fake_processor
            ):
                with self.assertRaises(app.gr.Error) as cm:
                    app.process_audio_path(
                        sample_path,
                        stem_type="guitar",
                        target_bpm=0,
                        convert_midi=False,
                        onset_threshold=0.5,
                        frame_threshold=0.4,
                        min_note_length=150,
                        multiple_pitch_bends=False,
                        separator_model="htdemucs",
                        progress=progress,
                    )

        self.assertIn("Available stems", str(cm.exception))

    def test_process_audio_with_midi_returns_artifacts(self):
        fake_processor = _FakeProcessor()
        fake_amt_processor = _FakeAmtProcessor()
        progress = _ProgressRecorder()
        expected_roll = np.zeros((12, 24, 3), dtype=np.uint8)

        with TemporaryDirectory() as tmpdir:
            sample_path = _write_sample_wav(tmpdir)
            with patch.object(app, "OUTPUT_DIR", Path(tmpdir)), patch.object(
                app, "get_processor", return_value=fake_processor
            ), patch.object(
                app, "get_amt_processor", return_value=fake_amt_processor
            ), patch.object(
                app, "render_piano_roll", return_value=expected_roll
            ):
                audio_out, midi_path, manifest_path, lyrics_path, piano_roll = app.process_audio_path(
                    sample_path,
                    stem_type="other",
                    target_bpm=0,
                    convert_midi=True,
                    onset_threshold=0.55,
                    frame_threshold=0.45,
                    min_note_length=180,
                    multiple_pitch_bends=True,
                    separator_model="htdemucs_ft",
                    amt_model="basic_pitch",
                    progress=progress,
                )
                manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

        sample_rate, samples = audio_out
        self.assertEqual(sample_rate, 44100)
        self.assertEqual(samples.dtype, np.int16)
        self.assertIsInstance(midi_path, str)
        self.assertTrue(midi_path.endswith(".mid"))
        self.assertTrue(fake_amt_processor.calls)
        _, _, amt_kwargs = fake_amt_processor.calls[0]
        self.assertEqual(amt_kwargs["onset_threshold"], 0.55)
        self.assertEqual(amt_kwargs["frame_threshold"], 0.45)
        self.assertEqual(amt_kwargs["minimum_note_length"], 180)
        self.assertTrue(amt_kwargs["multiple_pitch_bends"])
        self.assertTrue(piano_roll["visible"])
        self.assertTrue(np.array_equal(piano_roll["value"], expected_roll))
        self.assertEqual(manifest["transcription"]["model"], "basic_pitch")
        self.assertEqual(manifest["config"]["separator_model"], "htdemucs_ft")
        self.assertIsNone(lyrics_path)
        self.assertEqual(progress.events[-1][0], 1.0)

    def test_process_audio_with_mt3_uses_selected_amt_model(self):
        fake_processor = _FakeProcessor()
        fake_amt_processor = _FakeAmtProcessor()
        progress = _ProgressRecorder()
        expected_roll = np.zeros((8, 16, 3), dtype=np.uint8)

        with TemporaryDirectory() as tmpdir:
            sample_path = _write_sample_wav(tmpdir)
            with patch.object(app, "OUTPUT_DIR", Path(tmpdir)), patch.object(
                app, "get_processor", return_value=fake_processor
            ), patch.object(
                app, "get_amt_processor", return_value=fake_amt_processor
            ) as get_amt_processor, patch.object(
                app, "render_piano_roll", return_value=expected_roll
            ):
                _, midi_path, manifest_path, _, piano_roll = app.process_audio_path(
                    sample_path,
                    stem_type="vocals",
                    target_bpm=0,
                    convert_midi=True,
                    onset_threshold=0.5,
                    frame_threshold=0.4,
                    min_note_length=150,
                    multiple_pitch_bends=False,
                    separator_model="mdx",
                    amt_model="yourmt3",
                    progress=progress,
                )
                manifest_exists = Path(manifest_path).is_file()

        get_amt_processor.assert_called_once_with("yourmt3")
        self.assertTrue(midi_path.endswith("_yourmt3.mid"))
        self.assertTrue(manifest_exists)
        self.assertTrue(fake_amt_processor.calls)
        self.assertTrue(piano_roll["visible"])

    def test_process_audio_with_analysis_and_lyrics_writes_optional_artifacts(self):
        fake_processor = _FakeProcessor()
        fake_amt_processor = _FakeAmtProcessor()
        progress = _ProgressRecorder()
        expected_roll = np.zeros((8, 16, 3), dtype=np.uint8)
        fake_analysis = {
            "status": "ok",
            "tempo": 120.0,
            "key": "A4",
            "sections": [
                {"index": 0, "label": "A", "start": 0.0, "end": 1.0},
            ],
            "structure_labels": ["section_a"],
            "mood_tags": ["midtempo"],
        }

        def _fake_lyrics(audio_path, output_path, **kwargs):
            del audio_path, kwargs
            Path(output_path).write_text("{}", encoding="utf-8")
            return {
                "status": "ok",
                "output_path": str(output_path),
                "normalized_excerpt": "hello world",
                "aligned_sections": [{"index": 0, "text": "hello world"}],
            }

        with TemporaryDirectory() as tmpdir:
            sample_path = _write_sample_wav(tmpdir)
            with patch.object(app, "OUTPUT_DIR", Path(tmpdir)), patch.object(
                app, "get_processor", return_value=fake_processor
            ), patch.object(
                app, "get_amt_processor", return_value=fake_amt_processor
            ), patch.object(
                app, "render_piano_roll", return_value=expected_roll
            ), patch.object(
                app, "run_optional_analysis", return_value=fake_analysis
            ) as analysis_mock, patch.object(
                app, "run_optional_lyrics", side_effect=_fake_lyrics
            ) as lyrics_mock:
                _, _, manifest_path, lyrics_path, _ = app.process_audio_path(
                    sample_path,
                    stem_type="vocals",
                    target_bpm=0,
                    convert_midi=True,
                    onset_threshold=0.5,
                    frame_threshold=0.4,
                    min_note_length=150,
                    multiple_pitch_bends=False,
                    separator_model="mdx",
                    amt_model="mt3_pytorch",
                    analyze_audio_metadata=True,
                    transcribe_lyrics=True,
                    progress=progress,
                )
                lyrics_exists = Path(lyrics_path).is_file()
                manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

        self.assertEqual(analysis_mock.call_count, 2)
        lyrics_mock.assert_called_once()
        self.assertTrue(lyrics_exists)
        self.assertTrue(manifest["analysis"]["enabled"])
        self.assertEqual(manifest["lyrics"]["status"], "ok")
        self.assertEqual(manifest["config"]["amt_model"], "mt3_pytorch")


if __name__ == "__main__":
    unittest.main()
