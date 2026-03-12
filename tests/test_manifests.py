import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from clap_retrieval import collect_audio_assets_from_manifest, collect_audio_assets_from_root
from run_manifest import build_run_manifest, write_run_manifest


class ManifestTests(unittest.TestCase):
    def test_build_and_write_run_manifest(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_audio = tmp_path / "source.wav"
            stem_audio = tmp_path / "vocals.wav"
            midi_file = tmp_path / "vocals.mid"
            source_audio.write_bytes(b"audio")
            stem_audio.write_bytes(b"audio")
            midi_file.write_bytes(b"MThd")

            manifest = build_run_manifest(
                run_type="app",
                source_audio_path=source_audio,
                status="ok",
                config={"separator_model": "mdx", "amt_model": "basic_pitch"},
                separator={"status": "ok", "model": "mdx"},
                transcription={"status": "ok", "model": "basic_pitch"},
                analysis={
                    "enabled": True,
                    "status": "ok",
                    "source_audio": {"status": "ok", "tempo": 120.0, "structure_labels": ["section_a"], "mood_tags": ["midtempo"]},
                    "target_audio": {"status": "ok", "tempo": 120.0, "structure_labels": ["section_a"], "mood_tags": ["midtempo"]},
                    "stems": [{"stem": "vocals", "analysis": {"status": "ok", "tempo": 120.0}}],
                },
                references={
                    "source_audio": str(source_audio),
                    "selected_stem": "vocals",
                    "stems": [{"stem": "vocals", "path": str(stem_audio)}],
                    "midi": [{"kind": "midi", "path": str(midi_file)}],
                },
            )
            manifest_path = tmp_path / "run_manifest.json"
            write_run_manifest(manifest_path, manifest)

            parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(parsed["status"], "ok")
            self.assertEqual(parsed["separator"]["model"], "mdx")
            self.assertEqual(parsed["references"]["selected_stem"], "vocals")
            self.assertEqual(parsed["retrieval"]["item_id"].split(":")[-1], "source")

    def test_collect_audio_assets_from_manifest(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_audio = tmp_path / "source.wav"
            stem_audio = tmp_path / "vocals.wav"
            source_audio.write_bytes(b"audio")
            stem_audio.write_bytes(b"audio")

            manifest = build_run_manifest(
                run_type="separator_sweep",
                source_audio_path=source_audio,
                status="ok",
                config={"separator_model": "mdx"},
                analysis={
                    "enabled": True,
                    "status": "ok",
                    "source_audio": {"status": "ok", "tempo": 120.0, "key": "A4", "structure_labels": ["section_a"], "mood_tags": ["midtempo"]},
                    "target_audio": {"status": "ok"},
                    "stems": [
                        {"stem": "vocals", "analysis": {"status": "ok", "tempo": 121.0, "key": "A4", "structure_labels": ["section_a"], "mood_tags": ["midtempo"]}},
                    ],
                },
                references={
                    "source_audio": str(source_audio),
                    "selected_stem": "vocals",
                    "stems": [{"stem": "vocals", "path": str(stem_audio)}],
                },
            )
            manifest_path = tmp_path / "run_manifest.json"
            write_run_manifest(manifest_path, manifest)

            assets = collect_audio_assets_from_manifest(manifest_path)
            self.assertEqual(len(assets), 2)
            stem_asset = next(item for item in assets if item["kind"] == "stem")
            self.assertEqual(stem_asset["stem_role"], "vocals")
            self.assertEqual(stem_asset["tempo"], 121.0)

    def test_collect_audio_assets_from_root_dedupes_by_path(self):
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_audio = tmp_path / "source.wav"
            stem_audio = tmp_path / "vocals.wav"
            source_audio.write_bytes(b"audio")
            stem_audio.write_bytes(b"audio")

            for idx in range(2):
                run_dir = tmp_path / f"run-{idx}"
                run_dir.mkdir()
                manifest = build_run_manifest(
                    run_type="full_pipeline_sweep",
                    source_audio_path=source_audio,
                    status="ok",
                    config={"separator_model": "mdx", "amt_model": f"mt3-{idx}"},
                    references={
                        "source_audio": str(source_audio),
                        "selected_stem": "vocals",
                        "stems": [{"stem": "vocals", "path": str(stem_audio)}],
                    },
                )
                write_run_manifest(run_dir / "run_manifest.json", manifest)

            assets = collect_audio_assets_from_root(tmp_path)
            paths = sorted(item["path"] for item in assets)
            self.assertEqual(paths, [str(source_audio.resolve()), str(stem_audio.resolve())])


if __name__ == "__main__":
    unittest.main()
