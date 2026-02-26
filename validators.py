import os
import logging
import torchaudio
from typing import Tuple

logger = logging.getLogger(__name__)


class AudioValidator:
    SUPPORTED_FORMATS = [".mp3", ".wav", ".flac"]
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB (raised from 30 MB)

    @staticmethod
    def validate_audio_file(file_path: str) -> Tuple[bool, str]:
        try:
            if not os.path.exists(file_path):
                return False, "File does not exist"

            size = os.path.getsize(file_path)
            limit_mb = AudioValidator.MAX_FILE_SIZE // 1024 // 1024
            if size > AudioValidator.MAX_FILE_SIZE:
                return False, f"File exceeds {limit_mb} MB limit"

            ext = os.path.splitext(file_path)[1].lower()
            if ext not in AudioValidator.SUPPORTED_FORMATS:
                return (
                    False,
                    f"Unsupported format '{ext}'. "
                    f"Accepted: {', '.join(AudioValidator.SUPPORTED_FORMATS)}",
                )

            try:
                _, sample_rate = torchaudio.load(file_path)
                # Only reject extremely low sample rates; high-res (96 kHz, 192 kHz)
                # is fine — Demucs resamples internally.
                if sample_rate < 8000:
                    return False, f"Sample rate too low ({sample_rate} Hz)"
            except Exception as exc:
                return False, f"Cannot read audio file: {exc}"

            return True, "OK"

        except Exception as exc:
            logger.error("Validation error: %s", exc)
            return False, str(exc)
