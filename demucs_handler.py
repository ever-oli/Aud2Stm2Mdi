import torch
import torchaudio
import logging
import numpy as np
import soundfile as sf
from pathlib import Path
from demucs.pretrained import get_model
from demucs.apply import apply_model
from typing import Tuple

from demucs_models import DEFAULT_DEMUCS_MODEL, get_demucs_model_spec

logger = logging.getLogger(__name__)


class DemucsProcessor:
    def __init__(self, model_name: str = DEFAULT_DEMUCS_MODEL):
        self.model_name = get_demucs_model_spec(model_name).name
        # Device priority: CUDA → Apple MPS → CPU
        # PYTORCH_ENABLE_MPS_FALLBACK=1 must be set before torch import so that
        # any MPS-unsupported ops automatically fall back to CPU instead of crashing.
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"[Demucs] device: {self.device}")

        self.model = get_model(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self.sources = tuple(self.model.sources)

        print(
            f"[Demucs] model '{self.model_name}' ready  "
            f"| sources: {self.model.sources}  "
            f"| native sr: {self.model.samplerate} Hz"
        )

    # ------------------------------------------------------------------
    def separate_stems(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        Separate *audio_path* into stems.

        Returns
        -------
        sources : Tensor  shape (1, n_stems, 2, time)  at model.samplerate
        sample_rate : int  the model's native sample rate (44 100 Hz for htdemucs)
        """
        # Use soundfile to load audio — avoids the TorchCodec dependency that
        # torchaudio.load() triggers on Apple Silicon with torchaudio >= 2.0.
        audio_np, file_sr = sf.read(audio_path, always_2d=True)  # (samples, channels)
        audio_np = audio_np.T.astype(np.float32)                  # (channels, samples)
        waveform = torch.from_numpy(audio_np)

        # Resample to model's native sample rate if necessary
        if file_sr != self.model.samplerate:
            waveform = torchaudio.functional.resample(waveform, file_sr, self.model.samplerate)

        print(f"[Demucs] loaded  shape={waveform.shape}  sr={file_sr}->{self.model.samplerate}")

        # Ensure 2-D (channels, time)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Duplicate mono to stereo — htdemucs expects 2 channels
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
            print("[Demucs] mono → stereo (channel duplication)")

        # Add batch dim → (1, 2, time)
        waveform = waveform.unsqueeze(0)
        print(f"[Demucs] inference input shape: {waveform.shape}")

        with torch.no_grad():
            sources = apply_model(self.model, waveform.to(self.device))

        print(f"[Demucs] output shape: {sources.shape}  stems: {self.model.sources}")
        # Return the model's native sample rate — the tensor is already at that rate
        return sources, self.model.samplerate

    # ------------------------------------------------------------------
    def save_stem(
        self,
        stem: torch.Tensor,
        stem_name: str,
        output_dir: str,
    ) -> Path:
        """Save *stem* (shape: 2, time) as a WAV at the model's sample rate."""
        out = Path(output_dir) / f"{stem_name}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        # Use soundfile to save — avoids the TorchCodec dependency in
        # torchaudio.save() on Apple Silicon with torchaudio >= 2.0.
        # stem shape: (2, time) — soundfile expects (time, channels).
        audio_np = stem.cpu().numpy().T.astype("float32")
        sf.write(str(out), audio_np, self.model.samplerate)
        print(f"[Demucs] stem saved → {out}")
        return out

    def separate_to_dir(self, audio_path: str, output_dir: str) -> dict[str, Path]:
        sources, _ = self.separate_stems(audio_path)
        stem_paths: dict[str, Path] = {}
        for index, stem_name in enumerate(self.sources):
            stem_paths[stem_name] = self.save_stem(sources[0, index], stem_name, output_dir)
        return stem_paths
