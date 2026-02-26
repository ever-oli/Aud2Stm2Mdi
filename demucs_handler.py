import torch
import torchaudio
import logging
from pathlib import Path
from demucs.pretrained import get_model
from demucs.apply import apply_model
from typing import Tuple

logger = logging.getLogger(__name__)


class DemucsProcessor:
    def __init__(self, model_name: str = "htdemucs"):
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

        self.model = get_model(model_name)
        self.model.to(self.device)
        self.model.eval()

        print(
            f"[Demucs] model '{model_name}' ready  "
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
        waveform, _ = torchaudio.load(audio_path)
        print(f"[Demucs] loaded  shape={waveform.shape}")

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
        torchaudio.save(str(out), stem.cpu(), self.model.samplerate)
        print(f"[Demucs] stem saved → {out}")
        return out
