from __future__ import annotations

import os
import tempfile
import inspect
from pathlib import Path

import librosa
import torch

from amt_registry import get_amt_model_spec
from basic_pitch_handler import BasicPitchConverter

PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = Path(
    os.environ.get("AUD2STM2MDI_RUNTIME_ROOT", PROJECT_ROOT / ".tmp" / "amt_runtime")
)
RUNTIME_CACHE_DIR = RUNTIME_ROOT / "cache"
RUNTIME_MPLCONFIG_DIR = RUNTIME_ROOT / "mplconfig"
RUNTIME_TMP_DIR = RUNTIME_ROOT / "tmp"
RUNTIME_TORCH_HOME = RUNTIME_CACHE_DIR / "torch"
RUNTIME_HF_HOME = RUNTIME_CACHE_DIR / "huggingface"
RUNTIME_MT3_CHECKPOINT_DIR = PROJECT_ROOT / ".mt3_checkpoints"

for runtime_dir in (
    RUNTIME_ROOT,
    RUNTIME_CACHE_DIR,
    RUNTIME_MPLCONFIG_DIR,
    RUNTIME_TMP_DIR,
    RUNTIME_TORCH_HOME,
    RUNTIME_HF_HOME,
    RUNTIME_MT3_CHECKPOINT_DIR,
):
    runtime_dir.mkdir(parents=True, exist_ok=True)


def _configure_runtime_path(var_name: str, fallback: Path, *, force: bool = False) -> Path:
    current = os.environ.get(var_name)
    current_path = Path(current) if current else None
    if force or current_path is None or not current_path.exists() or not os.access(current_path, os.W_OK):
        os.environ[var_name] = str(fallback)
        return fallback
    return current_path


_configure_runtime_path("XDG_CACHE_HOME", RUNTIME_CACHE_DIR)
_configure_runtime_path("MPLCONFIGDIR", RUNTIME_MPLCONFIG_DIR)
_configure_runtime_path("TMPDIR", RUNTIME_TMP_DIR)
_configure_runtime_path("TORCH_HOME", RUNTIME_TORCH_HOME)
_configure_runtime_path("HF_HOME", RUNTIME_HF_HOME)
_configure_runtime_path("MT3_CHECKPOINT_DIR", RUNTIME_MT3_CHECKPOINT_DIR)
tempfile.tempdir = str(RUNTIME_TMP_DIR)

_MT3_COMPAT_PATCHED = False


def _apply_mt3_compat_patches() -> None:
    global _MT3_COMPAT_PATCHED
    if _MT3_COMPAT_PATCHED:
        return

    from transformers.modeling_utils import PreTrainedModel
    from transformers.models.t5 import modeling_t5

    original_mask_fn = PreTrainedModel.get_extended_attention_mask
    if not getattr(original_mask_fn, "_aud2stm2mdi_compat", False):
        def compat_get_extended_attention_mask(self, attention_mask, input_shape, dtype=None):
            if isinstance(dtype, torch.device):
                dtype = self.dtype
            return original_mask_fn(self, attention_mask, input_shape, dtype=dtype)

        compat_get_extended_attention_mask._aud2stm2mdi_compat = True
        PreTrainedModel.get_extended_attention_mask = compat_get_extended_attention_mask

    original_t5_block_forward = modeling_t5.T5Block.forward
    if not getattr(original_t5_block_forward, "_aud2stm2mdi_compat", False):
        original_signature = inspect.signature(original_t5_block_forward)
        supported_params = set(original_signature.parameters)

        def compat_t5_block_forward(
            self,
            *args,
            past_key_values=None,
            cache_position=None,
            **kwargs,
        ):
            if past_key_values is not None:
                if "past_key_values" in supported_params:
                    kwargs.setdefault("past_key_values", past_key_values)
                elif "past_key_value" in supported_params:
                    kwargs.setdefault("past_key_value", past_key_values)
            if cache_position is not None and "cache_position" in supported_params:
                kwargs.setdefault("cache_position", cache_position)
            return original_t5_block_forward(self, *args, **kwargs)

        compat_t5_block_forward._aud2stm2mdi_compat = True
        modeling_t5.T5Block.forward = compat_t5_block_forward

    _MT3_COMPAT_PATCHED = True


class BasicPitchAmtProcessor:
    def __init__(self, model_name: str):
        self.spec = get_amt_model_spec(model_name)
        self._converter = BasicPitchConverter()

    def convert_to_midi(
        self,
        audio_path: str,
        output_path: str,
        *,
        onset_threshold: float,
        frame_threshold: float,
        minimum_note_length: float,
        multiple_pitch_bends: bool,
    ) -> str:
        # Basic Pitch's CoreML/TFLite path is sensitive to TMPDIR and related
        # cache env vars when invoked outside the Gradio app bootstrap.
        os.environ["TMPDIR"] = str(RUNTIME_TMP_DIR)
        os.environ["MPLCONFIGDIR"] = str(RUNTIME_MPLCONFIG_DIR)
        os.environ["XDG_CACHE_HOME"] = str(RUNTIME_CACHE_DIR)
        tempfile.tempdir = str(RUNTIME_TMP_DIR)
        self._converter.set_process_options(
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length,
            multiple_pitch_bends=multiple_pitch_bends,
        )
        return self._converter.convert_to_midi(audio_path, output_path)


class MT3AmtProcessor:
    def __init__(self, model_name: str):
        self.spec = get_amt_model_spec(model_name)

    def convert_to_midi(
        self,
        audio_path: str,
        output_path: str,
        *,
        onset_threshold: float,
        frame_threshold: float,
        minimum_note_length: float,
        multiple_pitch_bends: bool,
    ) -> str:
        del onset_threshold, frame_threshold, minimum_note_length, multiple_pitch_bends

        try:
            from mt3_infer import transcribe
        except ImportError as exc:
            raise RuntimeError(
                f"{self.spec.display_name} is not installed. Install it with "
                "`pip install -r requirements.txt -r requirements-amt-backends.txt`."
            ) from exc
        _apply_mt3_compat_patches()

        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        midi = transcribe(audio, sr=16000, model=self.spec.mt3_model_name or "mr_mt3")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        midi.save(str(output))
        return str(output)


def create_amt_processor(model_name: str):
    spec = get_amt_model_spec(model_name)
    if spec.backend == "basic_pitch":
        return BasicPitchAmtProcessor(model_name)
    if spec.backend == "mt3":
        return MT3AmtProcessor(model_name)
    raise ValueError(f"Unsupported AMT backend '{spec.backend}' for model '{model_name}'")
