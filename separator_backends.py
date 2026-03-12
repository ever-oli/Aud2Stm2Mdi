import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from demucs_handler import DemucsProcessor
from separator_registry import DEFAULT_SEPARATOR_MODEL, get_separator_model_spec


PROJECT_ROOT = Path(__file__).resolve().parent
SEPARATOR_RUNTIME_ROOT = PROJECT_ROOT / ".tmp" / "separator_runtime"
SEPARATOR_CACHE_DIR = SEPARATOR_RUNTIME_ROOT / "cache"
SEPARATOR_TORCH_HOME = SEPARATOR_CACHE_DIR / "torch"
SEPARATOR_MPLCONFIG_DIR = SEPARATOR_RUNTIME_ROOT / "mplconfig"
SEPARATOR_TMP_DIR = SEPARATOR_RUNTIME_ROOT / "tmp"
SEPARATOR_DOWNLOADS_DIR = SEPARATOR_RUNTIME_ROOT / "downloads"

for runtime_dir in (
    SEPARATOR_RUNTIME_ROOT,
    SEPARATOR_CACHE_DIR,
    SEPARATOR_TORCH_HOME,
    SEPARATOR_MPLCONFIG_DIR,
    SEPARATOR_TMP_DIR,
    SEPARATOR_DOWNLOADS_DIR,
):
    runtime_dir.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("XDG_CACHE_HOME", str(SEPARATOR_CACHE_DIR))
os.environ.setdefault("TORCH_HOME", str(SEPARATOR_TORCH_HOME))
os.environ.setdefault("MPLCONFIGDIR", str(SEPARATOR_MPLCONFIG_DIR))
os.environ.setdefault("TMPDIR", str(SEPARATOR_TMP_DIR))
tempfile.tempdir = str(SEPARATOR_TMP_DIR)


def get_msst_repo_dir() -> Path:
    repo_dir = Path(
        os.environ.get(
            "AUD2STM2MDI_MSST_REPO_DIR",
            str(PROJECT_ROOT / ".tmp" / "Music-Source-Separation-Training"),
        )
    ).expanduser()
    if not repo_dir.is_dir():
        raise RuntimeError(
            "Music-Source-Separation-Training repo not found. "
            "Clone it to "
            f"'{repo_dir}' or set AUD2STM2MDI_MSST_REPO_DIR."
        )
    inference_path = repo_dir / "inference.py"
    if not inference_path.is_file():
        raise RuntimeError(f"Missing inference entrypoint: {inference_path}")
    return repo_dir


def build_backend_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("XDG_CACHE_HOME", str(SEPARATOR_CACHE_DIR))
    env.setdefault("TORCH_HOME", str(SEPARATOR_TORCH_HOME))
    env.setdefault("MPLCONFIGDIR", str(SEPARATOR_MPLCONFIG_DIR))
    env.setdefault("TMPDIR", str(SEPARATOR_TMP_DIR))
    env.setdefault("WANDB_MODE", "disabled")
    return env


def _download_to_cache(url: str, destination: Path) -> Path:
    if destination.is_file():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as out_file:
            shutil.copyfileobj(response, out_file)
        tmp_path.replace(destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return destination


class MSSTProcessor:
    def __init__(self, model_name: str):
        self.spec = get_separator_model_spec(model_name)
        if self.spec.backend != "msst":
            raise ValueError(f"Model '{model_name}' is not an MSST backend")

        self.model_name = self.spec.key
        self.sources = self.spec.sources
        self.repo_dir = get_msst_repo_dir()
        self.download_dir = SEPARATOR_DOWNLOADS_DIR / self.model_name
        self.download_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_artifacts(self) -> tuple[Path, Path]:
        if not self.spec.config_url or not self.spec.weights_url or not self.spec.model_type:
            raise RuntimeError(f"Incomplete MSST spec for {self.model_name}")

        config_path = self.download_dir / Path(self.spec.config_url).name
        weights_path = self.download_dir / Path(self.spec.weights_url).name

        _download_to_cache(self.spec.config_url, config_path)
        _download_to_cache(self.spec.weights_url, weights_path)
        return config_path, weights_path

    def _run_inference(
        self,
        input_dir: Path,
        store_dir: Path,
        config_path: Path,
        weights_path: Path,
        *,
        force_cpu: bool,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [
            sys.executable,
            str(self.repo_dir / "inference.py"),
            "--model_type",
            self.spec.model_type,
            "--config_path",
            str(config_path),
            "--start_check_point",
            str(weights_path),
            "--input_folder",
            str(input_dir),
            "--store_dir",
            str(store_dir),
        ]
        if force_cpu:
            cmd.append("--force_cpu")

        return subprocess.run(
            cmd,
            cwd=self.repo_dir,
            env=build_backend_env(),
            capture_output=True,
            text=True,
        )

    def separate_to_dir(self, audio_path: str, output_dir: str) -> dict[str, Path]:
        output_root = Path(output_dir)
        cached_outputs = {
            stem: output_root / f"{stem}.wav"
            for stem in self.sources
            if (output_root / f"{stem}.wav").is_file()
        }
        if len(cached_outputs) == len(self.sources):
            return cached_outputs

        config_path, weights_path = self._ensure_artifacts()
        input_dir = output_root / "_msst_input"
        shutil.rmtree(input_dir, ignore_errors=True)
        input_dir.mkdir(parents=True, exist_ok=True)

        source_audio = Path(audio_path).resolve()
        input_audio = input_dir / source_audio.name
        if input_audio.exists() or input_audio.is_symlink():
            input_audio.unlink()
        try:
            input_audio.symlink_to(source_audio)
        except OSError:
            shutil.copy2(source_audio, input_audio)

        store_dir = output_root / "_msst_output"
        shutil.rmtree(store_dir, ignore_errors=True)
        store_dir.mkdir(parents=True, exist_ok=True)

        force_cpu = os.environ.get("AUD2STM2MDI_FORCE_CPU") == "1"
        result = self._run_inference(
            input_dir,
            store_dir,
            config_path,
            weights_path,
            force_cpu=force_cpu,
        )
        if result.returncode != 0 and not force_cpu:
            combined_output = f"{result.stdout}\n{result.stderr}"
            if (
                "Using device:  mps" in combined_output
                or "Using device: mps" in combined_output
                or "MPS backend out of memory" in combined_output
            ):
                result = self._run_inference(
                    input_dir,
                    store_dir,
                    config_path,
                    weights_path,
                    force_cpu=True,
                )

        if result.returncode != 0:
            raise RuntimeError(
                f"MSST inference failed for '{self.model_name}'.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        output_subdir = store_dir / source_audio.stem
        stem_paths: dict[str, Path] = {}
        for stem_name in self.sources:
            source_path = output_subdir / f"{stem_name}.wav"
            if not source_path.is_file():
                continue
            final_path = output_root / f"{stem_name}.wav"
            shutil.copy2(source_path, final_path)
            stem_paths[stem_name] = final_path

        if len(stem_paths) == 0:
            raise RuntimeError(
                f"No stems were produced for model '{self.model_name}'. "
                f"Expected outputs in {output_subdir}"
            )

        return stem_paths


def create_separator_processor(model_name: str):
    spec = get_separator_model_spec(model_name)
    if spec.backend == "demucs":
        return DemucsProcessor(model_name=model_name)
    if spec.backend == "msst":
        return MSSTProcessor(model_name=model_name)
    raise ValueError(f"Unsupported backend '{spec.backend}' for model '{model_name}'")


def get_default_separator_model() -> str:
    return DEFAULT_SEPARATOR_MODEL
