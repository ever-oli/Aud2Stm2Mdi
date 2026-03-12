from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AmtModelSpec:
    key: str
    display_name: str
    backend: str
    description: str
    supports_basic_pitch_controls: bool = False
    mt3_model_name: str | None = None


DEFAULT_AMT_MODEL = "basic_pitch"


AMT_MODELS: dict[str, AmtModelSpec] = {
    "basic_pitch": AmtModelSpec(
        key="basic_pitch",
        display_name="Basic Pitch / Spotify",
        backend="basic_pitch",
        description="Fast, lightweight audio-to-MIDI transcription with tunable note thresholds.",
        supports_basic_pitch_controls=True,
    ),
    "mt3": AmtModelSpec(
        key="mt3",
        display_name="MT3 / MR-MT3",
        backend="mt3",
        description="Heavier multi-instrument transcription via mt3-infer using the MR-MT3 checkpoint.",
        mt3_model_name="mr_mt3",
    ),
}


def get_amt_model_spec(model_name: str) -> AmtModelSpec:
    try:
        return AMT_MODELS[model_name]
    except KeyError as exc:
        available = ", ".join(sorted(AMT_MODELS))
        raise ValueError(f"Unknown AMT model '{model_name}'. Available models: {available}") from exc


def list_amt_dropdown_choices() -> list[tuple[str, str]]:
    return [
        (spec.display_name, spec.key)
        for spec in AMT_MODELS.values()
    ]


def list_amt_model_specs() -> list[AmtModelSpec]:
    return list(AMT_MODELS.values())


def resolve_amt_model_names(requested_models: list[str] | None) -> list[str]:
    if not requested_models or requested_models == ["all"]:
        return list(AMT_MODELS)

    resolved: list[str] = []
    unknown: list[str] = []
    for model_name in requested_models:
        if model_name in AMT_MODELS:
            resolved.append(model_name)
        else:
            unknown.append(model_name)

    if unknown:
        available = ", ".join(sorted(AMT_MODELS))
        missing = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown AMT models: {missing}. Available models: {available}")

    return resolved


def get_amt_model_help_text(model_name: str) -> str:
    spec = get_amt_model_spec(model_name)
    if spec.supports_basic_pitch_controls:
        return (
            f"Using **{spec.display_name}**. The MIDI sliders below are active for this backend."
        )
    return (
        f"Using **{spec.display_name}**. The Basic Pitch sliders below are ignored for this backend."
    )
