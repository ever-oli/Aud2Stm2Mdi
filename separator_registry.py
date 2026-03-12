from dataclasses import dataclass

from demucs_models import DEMUCS_MODEL_SPECS


@dataclass(frozen=True)
class SeparatorModelSpec:
    key: str
    backend: str
    family: str
    name: str
    description: str
    sources: tuple[str, ...]
    model_type: str | None = None
    config_url: str | None = None
    weights_url: str | None = None

    @property
    def display_name(self) -> str:
        return f"{self.family} / {self.name}"


DEFAULT_SEPARATOR_MODEL = "htdemucs"


SEPARATOR_MODEL_SPECS: dict[str, SeparatorModelSpec] = {
    key: SeparatorModelSpec(
        key=key,
        backend="demucs",
        family="Demucs",
        name=spec.name,
        description=spec.description,
        sources=spec.sources,
    )
    for key, spec in DEMUCS_MODEL_SPECS.items()
}


SEPARATOR_MODEL_SPECS.update(
    {
        "msst_bs_roformer": SeparatorModelSpec(
            key="msst_bs_roformer",
            backend="msst",
            family="RoFormer",
            name="BS RoFormer",
            description="ZFTurbo MSST BS RoFormer MUSDB all-stems checkpoint.",
            sources=("vocals", "bass", "drums", "other"),
            model_type="bs_roformer",
            config_url="https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/config_bs_roformer_384_8_2_485100.yaml",
            weights_url="https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.12/model_bs_roformer_ep_17_sdr_9.6568.ckpt",
        ),
        "msst_scnet": SeparatorModelSpec(
            key="msst_scnet",
            backend="msst",
            family="SCNet",
            name="SCNet Small",
            description="ZFTurbo MSST SCNet Small MUSDB all-stems checkpoint.",
            sources=("drums", "bass", "other", "vocals"),
            model_type="scnet",
            config_url="https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/config_musdb18_scnet.yaml",
            weights_url="https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v.1.0.6/scnet_checkpoint_musdb18.ckpt",
        ),
        "msst_mdx23c": SeparatorModelSpec(
            key="msst_mdx23c",
            backend="msst",
            family="MDX23C",
            name="MDX23C",
            description="ZFTurbo MSST MDX23C MUSDB all-stems checkpoint.",
            sources=("vocals", "bass", "drums", "other"),
            model_type="mdx23c",
            config_url="https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.1/config_musdb18_mdx23c.yaml",
            weights_url="https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.1/model_mdx23c_ep_168_sdr_7.0207.ckpt",
        ),
    }
)


def list_separator_model_specs() -> list[SeparatorModelSpec]:
    return list(SEPARATOR_MODEL_SPECS.values())


def list_separator_dropdown_choices() -> list[tuple[str, str]]:
    return [(spec.display_name, spec.key) for spec in list_separator_model_specs()]


def get_separator_model_spec(model_name: str) -> SeparatorModelSpec:
    try:
        return SEPARATOR_MODEL_SPECS[model_name]
    except KeyError as exc:
        available = ", ".join(SEPARATOR_MODEL_SPECS)
        raise ValueError(f"Unsupported separator model '{model_name}'. Available: {available}") from exc


def get_separator_model_sources(model_name: str) -> list[str]:
    return list(get_separator_model_spec(model_name).sources)


def get_default_stem_for_model(model_name: str) -> str:
    sources = get_separator_model_sources(model_name)
    return "vocals" if "vocals" in sources else sources[0]


def resolve_separator_model_names(model_names: list[str] | None) -> list[str]:
    if not model_names:
        return list(SEPARATOR_MODEL_SPECS)

    if len(model_names) == 1 and model_names[0] == "all":
        return list(SEPARATOR_MODEL_SPECS)

    resolved = []
    seen = set()
    for model_name in model_names:
        spec = get_separator_model_spec(model_name)
        if spec.key not in seen:
            resolved.append(spec.key)
            seen.add(spec.key)
    return resolved
