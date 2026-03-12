from dataclasses import dataclass


@dataclass(frozen=True)
class DemucsModelSpec:
    name: str
    description: str
    sources: tuple[str, ...]


DEFAULT_DEMUCS_MODEL = "htdemucs"


DEMUCS_MODEL_SPECS: dict[str, DemucsModelSpec] = {
    "htdemucs": DemucsModelSpec(
        name="htdemucs",
        description="Hybrid Transformer Demucs, 4 stems.",
        sources=("vocals", "drums", "bass", "other"),
    ),
    "htdemucs_ft": DemucsModelSpec(
        name="htdemucs_ft",
        description="Fine-tuned Hybrid Transformer Demucs, 4 stems.",
        sources=("vocals", "drums", "bass", "other"),
    ),
    "htdemucs_6s": DemucsModelSpec(
        name="htdemucs_6s",
        description="Experimental Hybrid Transformer Demucs, 6 stems.",
        sources=("vocals", "drums", "bass", "other", "guitar", "piano"),
    ),
    "hdemucs_mmi": DemucsModelSpec(
        name="hdemucs_mmi",
        description="Original Hybrid Demucs bag-of-models release, 4 stems.",
        sources=("vocals", "drums", "bass", "other"),
    ),
    "mdx": DemucsModelSpec(
        name="mdx",
        description="MDX challenge Hybrid Demucs, 4 stems.",
        sources=("vocals", "drums", "bass", "other"),
    ),
    "mdx_extra": DemucsModelSpec(
        name="mdx_extra",
        description="Fine-tuned MDX challenge Hybrid Demucs, 4 stems.",
        sources=("vocals", "drums", "bass", "other"),
    ),
    "mdx_q": DemucsModelSpec(
        name="mdx_q",
        description="Quantized MDX challenge Hybrid Demucs, 4 stems.",
        sources=("vocals", "drums", "bass", "other"),
    ),
    "mdx_extra_q": DemucsModelSpec(
        name="mdx_extra_q",
        description="Quantized fine-tuned MDX challenge Hybrid Demucs, 4 stems.",
        sources=("vocals", "drums", "bass", "other"),
    ),
}


def list_demucs_model_specs() -> list[DemucsModelSpec]:
    return list(DEMUCS_MODEL_SPECS.values())


def get_demucs_model_spec(model_name: str) -> DemucsModelSpec:
    try:
        return DEMUCS_MODEL_SPECS[model_name]
    except KeyError as exc:
        available = ", ".join(DEMUCS_MODEL_SPECS)
        raise ValueError(f"Unsupported Demucs model '{model_name}'. Available: {available}") from exc


def get_demucs_model_sources(model_name: str) -> list[str]:
    return list(get_demucs_model_spec(model_name).sources)


def get_default_stem_for_model(model_name: str) -> str:
    sources = get_demucs_model_sources(model_name)
    return "vocals" if "vocals" in sources else sources[0]


def resolve_demucs_model_names(model_names: list[str] | None) -> list[str]:
    if not model_names:
        return list(DEMUCS_MODEL_SPECS)

    if len(model_names) == 1 and model_names[0] == "all":
        return list(DEMUCS_MODEL_SPECS)

    resolved = []
    seen = set()
    for model_name in model_names:
        spec = get_demucs_model_spec(model_name)
        if spec.name not in seen:
            resolved.append(spec.name)
            seen.add(spec.name)
    return resolved
