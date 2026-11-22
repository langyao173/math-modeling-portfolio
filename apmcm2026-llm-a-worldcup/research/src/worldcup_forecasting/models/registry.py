"""Explicit, typed registry for configurable forecasting models."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, TypeVar

from ._utils import config_values

TaskKind = Literal[
    "aggregate_regression",
    "match_regression",
    "outcome_classification",
    "goal_distribution",
    "half_time_increment",
]
ModelT = TypeVar("ModelT")


@dataclass(frozen=True)
class ModelSpec:
    """Immutable description of a registered model family."""

    name: str
    task_kind: TaskKind
    constructor: Callable[[object | None], Any]
    capabilities: frozenset[str]


_MODEL_REGISTRY: dict[str, ModelSpec] = {}
MODEL_REGISTRY: Mapping[str, ModelSpec] = MappingProxyType(_MODEL_REGISTRY)


def register_model(
    name: str,
    task_kind: TaskKind,
    capabilities: tuple[str, ...] | frozenset[str] = (),
) -> Callable[[type[ModelT]], type[ModelT]]:
    """Register a model class and fail immediately on duplicate keys."""
    key = _normalize_key(name)

    def decorator(model_class: type[ModelT]) -> type[ModelT]:
        if key in _MODEL_REGISTRY:
            existing = _MODEL_REGISTRY[key]
            raise ValueError(
                f"Model {key!r} is already registered by {existing.constructor.__name__}"
            )
        _MODEL_REGISTRY[key] = ModelSpec(
            name=key,
            task_kind=task_kind,
            constructor=model_class,
            capabilities=frozenset(capabilities),
        )
        return model_class

    return decorator


def list_models(task_kind: TaskKind | None = None) -> tuple[str, ...]:
    """List registered names, optionally restricted to one task kind."""
    names = (
        name
        for name, spec in _MODEL_REGISTRY.items()
        if task_kind is None or spec.task_kind == task_kind
    )
    return tuple(sorted(names))


def get_model_spec(name: str) -> ModelSpec:
    """Return a registered specification with an informative unknown-key error."""
    key = _normalize_key(name)
    try:
        return _MODEL_REGISTRY[key]
    except KeyError as error:
        choices = ", ".join(list_models()) or "<none>"
        raise KeyError(f"Unknown model {key!r}. Available models: {choices}") from error


def create_model(
    config: str | Mapping[str, object] | object,
    expected_task_kind: TaskKind | None = None,
) -> Any:
    """Create a registered model from a name or mapping-like configuration."""
    name = config if isinstance(config, str) else _model_name(config)
    spec = get_model_spec(name)
    if expected_task_kind is not None and spec.task_kind != expected_task_kind:
        raise ValueError(
            f"Model {spec.name!r} has task kind {spec.task_kind!r}, not {expected_task_kind!r}"
        )
    constructor_config: object | None = None if isinstance(config, str) else config
    return spec.constructor(constructor_config)


def _model_name(config: object) -> str:
    values = config_values(config)
    for key in ("name", "model_name", "type"):
        value = values.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError("Model configuration must contain 'name', 'model_name', or 'type'")


def _normalize_key(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Model registry names must be nonempty strings")
    return name.strip().casefold().replace("-", "_")
