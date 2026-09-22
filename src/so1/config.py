"""Configuration: YAML file plus ${ENV_VAR} expansion."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

# ${VAR} or ${VAR:-fallback}
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _lookup(match: re.Match[str]) -> str | None:
    """${VAR} -> the variable, ${VAR:-fallback} -> the variable or the fallback."""
    return os.environ.get(match.group(1)) or match.group(2)


def _expand(value: Any) -> Any:
    """Replace ${VAR} / ${VAR:-fallback} in strings.

    A whole-string reference that resolves to nothing becomes None, so an unset optional key
    (an API key, a URL override) reads as absent rather than as an empty string.
    """
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if not isinstance(value, str):
        return value
    match = _ENV_REF.fullmatch(value)
    if match:
        return _lookup(match) or None
    return _ENV_REF.sub(lambda m: _lookup(m) or "", value)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    url: str
    upstream_model: str
    description: str = ""
    release_date: str = "1970-01-01"
    enabled: bool = True
    tokenizer_url: str | None = None
    """Where to reach vLLM's POST /tokenize. Defaults to `url`.

    Set this when inference goes through a gateway (LiteLLM, a load balancer) that only exposes
    the OpenAI surface: /tokenize is a vLLM extension and will 404 there. Point it at a vLLM
    instance serving the same model. Without it the service falls back to assumed labels.
    """
    max_model_len: int = 0
    """Context length to assume when the upstream does not report one, as gateways do not."""

    @property
    def tokenize_url(self) -> str:
        return self.tokenizer_url or self.url


@dataclass(frozen=True)
class Limits:
    max_questions: int = 64
    max_choice_options: int = 64
    max_score_levels: int = 10
    max_body_bytes: int = 2 * 1024 * 1024
    max_prompt_tokens: int = 32768
    max_concurrent_requests: int = 16
    max_upstream_concurrency: int = 64
    request_timeout_s: float = 120.0
    upstream_timeout_s: float = 90.0


@dataclass(frozen=True)
class Config:
    default_model: str
    models: dict[str, ModelConfig]
    aliases: dict[str, str] = field(default_factory=dict)
    upstream_api_key: str | None = None
    api_key: str | None = None
    temperature: dict[str, float] = field(default_factory=lambda: {"noul": 1.0, "choice": 1.0, "score": 1.0})
    limits: Limits = field(default_factory=Limits)

    def resolve(self, name: str) -> ModelConfig | None:
        """Map a public model name or alias to a configured, enabled upstream."""
        target = self.aliases.get(name, name)
        model = self.models.get(target)
        return model if model is not None and model.enabled else None

    def public_names(self) -> list[tuple[str, ModelConfig]]:
        """Served model ids and aliases, aliases last, disabled models omitted."""
        names = [(n, m) for n, m in self.models.items() if m.enabled]
        names += [(a, self.models[t]) for a, t in self.aliases.items() if t in self.models and self.models[t].enabled]
        return names


def load(path: str | Path) -> Config:
    raw = _expand(yaml.safe_load(Path(path).read_text()) or {})
    models = {
        name: ModelConfig(name=name, **{k: v for k, v in (spec or {}).items() if v is not None})
        for name, spec in (raw.get("models") or {}).items()
    }
    limits = Limits(**{k: v for k, v in (raw.get("limits") or {}).items() if v is not None})
    default_model = raw.get("default_model") or next(iter(models), "")
    temperature = {"noul": 1.0, "choice": 1.0, "score": 1.0} | {
        k: float(v) for k, v in (raw.get("temperature") or {}).items() if v is not None
    }
    return Config(
        default_model=default_model,
        models=models,
        aliases={k: v for k, v in (raw.get("aliases") or {}).items()},
        upstream_api_key=raw.get("upstream_api_key"),
        api_key=raw.get("api_key"),
        temperature=temperature,
        limits=limits,
    )


def with_limits(config: Config, **overrides: Any) -> Config:
    return replace(config, limits=replace(config.limits, **overrides))
