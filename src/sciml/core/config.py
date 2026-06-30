"""Configuration primitives shared by all examples.

Configs are plain ``dataclasses`` with a small mixin (:class:`ConfigBase`) that
adds dict/JSON/YAML (de)serialization. Each worked example defines its own
config dataclasses (see ``sciml.problems.<name>.config``) composed from these
primitives. YAML support is optional; JSON works with the stdlib alone.
"""

from __future__ import annotations

import dataclasses
import json
import typing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

T = TypeVar("T", bound="ConfigBase")


class ConfigBase:
    """Mixin adding (de)serialization to dataclasses."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls: Type[T], d: Dict[str, Any]) -> T:
        """Build the (possibly nested) dataclass from a plain dict.

        Unknown keys raise ``TypeError`` so config-file typos surface early.
        """
        # ``from __future__ import annotations`` makes field types strings, so
        # resolve them via get_type_hints (evaluated in the dataclass's module).
        hints = typing.get_type_hints(cls)
        field_names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
        kwargs: Dict[str, Any] = {}
        for key, value in d.items():
            if key not in field_names:
                raise TypeError(f"{cls.__name__}: unknown config key {key!r}")
            kwargs[key] = _coerce(hints.get(key), value)
        return cls(**kwargs)  # type: ignore[call-arg]

    # -- file IO ----------------------------------------------------------
    @classmethod
    def load(cls: Type[T], path: str | Path) -> T:
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            d = _load_yaml(text)
        elif path.suffix.lower() == ".json":
            d = json.loads(text)
        else:
            raise ValueError(f"Unsupported config extension: {path.suffix!r}")
        return cls.from_dict(d or {})

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in {".yaml", ".yml"}:
            path.write_text(_dump_yaml(self.to_dict()), encoding="utf-8")
        else:
            path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def _coerce(ftype: Any, value: Any) -> Any:
    """Recursively coerce ``value`` into ``ftype`` for nested configs and
    ``List[SomeConfig]`` fields; pass everything else through unchanged."""
    if isinstance(value, dict) and isinstance(ftype, type) and issubclass(ftype, ConfigBase):
        return ftype.from_dict(value)
    origin = typing.get_origin(ftype)
    if origin in (list, typing.List) and isinstance(value, list):
        args = typing.get_args(ftype)
        if args and isinstance(args[0], type) and issubclass(args[0], ConfigBase):
            return [args[0].from_dict(v) if isinstance(v, dict) else v for v in value]
    return value


@dataclass
class DomainConfig(ConfigBase):
    """A spatio-temporal box ``[0, length] x [0, t_final]``."""

    length: float = 10.0
    t_final: float = 1.0


def _load_yaml(text: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Reading YAML configs requires PyYAML (`pip install pyyaml`), "
            "or use a .json config instead.") from exc
    return yaml.safe_load(text)


def _dump_yaml(d: Dict[str, Any]) -> str:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Writing YAML configs requires PyYAML (`pip install pyyaml`), "
            "or save to a .json path instead.") from exc
    return yaml.safe_dump(d, sort_keys=False)
