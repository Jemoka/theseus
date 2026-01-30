from dataclasses import field as f
from dataclasses import fields
from typing import Any, Union, Dict, Tuple
from collections import defaultdict

from omegaconf import OmegaConf

MISSING_VALUE = "???"


def field(
    key: str, default: Any = None, default_factory: Any = None, **kwargs: Any
) -> Any:
    if default is not None and default_factory is not None:
        raise ValueError("Cannot specify both default and default_factory")
    if default is not None:
        return f(metadata={"th_config_field": key}, default=default, **kwargs)
    elif default_factory is not None:
        return f(
            metadata={"th_config_field": key}, default_factory=default_factory, **kwargs
        )
    else:
        return f(metadata={"th_config_field": key}, **kwargs)


def generate_canonical_config(*classes: Any) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
    """generate a full configuration from components

    Args:
        *classes (List[Type]): list of dataclass types to extract fields from

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: canonical field types and default values
    """

    all_fields = [
        (j.metadata.get("th_config_field"), j.type)
        for i in classes
        for j in fields(i)
        if j.metadata.get("th_config_field") is not None
    ]
    all_fields_lub = defaultdict(list)
    for i, j in all_fields:
        all_fields_lub[i].append(j)
    canonical_fields = {
        key: value[0] if len(value) == 1 else Union[tuple(value)]
        for key, value in all_fields_lub.items()
    }

    all_field_defaults = [
        (j.metadata.get("th_config_field"), j.default)
        for i in classes
        for j in fields(i)
        if j.metadata.get("th_config_field") is not None
        and j.default is not j.default_factory
    ]
    all_field_defaults_lub = defaultdict(list)
    for i, j in all_field_defaults:
        if j is not None:
            all_field_defaults_lub[i].append(j)
    canonical_field_defaults = {
        key: value[0] if len(set(value)) == 1 else None
        for key, value in all_field_defaults_lub.items()
    }

    for i in all_fields_lub.keys():
        if i not in canonical_field_defaults:
            canonical_field_defaults[i] = None

    return canonical_fields, canonical_field_defaults


def nest_slash_keys(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Nests a flat dictionary with slash-separated keys into a nested dictionary.

    Args:
        flat (Dict[str, Any]): Flat dictionary with keys containing slashes.
    Returns:
        Dict[str, Any]: Nested dictionary.
    """

    root: Dict[str, Any] = {}
    for k, v in flat.items():
        parts = k.split("/")
        cur = root
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
            if not isinstance(cur, dict):
                raise ValueError(
                    f"Key collision: {k} tries to nest under a non-dict at {p}"
                )
        leaf = parts[-1]
        if leaf in cur and isinstance(cur[leaf], dict):
            raise ValueError(f"Key collision: {k} wants a value but existing is a dict")
        cur[leaf] = v
    return root


def build_default_config(types: Dict[str, Any], defaults: Dict[str, Any]) -> OmegaConf:
    """Builds a default OmegaConf configuration from types and defaults.
    Args:
        types (Dict[str, Any]): Dictionary of configuration types.
        defaults (Dict[str, Any]): Dictionary of default values.

    Returns:
        OmegaConf: OmegaConf configuration with defaults filled in.
    """

    flat_out: Dict[str, Any] = {}
    for k in types.keys():
        if k in defaults and defaults[k] is not None:
            flat_out[k] = defaults[k]
        else:
            flat_out[k] = MISSING_VALUE
    return OmegaConf.create(nest_slash_keys(flat_out))


def build(*classes: Any) -> OmegaConf:
    """Builds a default OmegaConf configuration from dataclass types.

    Args:
            *classes (DataclassInstance | type[DataclassInstance]): List of dataclass types.

    Returns:
            OmegaConf: OmegaConf configuration with defaults filled in.
    """

    types, defaults = generate_canonical_config(*classes)
    return build_default_config(types, defaults)
