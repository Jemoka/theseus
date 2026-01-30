from dataclasses import field as f
from dataclasses import fields

from typing import Any, Union, Dict, Tuple
from collections import defaultdict

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


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


def generate_canonical_config(
    *classes: DataclassInstance | type[DataclassInstance],
) -> Tuple[Dict[Any, Any], Dict[Any, Any]]:
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
