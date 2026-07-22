"""Small command-line configuration override helper."""

import json
from copy import deepcopy


def _parse_value(raw):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def apply_overrides(config, overrides):
    """Apply repeated ``key=value`` overrides, supporting dotted keys."""
    result = deepcopy(config)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(
                f"invalid override {override!r}; expected key=value"
            )
        dotted_key, raw_value = override.split("=", 1)
        keys = [key for key in dotted_key.split(".") if key]
        if not keys:
            raise ValueError(f"invalid override key in {override!r}")
        current = result
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        current[keys[-1]] = _parse_value(raw_value)
    return result
