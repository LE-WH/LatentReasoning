"""Lazy method registry for SFT data construction."""

from __future__ import annotations

from importlib import import_module


METHOD_CLASS_PATHS = {
    "direct": ("sft.methods.direct", "DirectMethod"),
    "concise": ("sft.methods.self_training_concise.select", "ConciseMethod"),
    "tokenskip": ("sft.methods.tokenskip.select", "TokenSkipMethod"),
}


def get_method_cls(name: str):
    """Resolve a method class lazily."""
    module_name, cls_name = METHOD_CLASS_PATHS[name]
    module = import_module(module_name)
    return getattr(module, cls_name)
