from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Optional


@dataclass
class PluginInfo:
    name: str
    module: str
    class_name: str


REGISTRY: dict[str, PluginInfo] = {
    # Map process name (as used in ASL state("PROCESS", ...)) to plugin
    # You can extend this by adding new entries for other games.
    "DELTARUNE": PluginInfo(
        name="Deltarune",
        module="games.Deltarune.deltarune",
        class_name="DeltarunePlugin",
    ),
}


def load_plugin(process_name: str):
    info: Optional[PluginInfo] = REGISTRY.get(process_name.upper())
    if not info:
        return None
    try:
        mod = importlib.import_module(info.module)
        cls = getattr(mod, info.class_name, None)
        if cls is None:
            return None
        return cls(process_name)
    except Exception as e:
        # If anything goes wrong, silently skip plugin loading
        return None
