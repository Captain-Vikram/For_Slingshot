from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "yolodex" / "config.json"

class ConfigManager:
    def __init__(self, config_path: Path | str | None = None):
        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Load config from file and apply environment variable overrides."""
        if not self.config_path.exists():
            # Fallback to empty if default doesn't exist, though it should.
            base_config = {}
        else:
            try:
                base_config = json.loads(self.config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                print(f"Warning: Malformed config at {self.config_path}, using empty defaults.")
                base_config = {}

        # Apply env var overrides
        # Pattern: YOLODEX_<KEY_UPPER>
        for key, value in os.environ.items():
            if key.startswith("YOLODEX_"):
                config_key = key[8:].lower()
                # Try to parse as JSON (for lists/bools/numbers), fallback to string
                try:
                    parsed_value = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    parsed_value = value
                
                base_config[config_key] = parsed_value

        return base_config

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def update(self, updates: dict[str, Any]) -> None:
        self.config.update(updates)

    def save(self, path: Path | str | None = None) -> None:
        target = Path(path) if path else self.config_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.config, indent=2), encoding="utf-8")

def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Helper to get config dict directly."""
    return ConfigManager(path).config
