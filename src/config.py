"""Configuration loading.

Every tunable lives in ``conf/*.yaml``; nothing that a reviewer might want to change
is hard-coded inside ``src/``. Paths in config are repo-relative so the same config
works from any working directory and inside a container.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = REPO_ROOT / "conf"


def load_config(name: str) -> dict[str, Any]:
    """Load ``conf/<name>.yaml`` into a plain dict."""
    path = CONF_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(relative: str | Path) -> Path:
    """Resolve a repo-relative path from config into an absolute path."""
    p = Path(relative)
    return p if p.is_absolute() else REPO_ROOT / p


def ensure_parent(path: Path) -> Path:
    """Create the parent directory of ``path`` if needed, then return ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
