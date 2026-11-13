from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .repro import ensure_dir, jsonable


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def environment_snapshot() -> dict:
    snap = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cpu_count": os.cpu_count(),
    }
    try:
        snap["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        snap["git_commit"] = None
    try:
        import numpy as np

        snap["numpy"] = np.__version__
    except Exception:
        pass
    try:
        import torch

        snap["torch"] = torch.__version__
        snap["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    try:
        import openfhe

        snap["openfhe_python"] = getattr(openfhe, "__version__", "installed")
    except Exception:
        snap["openfhe_python"] = None
    return snap


def write_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(obj), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(p)
    return p


def append_jsonl(path: str | Path, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(jsonable(obj), sort_keys=True) + "\n")
