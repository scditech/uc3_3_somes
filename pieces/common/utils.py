from __future__ import annotations

import os
from functools import cache
from pathlib import PurePosixPath
from typing import Tuple

import fsspec


@cache
def fs() -> fsspec.AbstractFileSystem:
    fs_type = os.environ.get("FSSPEC_TYPE", "file")
    if not fs_type or fs_type == "file":
        return fsspec.filesystem("file")
    return fsspec.filesystem(fs_type)


def fsspec_config_init(secrets_data) -> None:
    if not secrets_data:
        return

    onedata_onezone_host = getattr(secrets_data, "onedata_onezone_host", None)
    onedata_token = getattr(secrets_data, "onedata_token", None)

    if onedata_onezone_host and onedata_token:
        try:
            import onedatarestfsspec  # noqa: F401 — register onedata:// with fsspec
        except ImportError:
            pass
        fsspec.config.conf["onedata"] = {
            "token": onedata_token,
            "onezone_host": onedata_onezone_host,
        }


def ensure_parent_dir(path: str) -> None:
    """Create the parent directory for a file path or URL if it doesn't exist."""
    filesystem, fs_path = fsspec.core.url_to_fs(path)
    parent = str(PurePosixPath(fs_path).parent)
    if parent not in ("", ".", "/"):
        filesystem.makedirs(parent, exist_ok=True)


def makedirs(path: str, exist_ok: bool = True) -> None:
    """Create directories for the given path or URL."""
    filesystem, fs_path = fsspec.core.url_to_fs(path)
    filesystem.makedirs(fs_path, exist_ok=exist_ok)


def path_exists(path: str) -> bool:
    """Check whether the given path or URL exists."""
    filesystem, fs_path = fsspec.core.url_to_fs(path)
    return filesystem.exists(fs_path)


def isfile(path: str) -> bool:
    """Check whether the given path or URL exists."""
    filesystem, fs_path = fsspec.core.url_to_fs(path)
    return filesystem.isfile(fs_path)


def isdir(path: str) -> bool:
    """Check whether the given path or URL exists."""
    filesystem, fs_path = fsspec.core.url_to_fs(path)
    return filesystem.isdir(fs_path)


def listdir(path: str) -> list:
    """List contents of the given path or URL, returning full URLs."""
    filesystem, fs_path = fsspec.core.url_to_fs(path)
    proto, _ = fsspec.core.split_protocol(path)
    entries = filesystem.ls(fs_path, detail=False)
    if proto:
        return [f"{proto}:///{p}" for p in entries]
    return entries
