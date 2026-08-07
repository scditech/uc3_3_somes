"""Utility functions for OnedataRESTFSSpec."""

import posixpath
from typing import Optional, Tuple
from urllib.parse import unquote


def normalize_onedata_path(path: str) -> str:
    """Normalize a Onedata path.

    Parameters
    ----------
    path : str
        Path to normalize

    Returns
    -------
    str
        Normalized path
    """
    # Handle empty path
    if not path:
        return "/"

    # Remove protocol if present
    if "://" in path:
        path = path.split("://", 1)[1]
        # Remove host part if present
        if "/" in path:
            path = "/" + path.split("/", 1)[1]

    # URL decode
    path = unquote(path)

    # Normalize path
    path = posixpath.normpath(path)

    # Handle special case where normpath returns "."
    if path == ".":
        return "/"

    # Ensure it starts with /
    if not path.startswith("/"):
        path = "/" + path

    return path


def split_onedata_path(path: str) -> Tuple[str, Optional[str]]:
    """Split a Onedata path into space name and file path.

    Parameters
    ----------
    path : str
        Full path to split

    Returns
    -------
    tuple
        (space_name, file_path) where file_path can be None for space root
    """
    normalized = normalize_onedata_path(path)
    # Remove leading slash and handle empty case
    path_clean = normalized.lstrip("/")

    if not path_clean:
        return "", None

    if "/" in path_clean:
        space_name, file_path = path_clean.split("/", 1)
        return space_name, file_path
    return path_clean, None


def join_onedata_path(space_name: str, file_path: Optional[str] = None) -> str:
    """Join space name and file path into a full Onedata path.

    Parameters
    ----------
    space_name : str
        Name of the space
    file_path : str, optional
        Path within the space

    Returns
    -------
    str
        Full Onedata path
    """
    if not space_name:
        if file_path:
            return f"/{file_path}"
        return "/"

    if file_path:
        return f"/{space_name}/{file_path}"
    return f"/{space_name}"


def validate_onedata_path(path: str) -> bool:
    """Validate a Onedata path.

    Parameters
    ----------
    path : str
        Path to validate

    Returns
    -------
    bool
        True if path is valid
    """
    try:
        normalized = normalize_onedata_path(path)
        space_name, _ = split_onedata_path(normalized)

        # Allow empty space name for root path
        if not space_name:
            return True

        # Check space name doesn't contain invalid characters
        if any(char in space_name for char in ["/", "\\", "\0"]):
            return False

        # For paths like "/space/with/slash", the space_name would be "space"
        # and file_path would be "with/slash", which is valid
        return True
    except (ValueError, TypeError):
        return False


def get_parent_path(path: str) -> str:
    """Get parent directory path.

    Parameters
    ----------
    path : str
        Path to get parent of

    Returns
    -------
    str
        Parent path
    """
    normalized = normalize_onedata_path(path).rstrip("/")
    return posixpath.dirname(normalized) or "/"


def get_basename(path: str) -> str:
    """Get basename of path.

    Parameters
    ----------
    path : str
        Path to get basename of

    Returns
    -------
    str
        Basename
    """
    normalized = normalize_onedata_path(path).rstrip("/")
    return posixpath.basename(normalized)
