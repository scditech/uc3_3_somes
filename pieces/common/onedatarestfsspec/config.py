"""Configuration handling for OnedataRESTFSSpec."""

import os
from typing import Any, Dict, List, Union
from urllib.parse import parse_qs, urlparse


def parse_onedata_url(url: str) -> Dict[str, Any]:
    """Parse a Onedata URL and extract connection parameters.

    Expected format: onedata://token@onezone.example.com/space/path

    Parameters
    ----------
    url : str
        Onedata URL

    Returns
    -------
    dict
        Parsed connection parameters
    """
    if not url.startswith("onedata://"):
        raise ValueError("URL must start with 'onedata://'")

    parsed = urlparse(url)

    if not parsed.username:
        raise ValueError("Token must be provided as username in URL")

    if not parsed.hostname:
        raise ValueError("Onezone host must be provided in URL")

    # Extract path components
    path_parts = parsed.path.strip("/").split("/") if parsed.path.strip("/") else []

    # Parse query parameters
    query_params = parse_qs(parsed.query)

    config = {
        "onezone_host": f"{parsed.scheme.replace('onedata', 'https')}://{parsed.hostname}",
        "token": parsed.username,
        "path": "/" + "/".join(path_parts) if path_parts else "/",
        "verify_ssl": query_params.get("verify_ssl", ["true"])[0].lower() == "true",
        "timeout": float(query_params.get("timeout", [30])[0]),
    }

    # Add preferred providers if specified
    if "providers" in query_params:
        config["preferred_providers"] = query_params["providers"][0].split(",")

    return config


def get_onedata_config_from_env() -> (
    Dict[str, Union[str, List[str], bool, float, None]]
):
    """Get Onedata configuration from environment variables.

    Returns
    -------
    dict
        Configuration from environment variables
    """
    onezone_host = os.environ.get("ONEDATA_ONEZONE_HOST")
    token = os.environ.get("ONEDATA_TOKEN")

    timeout_str = os.environ.get("ONEDATA_TIMEOUT", "30")
    timeout = 30.0 if not timeout_str or timeout_str == "" else float(timeout_str)

    providers_str = os.environ.get("ONEDATA_PREFERRED_PROVIDERS")
    providers = (
        providers_str.split(",") if providers_str and providers_str.strip() else None
    )

    verify_ssl_str = os.environ.get("ONEDATA_VERIFY_SSL", "true")
    verify_ssl = (
        verify_ssl_str.lower() == "true"
        if verify_ssl_str and verify_ssl_str.strip()
        else True
    )

    return {
        "onezone_host": onezone_host if onezone_host and onezone_host.strip() else None,
        "token": token if token and token.strip() else None,
        "preferred_providers": providers,
        "verify_ssl": verify_ssl,
        "timeout": timeout,
    }


def merge_config(
    url_config: Dict[str, Any],
    env_config: Dict[str, Any],
    explicit_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge configuration from different sources.

    Priority: explicit_config > url_config > env_config

    Parameters
    ----------
    url_config : dict
        Configuration from URL
    env_config : dict
        Configuration from environment
    explicit_config : dict
        Explicitly provided configuration

    Returns
    -------
    dict
        Merged configuration
    """
    config = {}

    # Start with environment config
    for key, value in env_config.items():
        if value is not None:
            config[key] = value

    # Override with URL config
    for key, value in url_config.items():
        if value is not None:
            config[key] = value

    # Override with explicit config
    for key, value in explicit_config.items():
        if value is not None:
            config[key] = value

    return config
