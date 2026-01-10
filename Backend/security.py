"""
Lightweight security utilities for the API.

Right now this provides optional API key authentication that can be enabled
via an environment variable. This keeps the default local setup simple while
allowing you to secure the API in production.
"""

import os
from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader


API_KEY_HEADER_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


def get_expected_api_key() -> Optional[str]:
    """
    Read the expected API key from environment, if configured.
    """
    return os.getenv("EXOPLANET_API_KEY")


def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> None:
    """
    Dependency that enforces API key authentication when EXOPLANET_API_KEY is set.

    - If no API key is configured, this is a no-op (open API, suitable for local dev).
    - If an API key is configured, incoming requests must include X-API-Key.
    """
    expected = get_expected_api_key()
    if not expected:
        # No API key configured: do not enforce authentication.
        return

    if not api_key or api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )

