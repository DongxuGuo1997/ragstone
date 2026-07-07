"""Input-boundary security guards for document ingestion.

Two attack surfaces get checked before any loader runs:

- **Path traversal / local file exfiltration.** The MCP server and REST
  API accept a ``data_dir``; without a boundary, any connected client
  could point the pipeline at ``~/.ssh`` or ``/etc`` and read the
  contents back out through answers. Set RAGSTONE_DATA_ROOT to confine
  ingestion to one directory tree (recommended for any server
  deployment; unset keeps library usage unrestricted).

- **SSRF via page URLs.** ``page_urls`` are fetched server-side; without
  a guard they could target internal services or cloud metadata
  endpoints (169.254.169.254). Non-HTTP schemes and hosts resolving to
  private, loopback, link-local, or reserved addresses are rejected.
  This is a baseline guard, not a proxy-grade defense: DNS rebinding
  between check and fetch is out of scope (documented in SECURITY.md).
"""

import ipaddress
import logging
import socket
from pathlib import Path
from urllib.parse import urlparse

from ..config.settings import get_config
from .exceptions import ValidationError

logger = logging.getLogger(__name__)


def validate_data_dir(data_dir: str) -> str:
    """Enforce the RAGSTONE_DATA_ROOT boundary on a data directory.

    Returns the directory unchanged when no root is configured (library
    usage) or when it resolves inside the root.

    Raises:
        ValidationError: If a root is configured and data_dir escapes it.
    """
    root = get_config().loader.allowed_data_root
    if not root:
        return data_dir

    resolved_root = Path(root).resolve()
    resolved_dir = Path(data_dir).resolve()
    if not resolved_dir.is_relative_to(resolved_root):
        raise ValidationError(
            f"Data directory {data_dir!r} is outside the allowed root "
            f"({resolved_root}). Set RAGSTONE_DATA_ROOT to change the "
            "boundary."
        )
    return data_dir


def validate_page_url(url: str) -> str:
    """Reject URLs that could reach internal services (basic SSRF guard).

    Raises:
        ValidationError: For non-HTTP(S) schemes, unresolvable hosts, or
            hosts resolving to private/loopback/link-local/reserved
            addresses.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValidationError(
            f"URL {url!r} rejected: only http/https schemes are allowed."
        )
    host = parsed.hostname
    if not host:
        raise ValidationError(f"URL {url!r} rejected: no host.")

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise ValidationError(f"URL {url!r} rejected: cannot resolve host.") from exc

    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            raise ValidationError(
                f"URL {url!r} rejected: host resolves to a non-public "
                f"address ({address})."
            )
    return url
