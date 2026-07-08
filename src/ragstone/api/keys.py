"""Named API keys with per-key rate limits (ROADMAP 5.3).

One shared key can't answer the questions an operator actually asks:
who is calling, how much, and how do I cut one client off without
rotating everyone? A key store maps each key to a NAME, and everything
downstream — quotas, the audit log, the usage report — speaks names,
never key material.

Configuration (boot-time; parsing is fail-fast because a silently
dropped entry means a client's auth mysteriously failing later):

    RAGSTONE_API_KEYS="alice:sk-a1b2:60,batch:sk-c3d4"
                       name : key [: requests-per-minute]

The legacy single-key form RAGSTONE_API_KEY=<key> keeps working and
becomes the name "default"; both may be set together. No keys at all
means auth is disabled (development), exactly as before.

Rate limiting is a per-key sliding 60-second window, in-process and
in-memory: honest for a single server, and per-worker once horizontal
scale (ROADMAP 5.13) arrives — a distributed limiter is that item's
problem, not this one's.
"""

import hmac
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

_WINDOW_SECONDS = 60.0


@dataclass
class ApiKeyRecord:
    """One named key. ``rpm`` None = no rate limit."""

    name: str
    key: str
    rpm: Optional[int] = None
    # Usage attribution; mutated under the store lock.
    requests_total: int = 0
    window: Deque[float] = field(default_factory=deque)


class ApiKeyStore:
    """Named keys, constant-time auth, sliding-window quotas, usage.

    Thread-safe: the API serves concurrent requests from worker threads.
    """

    def __init__(
        self,
        records: Optional[List[ApiKeyRecord]] = None,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._records: Dict[str, ApiKeyRecord] = {}
        self._lock = threading.Lock()
        self._now = now  # injectable for deterministic window tests
        for record in records or []:
            if record.name in self._records:
                raise ValueError(f"Duplicate API key name: {record.name!r}")
            self._records[record.name] = record
        # Fixed at construction: if auth was configured, revoking the
        # LAST key must lock the API (fail closed), not silently turn
        # auth off for everyone.
        self._auth_required = bool(self._records)

    # -- construction --------------------------------------------------

    @classmethod
    def from_env(cls) -> "ApiKeyStore":
        """Build from RAGSTONE_API_KEYS + legacy RAGSTONE_API_KEY.

        Raises ValueError on malformed entries: refusing to boot beats a
        client discovering at request time that their key was dropped.
        """
        records = []
        spec = os.getenv("RAGSTONE_API_KEYS", "").strip()
        if spec:
            for entry in spec.split(","):
                records.append(cls._parse_entry(entry.strip()))
        legacy = os.getenv("RAGSTONE_API_KEY", "").strip()
        if legacy:
            records.append(ApiKeyRecord(name="default", key=legacy))
        return cls(records)

    @staticmethod
    def _parse_entry(entry: str) -> ApiKeyRecord:
        parts = entry.split(":")
        if len(parts) == 2:
            name, key = parts
            rpm: Optional[int] = None
        elif len(parts) == 3:
            name, key, rpm_text = parts
            try:
                rpm = int(rpm_text)
            except ValueError:
                raise ValueError(
                    f"API key {name!r}: rpm must be an integer, got {rpm_text!r}"
                )
            if rpm <= 0:
                raise ValueError(f"API key {name!r}: rpm must be positive, got {rpm}")
        else:
            raise ValueError(
                f"Malformed RAGSTONE_API_KEYS entry {entry!r} "
                "(expected name:key or name:key:rpm)"
            )
        if not name or not key:
            raise ValueError(f"Malformed RAGSTONE_API_KEYS entry {entry!r}")
        return ApiKeyRecord(name=name, key=key, rpm=rpm)

    # -- auth / quota ----------------------------------------------------

    @property
    def enabled(self) -> bool:
        """False = auth was never configured (development). Deliberately
        NOT "any keys remain": see __init__ on revoking the last key."""
        return self._auth_required

    def authenticate(self, provided: str) -> Optional[str]:
        """The key's name, or None. Constant-time per key, no early exit:
        response timing must not reveal whether a prefix matched, and the
        full scan keeps timing independent of WHICH key matched."""
        provided_bytes = provided.encode("utf-8")
        matched: Optional[str] = None
        with self._lock:
            for record in self._records.values():
                if hmac.compare_digest(provided_bytes, record.key.encode("utf-8")):
                    matched = record.name
        return matched

    def check_and_count(self, name: str) -> bool:
        """Count one request against ``name``; False = over its rpm.

        Refused requests are counted nowhere here (a client at the limit
        must not have its retries extend the lockout); they stay visible
        as 429 lines in the audit log.
        """
        with self._lock:
            record = self._records.get(name)
            if record is None:
                return False  # revoked between authenticate() and here
            if record.rpm is not None:
                cutoff = self._now() - _WINDOW_SECONDS
                window = record.window
                while window and window[0] <= cutoff:
                    window.popleft()
                if len(window) >= record.rpm:
                    return False
                window.append(self._now())
            record.requests_total += 1
            return True

    def revoke(self, name: str) -> bool:
        """Remove a key at runtime; True if it existed."""
        with self._lock:
            existed = self._records.pop(name, None) is not None
        if existed:
            logger.info(f"API key revoked: {name}")
        return existed

    # -- reporting -------------------------------------------------------

    def usage_report(self) -> List[Dict[str, object]]:
        """Per-key usage for operators. Names and counts only — key
        material never leaves the store."""
        with self._lock:
            cutoff = self._now() - _WINDOW_SECONDS
            return [
                {
                    "name": r.name,
                    "rpm_limit": r.rpm,
                    "requests_total": r.requests_total,
                    "window_used": sum(1 for t in r.window if t > cutoff),
                }
                for r in self._records.values()
            ]
