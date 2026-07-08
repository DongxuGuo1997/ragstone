"""
Unit tests for the named API key store (pure stdlib, no server).

What is under test: fail-fast parsing, constant-time-friendly auth by
name, the sliding-window quota (with an injected clock), runtime
revocation, and a usage report that never leaks key material.
"""

import pytest

from ragstone.api.keys import ApiKeyRecord, ApiKeyStore


def _store(*records, now=None):
    kwargs = {"now": now} if now else {}
    return ApiKeyStore(list(records), **kwargs)


class TestParsing:
    def test_named_keys_with_and_without_rpm(self, monkeypatch):
        monkeypatch.setenv("RAGSTONE_API_KEYS", "alice:sk-a:60, batch:sk-b")
        monkeypatch.delenv("RAGSTONE_API_KEY", raising=False)
        store = ApiKeyStore.from_env()
        assert store.authenticate("sk-a") == "alice"
        assert store.authenticate("sk-b") == "batch"
        report = {r["name"]: r for r in store.usage_report()}
        assert report["alice"]["rpm_limit"] == 60
        assert report["batch"]["rpm_limit"] is None

    def test_legacy_single_key_becomes_default(self, monkeypatch):
        monkeypatch.delenv("RAGSTONE_API_KEYS", raising=False)
        monkeypatch.setenv("RAGSTONE_API_KEY", "sk-legacy")
        store = ApiKeyStore.from_env()
        assert store.authenticate("sk-legacy") == "default"

    def test_both_sources_merge(self, monkeypatch):
        monkeypatch.setenv("RAGSTONE_API_KEYS", "alice:sk-a")
        monkeypatch.setenv("RAGSTONE_API_KEY", "sk-legacy")
        store = ApiKeyStore.from_env()
        assert store.authenticate("sk-a") == "alice"
        assert store.authenticate("sk-legacy") == "default"

    def test_no_keys_means_auth_disabled(self, monkeypatch):
        monkeypatch.delenv("RAGSTONE_API_KEYS", raising=False)
        monkeypatch.delenv("RAGSTONE_API_KEY", raising=False)
        assert ApiKeyStore.from_env().enabled is False

    @pytest.mark.parametrize(
        "bad",
        [
            "justakey",  # no name
            "alice:",  # empty key
            ":sk-a",  # empty name
            "alice:sk-a:often",  # non-integer rpm
            "alice:sk-a:0",  # rpm must be positive
            "alice:sk-a:-5",
            "a:b:c:d",  # too many fields
        ],
    )
    def test_malformed_entries_fail_fast(self, monkeypatch, bad):
        # A silently dropped key = a client's auth mysteriously failing
        # at request time; refusing to boot is the kinder failure.
        monkeypatch.setenv("RAGSTONE_API_KEYS", bad)
        with pytest.raises(ValueError):
            ApiKeyStore.from_env()

    def test_duplicate_names_rejected(self):
        with pytest.raises(ValueError, match="Duplicate"):
            _store(
                ApiKeyRecord(name="alice", key="sk-1"),
                ApiKeyRecord(name="alice", key="sk-2"),
            )


class TestAuthenticate:
    def test_wrong_or_empty_key_is_none(self):
        store = _store(ApiKeyRecord(name="alice", key="sk-a"))
        assert store.authenticate("sk-wrong") is None
        assert store.authenticate("") is None

    def test_non_ascii_input_rejects_cleanly(self):
        # Latin-1 header bytes reach auth as non-ASCII str; must be a
        # clean mismatch, not an encoding error (the old 500-vs-401 bug).
        store = _store(ApiKeyRecord(name="alice", key="sk-a"))
        assert store.authenticate("s\xe9cret") is None


class TestQuota:
    def test_window_slides(self):
        clock = {"t": 0.0}
        store = _store(
            ApiKeyRecord(name="alice", key="sk-a", rpm=2),
            now=lambda: clock["t"],
        )
        assert store.check_and_count("alice") is True
        assert store.check_and_count("alice") is True
        assert store.check_and_count("alice") is False  # at the limit
        clock["t"] = 61.0  # both requests age out
        assert store.check_and_count("alice") is True

    def test_refused_requests_do_not_extend_the_lockout(self):
        clock = {"t": 0.0}
        store = _store(
            ApiKeyRecord(name="alice", key="sk-a", rpm=1),
            now=lambda: clock["t"],
        )
        assert store.check_and_count("alice") is True
        for _ in range(10):  # hammering while locked out
            assert store.check_and_count("alice") is False
        clock["t"] = 60.5  # only the ACCEPTED request occupied the window
        assert store.check_and_count("alice") is True

    def test_unlimited_key_never_throttles(self):
        store = _store(ApiKeyRecord(name="batch", key="sk-b"))
        assert all(store.check_and_count("batch") for _ in range(100))

    def test_keys_have_independent_windows(self):
        store = _store(
            ApiKeyRecord(name="alice", key="sk-a", rpm=1),
            ApiKeyRecord(name="bob", key="sk-b", rpm=1),
        )
        assert store.check_and_count("alice") is True
        assert store.check_and_count("alice") is False
        assert store.check_and_count("bob") is True  # alice's limit, not bob's


class TestRevocation:
    def test_revoked_key_stops_authenticating(self):
        store = _store(ApiKeyRecord(name="alice", key="sk-a"))
        assert store.revoke("alice") is True
        assert store.authenticate("sk-a") is None
        assert store.revoke("alice") is False  # already gone

    def test_revoking_the_last_key_fails_closed(self):
        # The original implementation derived `enabled` from "any keys
        # remain", so revoking the only key silently DISABLED auth and
        # opened the API to everyone. It must lock instead.
        store = _store(ApiKeyRecord(name="alice", key="sk-a"))
        store.revoke("alice")
        assert store.enabled is True  # auth stays configured
        assert store.authenticate("sk-a") is None  # and nobody gets in

    def test_count_after_revocation_is_refused(self):
        # authenticate() and check_and_count() are separate calls; a key
        # revoked between them must not slip through.
        store = _store(ApiKeyRecord(name="alice", key="sk-a"))
        store.revoke("alice")
        assert store.check_and_count("alice") is False


class TestUsageReport:
    def test_counts_accumulate_per_key(self):
        store = _store(
            ApiKeyRecord(name="alice", key="sk-a", rpm=60),
            ApiKeyRecord(name="batch", key="sk-b"),
        )
        for _ in range(3):
            store.check_and_count("alice")
        store.check_and_count("batch")
        report = {r["name"]: r for r in store.usage_report()}
        assert report["alice"]["requests_total"] == 3
        assert report["alice"]["window_used"] == 3
        assert report["batch"]["requests_total"] == 1

    def test_report_never_contains_key_material(self):
        store = _store(ApiKeyRecord(name="alice", key="sk-secret-a"))
        assert "sk-secret-a" not in repr(store.usage_report())
