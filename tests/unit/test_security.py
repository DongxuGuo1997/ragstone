"""
Unit tests for input-boundary security guards (no network required —
SSRF checks use IP literals and localhost, which resolve locally).
"""

import pytest

from ragstone.config.settings import get_config
from ragstone.rag.pipeline import OpenAIPipeline
from ragstone.utils.exceptions import ValidationError
from ragstone.utils.security import validate_data_dir, validate_page_url


class TestDataRootBoundary:
    def test_unrestricted_when_no_root_configured(self, monkeypatch):
        monkeypatch.setattr(get_config().loader, "allowed_data_root", None)
        assert validate_data_dir("/etc") == "/etc"  # library usage: allowed

    def test_dir_inside_root_is_allowed(self, monkeypatch, tmp_path):
        monkeypatch.setattr(get_config().loader, "allowed_data_root", str(tmp_path))
        inside = tmp_path / "docs"
        inside.mkdir()
        assert validate_data_dir(str(inside)) == str(inside)

    def test_escape_via_absolute_path_is_rejected(self, monkeypatch, tmp_path):
        monkeypatch.setattr(get_config().loader, "allowed_data_root", str(tmp_path))
        with pytest.raises(ValidationError, match="outside the allowed root"):
            validate_data_dir("/etc")

    def test_escape_via_dotdot_is_rejected(self, monkeypatch, tmp_path):
        # ".." traversal must be resolved before the containment check.
        monkeypatch.setattr(get_config().loader, "allowed_data_root", str(tmp_path))
        sneaky = str(tmp_path / "docs" / ".." / "..")
        with pytest.raises(ValidationError, match="outside the allowed root"):
            validate_data_dir(sneaky)

    def test_pipeline_enforces_boundary_before_loading(self, monkeypatch, tmp_path):
        # The chokepoint: MCP/API pass client-supplied data_dir straight
        # into load_and_split, so the guard must fire there.
        monkeypatch.setattr(get_config().loader, "allowed_data_root", str(tmp_path))
        pipeline = OpenAIPipeline(model="gpt-4o-mini")
        with pytest.raises(ValidationError, match="outside the allowed root"):
            pipeline.load_and_split(data_dir="/etc")


class TestSsrfGuard:
    def test_public_looking_https_url_passes_scheme_check(self):
        # Uses an IP literal to stay offline: a public address is fine.
        assert validate_page_url("https://93.184.216.34/page") == (
            "https://93.184.216.34/page"
        )

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "ftp://example.com/file",
            "gopher://example.com",
        ],
    )
    def test_non_http_schemes_rejected(self, url):
        with pytest.raises(ValidationError, match="http/https"):
            validate_page_url(url)

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/admin",
            "http://localhost:8000/",
            "http://10.0.0.5/internal",
            "http://192.168.1.1/router",
            "http://169.254.169.254/latest/meta-data/",  # cloud metadata
            "http://[::1]/",
        ],
    )
    def test_internal_addresses_rejected(self, url):
        with pytest.raises(ValidationError, match="non-public address"):
            validate_page_url(url)

    def test_unresolvable_host_rejected(self):
        with pytest.raises(ValidationError, match="cannot resolve"):
            validate_page_url("https://this-host-does-not-exist.invalid/")

    def test_pipeline_rejects_internal_urls_before_scraping(self):
        pipeline = OpenAIPipeline(model="gpt-4o-mini")
        with pytest.raises(ValidationError, match="non-public address"):
            pipeline.load_and_split(
                page_urls=["http://169.254.169.254/latest/meta-data/"]
            )
