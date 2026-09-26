import pytest

from providers.search.serpapi import set_client


@pytest.fixture(autouse=True)
def _no_live_serpapi(monkeypatch):
    """Tests must never spend SerpApi credits: drop any real key from .env
    and reset the shared client around every test."""
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    monkeypatch.delenv("SERPAPI_RECORD_DIR", raising=False)
    set_client(None)
    yield
    set_client(None)
