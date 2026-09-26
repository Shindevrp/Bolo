SerpApi response fixtures replayed by `tests/test_serpapi.py` (tests never
hit the live API — free tier is 250 searches/month).

The current files are hand-written to match SerpApi's documented response
shapes, trimmed to the fields the tools read. To replace them with real
recordings, run the agent with `SERPAPI_RECORD_DIR=tests/fixtures/serpapi`
(the client drops `search_metadata`, which holds account-scoped URLs).
