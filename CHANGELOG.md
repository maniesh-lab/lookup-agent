# Changelog

## 0.1.0 (unreleased)

First full version.

### Added
- **Data**: Postgres schema for a fake shop (customers, products, orders, order items, support tickets), a read-only `lookup_reader` user, and a repeatable seed (`db/seed.py`) with planted cases for the eval questions. The seed prints a fingerprint so two runs can be compared.
- **Documents**: nine policy files in `data/policies/` (returns, refunds, shipping, premium membership, warranty, damaged items, order changes, payments, support).
- **Search**: ingestion that chunks each policy by section, embeds it (`BAAI/bge-small-en-v1.5`) and stores it in pgvector, skipping unchanged files. Three search layers: vector, hybrid (vector + full-text, merged with Reciprocal Rank Fusion) and hybrid + cross-encoder reranker.
- **Tools**: `search_docs`, `query_database` (single SELECT only, read-only user, 5 s timeout, 100-row limit) and `convert_currency` (Frankfurter exchange-rate API, timeout, clean error messages).
- **Agent**: LangGraph graph (agent node and tools node) with a step cap, conversation memory in Postgres per `thread_id`, versioned system prompts (`v1`, `v2`), and a sources list on every answer.
- **LLM calls** through [Steadfast](https://github.com/maniesh-lab/steadfast) 0.3: retries, rate limits, Groq with OpenAI as fallback, token counts.
- **API**: `GET /health`, `POST /chat`, `POST /ingest` (optional token). Every request gets a correlation id that appears on all its JSON log lines and in the `X-Request-ID` header. `/chat` returns 503 when every LLM provider is down.
- **Tracing**: optional Langfuse traces with the prompt version on each trace.
- **Evals**: 41-question test set across DOCS, DB, API, MIXED and NONE, scored with rules; database answers come from truth SQL and currency answers from a live rate at run time. A retrieval hit-rate script for the three search layers.
- **Tests and CI**: unit tests with a fake LLM and fake search models, integration tests against a real pgvector Postgres, GitHub Actions running ruff and pytest with coverage.
- **Docker**: Dockerfile and a Compose file that starts Postgres and the API, building the database on first start.
- **Docs**: `docs/architecture.md` with a diagram, and this README.

### Fixed (after the first real eval run, 2026-10-24)
- The scorer failed correct answers that used lookalike characters (non-breaking hyphens, narrow no-break spaces inside names and as thousands separators). Text is now normalized before every check.
- Refusal detection now also accepts "doesn't record" / "don't keep track" wording.

### Added (after the first real eval run)
- `eval/rescore.py`: score a saved run again without new AI calls, using the exchange rates of the run's day.
- Eval results now save each answer's sources, the expected values and the rate date, so later re-scoring is exact.
- `convert()` takes an optional `on_date` for historical rates.
- First measured results in the README.

### Changed
- Coverage is uploaded to Codecov on every push; the README badge shows it.
- `docker compose up --build` checked end to end on 2026-10-24 (a MIXED question answered with the database and the currency API).
- Steadfast now installs from PyPI (`steadfast-llm` 0.3.0) instead of from GitHub, so the Docker image no longer needs git.

### Not done yet
- Live deploy or demo GIF
