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
- `/chat` replies leave out empty source fields instead of sending them as `null`.

### Added (after the first real eval run, 2026-10-24)
- `eval/rescore.py`: score a saved run again without new AI calls, using the exchange rates of the run's day.
- Eval results now save each answer's sources, the expected values and the rate date, so later re-scoring is exact.
- `convert()` takes an optional `on_date` for historical rates.
- First measured results in the README.

### Changed (2026-10-24)
- Coverage is uploaded to Codecov on every push; the README badge shows it.
- `docker compose up --build` checked end to end on 2026-10-24 (a MIXED question answered with the database and the currency API).
- Steadfast now installs from PyPI (`steadfast-llm` 0.3.0) instead of from GitHub, so the Docker image no longer needs git.

### Added: chat page (2026-10-24)
- A chat page at `/` (`app/static/index.html`): suggested questions, conversation memory, a "checked" trail showing which sources each answer used, sources with the SQL, light and dark mode, works on mobile.
- A stop button (and Esc) on the chat page. `/chat` cancels the agent when the caller hangs up (returns 499), and unfinished tool steps from a stopped run are dropped before the next LLM call, so the conversation keeps working.
- The request-id middleware is now plain ASGI, so endpoints can see client disconnects.

### Added: harder evals (2026-10-24)
- `eval/hard_set.json`: 15 harder questions in five groups (TRAP, SAFETY, FOLLOWUP, PARTIAL, REASONING); run with `--set hard`.
- Two new scoring checks: `must_not_mention` (e.g. "has been deleted") and `max_tool_calls`.
- Eval cases can have earlier `turns` in the same conversation, to test memory.

### Hard-set results (2026-10-24, gpt-oss-120b)
- First hard-set runs: v1 12/15, v2 14/15 after re-scoring. Saved in `eval/results/` with `summary-hard.md`.
- Scorer: more "not found" wordings accepted for h01 and h15; h15 no longer requires a database query, because the category list is in the tool description.
- Prompt v2 gained a rule: for "most / least / top" questions, check for ties instead of using `LIMIT 1` (both prompts missed a three-way tie on h10). A re-run of h10 with the rule passed.
- `eval/rescore.py` prints the right summary file name for the hard set.

### Tracing checked (2026-10-24)
- Langfuse tracing checked with a real account: each chat turn shows its LLM and tool steps with timings and the prompt version. Screenshot in the README.

### Added (2026-10-25)
- Demo GIF of the chat page at the top of the README.

### Fixed (found while recording the demo, 2026-10-25)
- The agent converted one order with `convert_currency` and worked out a second one itself, 2 cents off ($58.00 became €51.84 instead of €51.82). Prompt v2 now says to never do arithmetic and to convert every amount with its own call.
- "Which product sold the most?" counted cancelled orders. Prompt v2 now treats "sold", "sales" and "revenue" as non-cancelled orders unless the user says otherwise, and tells the agent to mention it.

### Results with the current prompt (2026-10-25)
- Prompt v2 re-run with its new rules: 40/41 on the main set (gpt-oss-20b) and 15/15 on the hard set (gpt-oss-120b), both on Groq's free tier. The one miss (q28) said support reply times aren't stored instead of checking the support policy; the same question passed in the previous run. README tables updated.

### Fixed (review pass, 2026-10-30)
- The SQL tool read the whole result into memory before applying the row limit. It now uses a server-side cursor, so only the rows it returns are fetched, and text values over 2,000 characters are cut.
- The SQL check misread `E'\''` strings (which could hide a keyword after them), rejected `;` inside string literals, and allowed `SELECT ... INTO`. Writes were still blocked by the read-only user.
- Conversation memory used one Postgres connection that never reconnected, so after a database restart every `/chat` failed until the app was restarted. It now uses a connection pool that checks connections before use.
- A crash inside a request returned a 500 without the `X-Request-ID` header. Caller-supplied request ids are now capped at 128 characters.
- Log lines from httpx and uvicorn were plain text without the correlation id; they now go through the same JSON format.
- A wrong `PROMPT_VERSION` now stops the app at startup instead of failing every question.
- `setup_db --if-empty` now checks that the seed finished, not only that the tables exist, so an interrupted first start is repaired on the next one. The reranker model is downloaded during setup instead of during the first question.
- Docker: the Postgres healthcheck checks over TCP (the image's first-start server only listens on a socket), Postgres is published on 127.0.0.1 only, and the app container restarts if it exits.
- The eval scorer counted the digits of dates ("2026-02-18") as numbers in answers. Re-checked offline: no saved result changes.
- Removed the unused `embedding_dim` setting.

### Not done yet
- Live deploy
