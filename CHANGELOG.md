# Changelog

## 0.1.0 (unreleased)

First full version.

### Added
- **Data**: Postgres schema for a fake shop (customers, products, orders, order items, support tickets), a read-only `lookup_reader` user, and a repeatable seed (`db/seed.py`) with planted cases for the eval questions. The seed prints a fingerprint so two runs can be compared.
- **Documents**: nine policy files in `data/policies/` (returns, refunds, shipping, premium membership, warranty, damaged items, order changes, payments, support).
- **Search**: ingestion that chunks each policy by section, embeds it (`BAAI/bge-small-en-v1.5`) and stores it in pgvector, skipping unchanged files. Three search layers: vector, hybrid (vector + full-text, merged with Reciprocal Rank Fusion) and hybrid + cross-encoder reranker.
- **Tools**: `search_docs`, `query_database` (single SELECT only, read-only user, 5 s timeout, 100-row limit) and `convert_currency` (Frankfurter exchange-rate API, timeout, clean error messages).
- **Agent**: LangGraph graph (agent node and tools node) with a step cap, conversation memory in Postgres per `thread_id`, versioned system prompts (`v1`, `v2`, later `v3`), and a sources list on every answer.
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

### Changed: shorter conversations, faster search, a protected /chat, prompt v3 (2026-11-08)
- **Long conversations are trimmed before each LLM call.** Tool results from earlier turns (over 300 characters) are replaced by a short placeholder in what is sent; the current turn, the earlier answers and the saved history are left as they are, and every tool request keeps its result. Follow-up questions still work (checked with the hard-set FOLLOWUP cases and a 4-turn conversation in Docker).
- **Search is faster.** Measured first: the first search after a start took 10.6 s, 9.6 s of it the Hugging Face library checking online for newer versions of models already on disk. Models now load from the local cache first (downloading only when missing), the server loads them and opens the search pool at startup, and search uses a shared connection pool (pgvector types registered) that it hands back before reranking. First search after a start (not counting the torch import, which now also happens at startup): 10,568 ms to 206 ms; startup now spends about 3.5 s on it instead. Warm search 212 ms to 200 ms (the reranker is about 180 ms of it). Retrieval hit rates unchanged.
- **Optional API key on `/chat`** (`CHAT_API_KEY`, header `X-API-Key`, 401 when missing or wrong) and an **in-memory per-IP rate limit** (`CHAT_RATE_LIMIT_PER_MINUTE`, default 20 per sliding 60 seconds, 429 with `Retry-After`, no new dependency). Both run before the agent. The chat page shows short notices for 401 and 429.
- **Prompt v3** is the new default: v2 plus one rule sending how-the-shop-works questions to `search_docs` before saying the information isn't available, without searching for data that doesn't exist. Partial run: q28 (v2's miss), the NONE cases, one case per category, the hard-set TRAP and FOLLOWUP cases, 17/17 after re-scoring. v1 and v2 unchanged.

### Fixed (2026-11-08)
- A non-ASCII `X-Ingest-Token` header caused a 500 instead of a 401 (secrets are now compared as bytes; the same goes for `X-API-Key`).
- Scripts and evals that search printed a shutdown error on Python 3.14; the search pool now closes at exit.
- The eval scorer missed two correct refusal wordings ("does not contain ... can't calculate", "did not have any order"). Added to the phrase lists; no saved v1 or v2 result changes.
- `docker compose up --build` checked end to end on 2026-11-08: DOCS, DB and MIXED questions, a 4-turn conversation, 401 and 429, and the chat page (asking, follow-ups, Stop, both notices). The demo GIF's three questions still give the same answers with v3.

### Security and correctness review (2026-11-14)
Found in a final end-to-end review: attacks run against a throwaway database and the HTTP API, a read of all the code, and both PRs.

**Security**
- **Wide SQL results**: one `SELECT repeat('x', 200000000)` cost the app about 430 MB of memory, because rows were cut only after they arrived. Postgres now leaves out any row over 128 KB before sending it, and the agent is told why.
- **SQL check**: a quote character inside a comment, a dollar-quoted string or a quoted name could hide a second statement from the check (psycopg's one-command mode and the read-only role still stopped it). Comments, strings and quoted names are now blanked out the way Postgres reads them, and every query runs wrapped in a subquery. Integration tests run the hidden statements with the check switched off, to prove the layers behind it hold.
- **Read-only user**: `lookup_reader` could cancel or end other sessions' queries. Revoked. Its rules moved to `db/roles.sql`, which `setup_db` applies on every start, so an existing database (like the Docker volume) gets the fix on its next start.
- **Request size**: bodies were read in full before the API key and rate-limit checks (a 50 MB body without a key cost about 240 MB). Bodies over 64 KB now get 413 before they are read, and `thread_id` is limited to 128 safe characters.
- **Docker**: the app was published on every network interface while `/chat` and `/ingest` were open by default. It is now on 127.0.0.1; the README says how to open it up safely.
- **Chat page**: the CDN libraries are pinned with integrity hashes; images, forms and inline styles are stripped from answers; the page has a Content-Security-Policy (its own script by hash, no outside images); every response has `X-Content-Type-Options`, `X-Frame-Options` and `Referrer-Policy`.
- **Thread ids** no longer appear in logs or Langfuse (a short hash instead): whoever has one can continue the conversation.
- **CI**: the workflow's token is read-only, actions are pinned to exact commits, and Dependabot opens weekly update PRs for pip packages and actions.

**Correctness**
- Two questions on one thread at the same time could lose a turn; they now run one after the other. A thread holds at most `MAX_THREAD_TURNS` questions (default 50, then 409), and `scripts/prune_threads.py` deletes conversations unused for N days.
- Two `/ingest` calls at once could fail on a new file; they now run one at a time.
- A non-ASCII or over-long `X-Request-ID` came back garbled; it is now replaced with a fresh id.
- `scripts.ingest_docs <folder>` removed every document not in that folder (the whole policy index); it now only adds and updates, unless `--replace` is given.
- `scripts.setup_db` with no flag wiped a database that had data; it now refuses unless `--reset` is given (`--if-empty`, used by Docker, is unchanged).
- The integration tests refuse a database whose name doesn't end in `_test` (an exported `TEST_DATABASE_URL=.../lookup_db` would have wiped it on every push).
- The seed fingerprint no longer depends on the server's DateStyle, and a test pins it.

**Evals**
- The scorer passed some wrong answers: a count found in the question's own dates ("January 1" when the answer was 1), the opposite yes/no answer (h11, h12), a refusal with a made-up figure (q22, h01, h02, h08), and extra names in a list. Dates written with month names are now ignored, a count must belong to its unit ("3 orders"), a number phrase must be the whole number, a phrase must start at a word, names cases fail on extra names, and refusal cases forbid invented figures. All 187 saved answers score the same as before.
- `run_eval`: a failing truth query or currency call no longer stops the run, an error always fails its case, and a run that stops early still saves its answers.
- `rescore`: skips files that are already re-scored, keeps cases removed from the set, needs `--rate-date` only for runs with currency cases, and points out expected values that changed since the run.
- Truth SQL runs as the read-only user in autocommit, instead of one long superuser transaction.
- The retrieval eval also reports hit rate @1: vector 100%, hybrid 96%, rerank 92% (MRR unchanged).

**Tests**: 193 (156 without a database), up from 117.

### Not done yet
- Live deploy
