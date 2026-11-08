# lookup-agent

[![CI](https://github.com/maniesh-lab/lookup-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/maniesh-lab/lookup-agent/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/maniesh-lab/lookup-agent/branch/main/graph/badge.svg)](https://codecov.io/gh/maniesh-lab/lookup-agent)

**A chatbot that looks things up before it answers.**

<p align="center"><img src="docs/images/demo.gif" alt="The chat page answering a policy question, an order total in euros, and a best-seller question with a three-way tie, each with its sources" width="600"></p>

Most business questions don't live in one place. "What did Priya's February order cost in Canadian dollars?" needs the order database *and* a live exchange rate. "Which premium customers asked for refunds in April, and what does the refund policy say about premium members?" needs the database *and* the policy documents. A plain chatbot guesses. lookup-agent checks.

It has three tools and picks the right ones for each question:

| Tool | Looks in | For questions like |
|---|---|---|
| `search_docs` | policy documents (hybrid search + reranker) | "Who pays return shipping if you sent the wrong item?" |
| `query_database` | Postgres, through a read-only user | "Which 3 products sold the most from April to June?" |
| `convert_currency` | a live exchange-rate API | "Convert 250 euros to Indian rupees." |

Every answer comes with its **sources**: the policy file and section, the SQL that ran, or the exchange rate and its date. When the data simply doesn't exist (profit, stock levels, website visits), it says so instead of making something up.

The demo data is a fake small shop, Lumen Lane: customers, products, orders, support tickets and nine policy documents.

## How it works

```mermaid
flowchart LR
    user([User]) -->|chat page or POST /chat| api[FastAPI]
    api --> agent[LangGraph agent<br/>step cap + memory]
    agent <-->|Steadfast| llm[(Groq / OpenAI)]
    agent --> S[search_docs] --> pg[(Postgres + pgvector)]
    agent --> Q[query_database] --> pg
    agent --> C[convert_currency] --> fx[(Frankfurter API)]
```

The agent loops between the LLM and the tools until it has an answer, with a step cap so it can't loop forever. Conversations are saved in Postgres per `thread_id`; before each LLM call, tool results from earlier turns are swapped for a short placeholder, so a long conversation doesn't resend every old search result and SQL table. LLM calls go through [Steadfast](https://github.com/maniesh-lab/steadfast), my reliability layer for LLM APIs (retries, rate limits, provider fallback, token counts). The full design is in [docs/architecture.md](docs/architecture.md).

## Results

**Search** (26 queries, `python -m eval.run_retrieval_eval`, no LLM involved):

| Layer | Hit rate @5 | MRR |
|---|---|---|
| Vector | 100% (26/26) | 1.00 |
| Hybrid (vector + keyword, RRF) | 100% (26/26) | 0.97 |
| Hybrid + reranker | 100% (26/26) | 0.96 |

**Agent** (`python -m eval.run_eval`, scored by rules, see `eval/scoring.py`), measured with `openai/gpt-oss-20b`, a small model on Groq's free tier, 41 questions per run, one run per prompt version (v1 on 2026-10-24, v2 with its current rules on 2026-10-25). The default prompt is now **v3** (v2 plus one rule, see below); it has only had a partial run so far.

| | Prompt v1 | Prompt v2 |
|---|---|---|
| Passed | 41/41 | 40/41 |
| Picked the right tools | 41/41 | 40/41 |
| Tool calls, total | 59 | 50 |
| Tool calls on unanswerable (NONE) questions | 7 | 1 |
| Median answer length | 349 characters | 256 characters |
| Tokens used | 117,683 | 144,537 |

How to read this:

- **The rule-based checks barely separate the prompts on this set.** The first scoring pass of the first runs reported 78% (v1) and 80% (v2), but almost every failure was the scorer's fault: the model writes lookalike characters (a non-breaking hyphen in "USB‑C", narrow spaces in "26 568"). After fixing the scorer, those runs were re-scored without new AI calls (`python -m eval.rescore`) and both passed 41/41.
- **v2's one miss (q28):** asked how fast premium members get a support reply, it said reply times aren't stored instead of checking the support policy (12 hours). The same question passed in v2's previous run, so it may be run-to-run variation, but the likely trigger was v2's "when the data doesn't exist" rule being applied to a docs question. Prompt v3 adds one rule for that (see below).
- **The real differences only show up when reading the answers.** With v1 the agent searched the documents 7 times for data that doesn't exist (4 searches for stock levels alone), said the April refund tickets were "still open" when they're closed, assumed a delivery date it didn't have, and offered to "open a support ticket", which it can't do. v2 answered unanswerable questions without searching, gave shorter answers with a sources line, and said when it couldn't work something out.
- **v2 costs about 23% more tokens**, because its system prompt is longer and is sent with every call.
- **Search is saturated too.** With nine short documents on clearly different topics, plain vector search already finds the right file every time; hybrid search and the reranker only shuffle the order slightly. They would matter on a bigger or messier document set.
- **Small sample**: one run per prompt, one model.

**Harder questions** (`python -m eval.run_eval --set hard`, 15 questions, measured with `openai/gpt-oss-120b`, the larger model on Groq's free tier; v1 on 2026-10-24, v2 with its current rules on 2026-10-25):

| | Prompt v1 | Prompt v2 |
|---|---|---|
| Passed | 12/15 | 15/15 |
| TRAP (data that doesn't exist) | 4/5 | 5/5 |
| SAFETY (asked to delete or refund) | 2/2 | 2/2 |
| FOLLOWUP (needs the earlier messages) | 2/2 | 2/2 |
| PARTIAL (half answerable) | 1/2 | 2/2 |
| REASONING (ties, thresholds, dates) | 3/4 | 4/4 |
| Tool calls, total | 30 | 17 |

- **Here the prompts do differ.** With v1 the agent made 6 tool calls looking for a customer who doesn't exist (and searched the policy documents for her), and answered "which staff member handles each ticket" with a long ticket table without saying staff aren't stored. v2 handled both.
- **The tie.** Three products sold 25 units each. v1, like v2 before its tie rule, used `LIMIT 1` in its SQL and named only one; v2 now names all three.
- The v1 numbers are after re-scoring: the first pass marked a correct "there's no toys category" answer as wrong because of its wording, and the scorer's phrase list was extended.
- Run on the 120b model because the 20b model's free daily token limit was used up that day, so these are not directly comparable with the 41-question runs above.

**Prompt v3, partial run** (2026-11-08). v3 is v2 plus one rule: questions about how the shop works (time frames, reply or delivery times, what members get) go to `search_docs` before the agent says the information isn't available, while data that really doesn't exist (costs, stock, staff, website visits) still gets no search. To keep Groq's free tokens for the rest, it was run only on the cases that rule could help or hurt, plus a few others:

| Cases (model) | Prompt v2 (saved runs) | Prompt v3 |
|---|---|---|
| q28, the docs question v2 missed (20b) | 0/1 | 1/1 (searched, answered 12 hours) |
| NONE, data that doesn't exist: q22-q25, q41 (20b) | 5/5 | 5/5 |
| One per category: q01 DOCS, q08 DB, q14 API, q17 MIXED, q23 NONE (20b) | 5/5 | 5/5 |
| Hard set TRAP: h01-h03, h14, h15 (120b) | 5/5 | 5/5 |
| Hard set FOLLOWUP: h06, h07 (120b) | 2/2 | 2/2 |

- **Not a full run**: 17 of the 56 questions (q23 is in two rows and was run twice, passing both times). The v2 column is from v2's saved runs of the same cases.
- **Two v3 passes are after re-scoring.** The first scoring marked q22 ("does not contain cost, profit, or margin information, so I can't calculate or provide…") and h02 ("did not have any order recorded") as wrong, though both are correct refusals. The phrase lists got "does not contain", "can't calculate" and "not have any order", and both runs were re-scored without new AI calls. The new phrases change no saved v1 or v2 result.
- **The new rule caused no extra document searches** on the NONE and TRAP cases (one search on q41, the same as v2). v3 made three more database calls than v2 on the hard set TRAP cases (7 against 4: three queries on h02 instead of one, one on h15 instead of none).
- The FOLLOWUP cases also ran with the conversation trimming (earlier tool results replaced by a placeholder) and still passed.

**What the evals changed.** These runs led to three more rules in prompt v2: check for ties instead of using `LIMIT 1`, convert every amount with the currency tool instead of doing the maths, and leave cancelled orders out of "sold". With them v2 went from 14/15 to 15/15 on the hard set, and the demo above shows the tie and cancelled-order rules in action.

Every answer from every run, with its score, is saved in [`eval/results/`](eval/results/) (`summary.md` and `summary-hard.md` list the runs).

## Quick start (Docker)

You need Docker and a free [Groq API key](https://console.groq.com).

```bash
git clone https://github.com/maniesh-lab/lookup-agent
cd lookup-agent
cp .env.example .env          # then put your key after GROQ_API_KEY=
docker compose up --build
```

The first start builds the database (tables, fake shop data, policy search index) and downloads the search models, so it takes a few minutes. Later starts load the models from the `models` volume at startup, in a few seconds. Then:

- **Chat page:** open http://localhost:8000 and ask a question, or pick one of the suggestions. Each answer shows which sources were checked, in order, and the details underneath (the policy section, the SQL that ran, or the exchange rate). Press the stop button (or Esc) to cancel a question while it's being answered; the server stops working on it too. Light and dark mode follow your system, or switch with the button in the corner. If the server asks for an API key or a client sends too many questions, the page says so in a short notice instead of a generic error.
- **API:** open http://localhost:8000/docs and try `POST /chat`:

```json
{"message": "What was the total of the order Priya Sharma placed on February 18, 2026, in Canadian dollars at today's rate?"}
```

The reply has `answer`, `sources`, `tools_used`, `steps`, `prompt_version` and a `thread_id`. Send the `thread_id` back with your next message to continue the same conversation. The chat page uses this same endpoint, so anything it does, another website, app or bot can do too.

## Run it without Docker (for development)

Needs Python 3.11 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only build, much smaller
pip install -r requirements.txt
cp .env.example .env                     # add GROQ_API_KEY
docker compose up -d postgres            # just the database
python -m scripts.setup_db               # schema + seed data + policy ingestion
uvicorn app.main:app --reload            # chat page and API on http://127.0.0.1:8000
```

Other useful commands:

```bash
python -m scripts.ask "what's your return window?"      # chat in the terminal
python -m scripts.try_tools sql "SELECT count(*) FROM orders"   # run one tool alone
python -m scripts.try_tools search "express delivery cost"
python -m scripts.try_tools fx 250 EUR INR
```

## API

| Method | Path | What it does |
|---|---|---|
| `GET` | `/` | The chat page |
| `GET` | `/health` | `{"status": "ok"}` |
| `POST` | `/chat` | Ask a question. Body: `message`, optional `thread_id`. Needs `X-API-Key` if `CHAT_API_KEY` is set (401 if missing or wrong). 429 with a `Retry-After` header after `CHAT_RATE_LIMIT_PER_MINUTE` questions from one IP in 60 seconds |
| `POST` | `/ingest` | Re-index the policy documents (unchanged files are skipped). Needs `X-Ingest-Token` if `INGEST_TOKEN` is set |

Every response carries an `X-Request-ID` header. The same id is on every log line for that request.

## Safety of the SQL tool

The model writes its own SQL, so it runs inside four layers: only single `SELECT` statements pass the check; it connects as a database user that can only read the five shop tables; every transaction is read-only with a 5-second timeout; and at most 100 rows come back.

## Evals

| File | What it is |
|---|---|
| `eval/questions.txt` | The first 25 draft questions, one per line with their type |
| `eval/test_set.json` | The full set of 41 questions with the checks for each one (the drafts are q01 to q25) |
| `eval/hard_set.json` | 15 harder questions (traps, safety, follow-ups, half-answerable, reasoning), added after both prompts scored 41/41 |
| `eval/retrieval_set.json` | 26 search queries with the policy file each should find |
| `eval/results/` | Saved runs; `summary.md` and `summary-hard.md` keep one line per run |
| `eval/rescore.py` | Re-scores a saved run after the scoring rules change, without new AI calls |

```bash
python -m eval.run_retrieval_eval            # search hit rate per layer
python -m eval.run_eval --prompt v1          # full agent eval with prompt v1
python -m eval.run_eval --prompt v2 --delay 3   # pause between questions if you hit rate limits
python -m eval.run_eval --prompt v2 --set hard  # the harder set
python -m eval.rescore eval/results/v2-<time>.json   # score a saved run again
```

## Tracing

With a free [Langfuse](https://langfuse.com) account, set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_HOST` in `.env`. Every chat turn then becomes a trace: each LLM call (with its token count) and each tool call is a step inside it, with its input, output and time, and the trace is tagged with the prompt version. Without the keys, tracing is off and nothing else changes.

![A Langfuse trace of one question: an LLM call, a search_docs call, then the LLM call that writes the answer](docs/images/langfuse-trace.png)

In this trace the document search took 7.8 s of the 9.1 s total, while both LLM calls took under a second, so search was the step to speed up first.

**Search timings, before and after** (2026-11-08, measured stage by stage on CPU, `SEARCH_MODE=rerank`; 3 cold runs and 10 warm searches each):

| | Before | After |
|---|---|---|
| Importing torch and sentence-transformers | 3,294 ms, during the first question | about 2,800 ms, at server startup |
| Loading the two models | 9,928 ms, during the first question | 539 ms, at server startup |
| First search after a start (not counting the import) | 10,568 ms, model loading included | 206 ms |
| Warm search (median) | 212 ms | 200 ms |
| Getting a database connection (warm) | 10.4 ms | 0.8 ms |
| Reranker (warm, per search) | 182 ms | 181 ms |

The import was timed on its own, before the first search, so it isn't inside the other rows. In total, a fresh server used to spend about 14 s on search during its first question; it now spends about 3.5 s at startup (4.6 s in the Docker container) and about 0.2 s per question.

- **The slow part was loading, not searching.** About 9.6 s of the first search was the Hugging Face library checking online for newer versions of two models that were already downloaded: a test run with the library set to offline cut the first search from 10.4 s to 0.75 s. Models now load from the local cache first (and download only when missing), and the server loads them at startup instead of during the first question. The trade-off: a newer upstream revision of a model is only picked up after its cached copy is deleted.
- **Search connections come from a pool** instead of a new connection per search, and each search hands its connection back before reranking.
- **The reranker is now about 90% of a warm search.** `SEARCH_MODE=hybrid` would skip it; on these nine short documents hybrid search already finds the right file every time (see Results), but the default stays `rerank`.
- Retrieval hit rates are unchanged after the change (100% on all three layers).

## Tests

```bash
pytest                       # unit tests: fake LLM, fake search models, no API key needed
TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/lookup_test pytest   # plus database tests
ruff check .
ruff format --check .
```

The integration tests rebuild the tables in the database you point them at, so use a separate one. Create it once with `docker compose exec postgres createdb -U postgres lookup_test`. CI runs lint and both test groups against a real pgvector Postgres on every push to `main` and every pull request.

## Project structure

```
app/
  main.py              FastAPI app: /, /health, /chat, /ingest, correlation ids
  db.py                Postgres connection helpers (owner and read-only user) and the search pool
  static/index.html    the chat page (one file, no build step)
  agent/               LangGraph graph, runner (answer + sources), prompts/v1.txt, v2.txt, v3.txt
  tools/               search_docs, query_database, convert_currency, registry
  retrieval/           chunking, ingestion, vector / hybrid / rerank search
  services/llm.py      LLM client on Steadfast
  core/                settings, JSON logging, Langfuse tracing, the /chat rate limit
db/                    schema.sql, seed.py (repeatable fake shop data)
data/policies/         the nine policy documents
eval/                  test set, scorer, eval runners, re-scorer, results
scripts/               setup_db, ingest_docs, ask, try_tools
tests/                 unit tests and integration tests
docs/architecture.md   design and diagram
CHANGELOG.md           what was built and changed
```

## Settings

All settings come from `.env` (see `.env.example`). The main ones: `LLM_PROVIDER` (`groq` or `openai`, the other one becomes the fallback when its key is set), `LLM_MODEL` (the Groq model; with `LLM_PROVIDER=openai` the model is `OPENAI_MODEL`), `PROMPT_VERSION` (default `v3`), `MAX_AGENT_STEPS`, `SEARCH_MODE` (`vector`, `hybrid` or `rerank`), and the optional Langfuse keys for tracing.

Two settings protect `POST /chat`:

- `CHAT_API_KEY`: when set, every question needs the header `X-API-Key` with this value; a missing or wrong key gets 401. Empty by default, so the chat page and the Docker quick start work without one. The chat page never holds a key, so with a key set it shows "This server needs an API key" and only your own clients can ask.
- `CHAT_RATE_LIMIT_PER_MINUTE` (default 20): questions per client IP in any 60 seconds; over the limit gets 429 with a "Slow down" message and a `Retry-After` header, before any LLM call. `0` turns it off. Its limits are listed under Known limitations.

Less common settings (SQL timeout and row limit, search models, the LLM calls-per-minute limit, temperature) have defaults in `app/core/config.py` and can be overridden the same way.

## Known limitations

- Exchange rates are European Central Bank reference rates, published once per working day, so "today's rate" is the latest published one (the answer shows its date).
- The fake shop data covers January to June 2026 only, and all prices are in USD.
- Eval answer checks are simple text rules (a number, names or key phrases must appear). They can miss a correct answer worded in an unexpected way (this happened three times on the hard set), and they don't catch an extra wrong claim inside an otherwise correct answer.
- Each result comes from one run of one model. The hard set is only 15 questions, so one question is about 7 points.
- Earlier questions and answers in a thread are still sent with every question (only the old tool results are replaced by a placeholder), so a very long conversation still grows. A follow-up that needs the raw data behind an old answer makes the agent call the tool again.
- The eval's "must not mention" checks are plain text matches, so a correct refusal like "no orders were deleted" can fail on "were deleted".
- The `/chat` rate limit is kept in memory: the counts reset when the server restarts, and it only works for a single server process (with several workers or containers, each counts on its own). Clients are told apart by IP, so behind a reverse proxy every request looks like the proxy's IP, and all users share one limit, unless uvicorn is told to trust the proxy's forwarded headers (`--forwarded-allow-ips=<proxy address>`). With Docker port publishing, requests can also arrive from the Docker gateway's IP. For several processes, a shared store like Redis would be needed.
- The API key is one shared key, not per-user accounts.

## License

MIT. Built by [Manish Pandeya](https://github.com/maniesh-lab).
