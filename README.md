# lookup-agent

[![CI](https://github.com/maniesh-lab/lookup-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/maniesh-lab/lookup-agent/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/maniesh-lab/lookup-agent/branch/main/graph/badge.svg)](https://codecov.io/gh/maniesh-lab/lookup-agent)

**A chatbot that looks things up before it answers.**

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
    user([User]) -->|POST /chat| api[FastAPI]
    api --> agent[LangGraph agent<br/>step cap + memory]
    agent <-->|Steadfast| llm[(Groq / OpenAI)]
    agent --> S[search_docs] --> pg[(Postgres + pgvector)]
    agent --> Q[query_database] --> pg
    agent --> C[convert_currency] --> fx[(Frankfurter API)]
```

The agent loops between the LLM and the tools until it has an answer, with a step cap so it can't loop forever. Conversations are saved in Postgres per `thread_id`. LLM calls go through [Steadfast](https://github.com/maniesh-lab/steadfast), my reliability layer for LLM APIs (retries, rate limits, provider fallback, token counts). The full design is in [docs/architecture.md](docs/architecture.md).

## Results

Measured with the scripts in `eval/`; numbers go here only after a real run.

| What | Script | Result |
|---|---|---|
| Search hit rate @5 (vector / hybrid / reranked) | `python -m eval.run_retrieval_eval` | not run yet |
| Agent eval, prompt v1 (41 questions) | `python -m eval.run_eval --prompt v1` | not run yet |
| Agent eval, prompt v2 (41 questions) | `python -m eval.run_eval --prompt v2` | not run yet |

The eval has five question types: DOCS, DB, API, MIXED (several sources in one question) and NONE (the answer isn't in any source). Expected database answers are computed from the database at run time and currency answers from a live rate, so the checks can't go stale. Each run is saved in `eval/results/`.

## Quick start (Docker)

You need Docker and a free [Groq API key](https://console.groq.com).

```bash
git clone https://github.com/maniesh-lab/lookup-agent
cd lookup-agent
cp .env.example .env          # then put your key after GROQ_API_KEY=
docker compose up --build
```

The first start builds the database (tables, fake shop data, policy search index) and downloads the search models, so it takes a few minutes. Then open http://localhost:8000/docs and try `POST /chat`:

```json
{"message": "What was the total of the order Priya Sharma placed on February 18, 2026, in Canadian dollars at today's rate?"}
```

The reply has `answer`, `sources`, `tools_used` and a `thread_id`. Send the `thread_id` back with your next message to continue the same conversation.

## Run it without Docker (for development)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                     # add GROQ_API_KEY
docker compose up -d postgres            # just the database
python -m scripts.setup_db               # schema + seed data + policy ingestion
uvicorn app.main:app --reload            # API on http://127.0.0.1:8000
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
| `GET` | `/health` | `{"status": "ok"}` |
| `POST` | `/chat` | Ask a question. Body: `message`, optional `thread_id` |
| `POST` | `/ingest` | Re-index the policy documents (unchanged files are skipped). Needs `X-Ingest-Token` if `INGEST_TOKEN` is set |

Every response carries an `X-Request-ID` header. The same id is on every log line for that request.

## Safety of the SQL tool

The model writes its own SQL, so it runs inside four layers: only single `SELECT` statements pass the check; it connects as a database user that can only read the five shop tables; every transaction is read-only with a 5-second timeout; and at most 100 rows come back.

## Tests

```bash
pytest                       # unit tests: fake LLM, fake search models, no API key needed
TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/lookup_test pytest   # plus database tests
ruff check .
```

The integration tests rebuild the tables in the database you point them at, so use a separate one. CI runs lint and both test groups against a real pgvector Postgres on every push.

## Project structure

```
app/
  main.py              FastAPI app: /health, /chat, /ingest, correlation ids
  agent/               LangGraph graph, runner (answer + sources), prompts/v1.txt, v2.txt
  tools/               search_docs, query_database, convert_currency, registry
  retrieval/           chunking, ingestion, vector / hybrid / rerank search
  services/llm.py      LLM client on Steadfast
  core/                settings, JSON logging, Langfuse tracing
db/                    schema.sql, seed.py (repeatable fake shop data)
data/policies/         the nine policy documents
eval/                  test set, scorer, eval runners, results
scripts/               setup_db, ingest_docs, ask, try_tools
tests/                 unit tests and integration tests
docs/architecture.md   design and diagram
```

## Settings

All settings come from `.env` (see `.env.example`). The main ones: `LLM_PROVIDER` (`groq` or `openai`, the other one becomes the fallback when its key is set), `LLM_MODEL`, `PROMPT_VERSION`, `MAX_AGENT_STEPS`, `SEARCH_MODE` (`vector`, `hybrid` or `rerank`), and the optional Langfuse keys for tracing.

## Known limitations

- Exchange rates are European Central Bank reference rates, published once per working day, so "today's rate" is the latest published one (the answer shows its date).
- The fake shop data covers January to June 2026 only, and all prices are in USD.
- Eval answer checks are simple text rules (a number, names or key phrases must appear), which can miss a correct answer worded in an unexpected way.
- Steadfast 0.3 (with tool calling) is installed from GitHub until it's published to PyPI.

## License

MIT. Built by [Manish Pandeya](https://github.com/maniesh-lab).
