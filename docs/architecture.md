# Architecture

lookup-agent answers questions about a small online shop by looking things up first. It has three places to look:

| Source | What's in it | Tool |
|---|---|---|
| Policy documents | 9 Markdown files: returns, refunds, shipping, warranty... | `search_docs` |
| Postgres database | customers, products, orders, order items, support tickets | `query_database` |
| Live API | exchange rates (Frankfurter, European Central Bank rates) | `convert_currency` |

The model decides which tools to call, can call several in a row (for example: look up an order total, then convert it to CAD), and the answer comes back with a list of the sources it used.

## The big picture

```mermaid
flowchart LR
    user([User]) -->|POST /chat| api[FastAPI<br/>correlation id per request]
    api --> agent

    subgraph agent [LangGraph agent]
        direction TB
        A[agent node<br/>LLM call] -->|asked for tools| T[tools node]
        T --> A
        A -->|plain answer or step cap| done([answer + sources])
    end

    A <-->|Steadfast: retries, rate limit,<br/>fallback provider, token counts| llm[(Groq / OpenAI)]

    T --> S[search_docs]
    T --> Q[query_database]
    T --> C[convert_currency]

    S -->|vector + keyword, RRF,<br/>cross-encoder rerank| pg[(Postgres + pgvector)]
    Q -->|read-only user,<br/>timeout, row limit| pg
    C -->|HTTPS, timeout| fx[(Frankfurter API)]
    agent -->|checkpointer: memory per thread_id| pg

    agent -.->|optional| lf[(Langfuse traces<br/>with prompt version)]
```

## One request, step by step

1. **FastAPI** receives `POST /chat` with a message and an optional `thread_id`. A middleware gives the request a correlation id (or keeps the caller's `X-Request-ID`) and binds it to every log line, including Steadfast's logs of each LLM call.
2. **The agent node** sends the system prompt (`app/agent/prompts/v2.txt`), the conversation so far and the three tool descriptions to the LLM through **Steadfast**.
3. If the model asks for tools, **the tools node** runs each one (in a thread, so the server isn't blocked) and adds the results to the conversation. Bad arguments, SQL errors and API failures come back to the model as short error messages it can act on, never as crashes.
4. Steps 2 and 3 repeat until the model answers in plain text. A **step cap** (`MAX_AGENT_STEPS`, default 6) stops a loop: at the cap, tools are switched off and the model is told to answer with what it has.
5. The **checkpointer** saves the conversation in Postgres under its `thread_id`, so sending the same `thread_id` again continues the conversation, even after a restart.
6. The runner collects the answer, which tools were used, and a **sources** list built from the tool results (policy file and section, the SQL that ran, the exchange rate and its date).

## The pieces

### Search (`app/retrieval/`)

- **Ingestion** splits each policy file into one chunk per `##` section (long sections are split at paragraphs), puts the heading in front of the text, embeds it with `BAAI/bge-small-en-v1.5` (384 numbers per chunk) and stores it in the `chunks` table. A file whose sha256 hasn't changed is skipped.
- **Search** has three layers, each measured by `eval/run_retrieval_eval.py`:
  1. *vector*: nearest chunks by cosine distance (pgvector, HNSW index)
  2. *hybrid*: vector results plus Postgres full-text search, merged with Reciprocal Rank Fusion
  3. *rerank*: the hybrid top 20 re-scored by a cross-encoder (`ms-marco-MiniLM-L-6-v2`)

### The SQL tool (`app/tools/query_database.py`)

The model writes SQL itself, so the tool is locked down in layers:

1. Only one `SELECT`/`WITH` statement, with no write keywords
2. It connects as `lookup_reader`, which can only `SELECT` from the five shop tables (not the search or memory tables)
3. Read-only transactions and a 5-second statement timeout, set both on the role and on the connection
4. At most 100 rows come back, and the result says when it was cut

The tool description includes a short schema guide (tables, columns, allowed values). An integration test checks the guide against the real database so they can't drift apart.

### The LLM client (`app/services/llm.py`)

Built on [Steadfast](https://github.com/maniesh-lab/steadfast), a reliability layer for LLM APIs. It retries busy or failing calls with backoff, keeps calls under a per-provider rate limit, falls back to the second provider when the first keeps failing, counts tokens, and uses one message/tool format for Groq and OpenAI. When every provider fails, `/chat` returns a 503 with a clear message.

### Data (`db/`)

`db/schema.sql` creates the tables, indexes and the read-only role. `db/seed.py` fills them with a fake shop: about 160 customers, 34 products, 460 orders and 80 support tickets between January and June 2026. Every random choice comes from one seeded generator and the name lists live in the file, so every run produces the same rows (it prints a fingerprint to prove it). A few cases are planted on purpose because the eval asks about them.

### Evals (`eval/`)

- `test_set.json`: 41 questions across five types: DOCS, DB, API, MIXED (several sources) and NONE (the data doesn't exist and the agent should say so). Database answers aren't hard-coded: the scorer runs each case's "truth SQL" against the seeded database. Currency answers are checked against a live rate fetched during the run.
- `run_eval.py`: runs the set through the real agent for a given prompt version and records pass rate, tool choice and tokens in `eval/results/`.
- `run_retrieval_eval.py`: hit rate @5 and MRR for each search layer.

### Observability

- **Logs**: one JSON object per line (structlog), every line tagged with the request's correlation id.
- **Traces** (optional): with Langfuse keys set, each chat turn is a trace with one child per LLM call and per tool call. The prompt version is stored on every trace as its version, a tag and metadata, so v1 and v2 runs can be compared.

## Design choices

- **Plain OpenAI-format messages in the graph state**, not LangChain message objects: Steadfast already speaks that format for every provider, so there's no conversion layer.
- **Tools return text, never raise**: the model can read an error and retry, which matters most for SQL.
- **Truth computed at eval time**: changing the seed can't silently break the expected answers.
- **Separate read-only database user** instead of trusting a keyword filter alone.
