from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All settings, read from environment variables or the .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    database_url: str
    # The agent's SQL tool connects as a separate read-only user.
    readonly_database_url: str = "postgresql://lookup_reader:lookup_reader@localhost:5432/lookup_db"
    sql_timeout_ms: int = 5000
    sql_row_limit: int = 100

    # LLM (through Steadfast)
    llm_provider: str = "groq"  # "groq" or "openai"
    llm_model: str = "openai/gpt-oss-20b"
    openai_model: str = "gpt-4o-mini"
    groq_api_key: str | None = None
    openai_api_key: str | None = None
    llm_calls_per_minute: int = 30
    llm_temperature: float = 0.0

    # Agent
    prompt_version: str = "v2"
    max_agent_steps: int = 6

    # Search
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    search_mode: str = "rerank"  # "vector", "hybrid" or "rerank"
    search_top_k: int = 5
    policies_dir: str = "data/policies"

    # External API (exchange rates)
    fx_api_url: str = "https://api.frankfurter.dev/v1"
    fx_api_timeout: float = 10.0

    # Tracing (optional: tracing is off while these are empty)
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str | None = None

    # API
    ingest_token: str | None = None  # if set, POST /ingest needs X-Ingest-Token
    log_level: str = "INFO"


settings = Settings()
