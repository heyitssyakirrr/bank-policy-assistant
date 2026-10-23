from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every tunable value in one place. Each can be overridden in .env
    (e.g. RETRIEVAL_CANDIDATES=50) without touching code."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- secrets
    gemini_api_key: str = ""
    cohere_api_key: str = ""

    # --- storage
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/bankpolicy"
    qdrant_url: str = "http://localhost:6333"  # becomes http://qdrant:6333 when the app runs in Docker
    qdrant_collection: str = "policy_chunks"  # one collection per embedding model (see indexer)

    # --- embeddings ("gemini-embedding-001" or "gemini-embedding-2")
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768

    # --- retrieval
    retrieval_candidates: int = 30  # how many results EACH search (dense, BM25) returns before fusion
    rrf_k: int = 60  # RRF constant; 60 is the standard value from the original paper

    # --- reranking (Cohere)
    rerank_model: str = "rerank-v4.0-pro"  # "rerank-v4.0-fast" trades some quality for latency
    rerank_top_n: int = 5  # chunks kept after reranking = what the LLM will read
    rerank_calls_per_minute: int = 10  # Cohere trial key limit
    default_sector: str = "banking"  # the users are bank staff: prefer the banking copy of repeated rules

    # --- LLM roles: "<provider>:<model>", built with LangChain's init_chat_model
    planner_model: str = "google_genai:gemini-3.1-flash-lite"
    reviewer_model: str = "google_genai:gemini-3.1-flash-lite"
    generator_model: str = "google_genai:gemini-3.5-flash-lite"
    judge_model: str = "google_genai:gemini-3.1-flash-lite"
    planner_fallback_model: str = "google_genai:gemini-3.5-flash-lite"
    reviewer_fallback_model: str = "google_genai:gemini-3.5-flash-lite"
    generator_fallback_model: str = "google_genai:gemini-3.1-flash-lite"
    judge_fallback_model: str = "google_genai:gemini-3.5-flash-lite"
    structured_output_method: Literal["json_schema", "function_calling"] = "json_schema"
    llm_requests_per_minute: int = 15  # per model: every role using a model shares its limit
    llm_timeout_s: float = 30.0
    rerank_timeout_s: float = 30.0
    max_question_chars: int = 2000  # longer input is cut: no policy question needs more

    # --- response cache (development and evaluation only; the API never enables it)
    response_cache_path: Path = Path("app/data/interim/response_cache.sqlite")

    log_level: str = "INFO"


settings = Settings()
