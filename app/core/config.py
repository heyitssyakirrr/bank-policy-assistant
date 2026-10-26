import os
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
    deepseek_api_key: str = ""  # without it, a DeepSeek role uses its Gemini fallback

    # --- storage
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/bankpolicy"
    qdrant_url: str = "http://localhost:6333"  # becomes http://qdrant:6333 when the app runs in Docker
    qdrant_api_key: str = ""  # Qdrant Cloud only; the local container has no auth
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

    # --- LLM roles: "<provider>:<model>", built with LangChain's init_chat_model.
    # DeepSeek (paid) answers; Gemini is each role's fallback and the judge, so the
    # judge is a different model family from the generator.
    planner_model: str = "deepseek:deepseek-flash"
    reviewer_model: str = "deepseek:deepseek-flash"
    generator_model: str = "deepseek:deepseek-flash"
    judge_model: str = "google_genai:gemini-3.1-flash-lite"
    planner_fallback_model: str = "google_genai:gemini-3.1-flash-lite"
    reviewer_fallback_model: str = "google_genai:gemini-3.1-flash-lite"
    generator_fallback_model: str = "google_genai:gemini-3.5-flash-lite"
    judge_fallback_model: str = "google_genai:gemini-3.5-flash-lite"
    judge_cross_check_model: str = "google_genai:gemini-3.8-flash"  # 20 calls/day: a sample only
    structured_output_method: Literal["json_schema", "function_calling"] = "json_schema"  # DeepSeek: tool calling
    llm_requests_per_minute: int = 15  # per Gemini model (free tier): every role using a model shares it
    deepseek_requests_per_minute: int = 120  # DeepSeek has no fixed rate limit; this only paces bursts
    llm_cooldown_s: float = 120.0  # a model that failed with an overload/outage is tried last for this long
    llm_timeout_s: float = 20.0  # Gemini uses it as the server deadline: a stalled call fails fast and is retried
    # Gemini 3 "thinking", per role (DeepSeek's is always off: see app/llm/models.py). Default thinking made the planner take ~11 s; low keeps
    # routing quality. The generator needs medium to combine linked passages (e-KYC 8.10 -> 8.4).
    planner_thinking_level: Literal["minimal", "low", "medium", "high"] = "low"
    reviewer_thinking_level: Literal["minimal", "low", "medium", "high"] = "low"
    generator_thinking_level: Literal["minimal", "low", "medium", "high"] = "medium"
    judge_thinking_level: Literal["minimal", "low", "medium", "high"] = "low"
    rerank_timeout_s: float = 30.0
    max_question_chars: int = 2000  # the API rejects longer questions (422); scripts cut them

    # --- data files
    chunks_path: Path = Path("app/data/processed/chunks.jsonl")  # hashed into each turn's index_version
    raw_pdf_dir: Path = Path("app/data/raw_pdfs")  # served so citations can open the PDF at the page
    eval_results_dir: Path = Path("evals/results")  # one folder per evaluation run, shown on the Evaluation page

    # --- response cache (development and evaluation; live answers only with LIVE_RESPONSE_CACHE)
    response_cache_path: Path = Path("app/data/interim/response_cache.sqlite")
    live_response_cache: bool = False  # public deployment: a repeated question costs no quota

    # --- quota protection for a public link (0 = no limit)
    daily_question_limit: int = 0  # questions per quota day, all visitors together (from the audit log)
    quota_timezone: str = "America/Los_Angeles"  # Gemini free-tier quotas reset at its midnight
    questions_per_session: int = 0  # questions per visitor's browser session

    # --- Streamlit UI: "api" calls FastAPI over HTTP; "local" runs the agent in the UI's
    # own process (Streamlit Community Cloud, which runs a single process)
    ui_backend: Literal["api", "local"] = "api"
    api_url: str = "http://localhost:8000"  # http://api:8000 inside Docker
    public_api_url: str = "http://localhost:8000"  # the API address the browser uses for PDF links

    # --- LangSmith tracing (optional): every node, prompt, token count and latency
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "bank-policy-assistant"

    log_level: str = "INFO"


settings = Settings()


def enable_tracing(config: Settings = settings) -> bool:
    """LangChain reads its LANGSMITH_* variables from the process environment
    itself, so this hands it our settings (from .env) once at startup, and
    switches tracing off explicitly when it isn't fully configured (a
    container's env_file could otherwise turn it on without a key)."""
    if not (config.langsmith_tracing and config.langsmith_api_key):
        os.environ["LANGSMITH_TRACING"] = "false"
        return False
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = config.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = config.langsmith_project
    return True
