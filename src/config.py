"""
src/config.py
Centralised configuration — loads .env and exposes typed settings.
Uses pydantic-settings so every variable is validated at startup.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",         # ignore unknown env vars
    )

    # --- Database ---
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "banking_intelligence"
    db_user: str = "postgres"
    db_password: str = "postgres"
    database_url: str = "postgresql://postgres:postgres@localhost:5432/banking_intelligence"

    # --- OpenAI ---
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # --- App ---
    app_env: str = "development"
    log_level: str = "INFO"
    secret_key: str = "changeme"

    # --- Data Paths ---
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")

    # --- Model settings ---
    random_seed: int = 42

    @property
    def customer_csv(self) -> Path:
        return self.raw_data_dir / "customer_data.csv"

    @property
    def transactions_csv(self) -> Path:
        return self.raw_data_dir / "transactions_data.csv"

    @property
    def customer_clean_parquet(self) -> Path:
        return self.processed_data_dir / "customers_clean.parquet"

    @property
    def transactions_clean_parquet(self) -> Path:
        return self.processed_data_dir / "transactions_clean.parquet"

    @property
    def customer_features_parquet(self) -> Path:
        return self.processed_data_dir / "customer_features.parquet"


# Singleton — import this everywhere
settings = Settings()
