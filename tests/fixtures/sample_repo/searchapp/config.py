"""Runtime settings loaded from the environment."""
import os


class Settings:
    def __init__(self):
        self.db_path = os.getenv("SEARCH_DB_PATH", "index.db")
        self.embedding_url = os.environ.get("EMBEDDING_URL", "http://localhost:8080")
        self.api_token = os.environ["API_TOKEN"] if "API_TOKEN" in os.environ else ""


def load_settings() -> Settings:
    return Settings()
