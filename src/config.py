import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = BASE_DIR / "platform.db"
DEFAULT_MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"

# Configuration settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
QDRANT_HOST = os.getenv("QDRANT_HOST", ":memory:")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
DB_URL = os.getenv("DB_URL", f"sqlite:///{DEFAULT_DB_PATH.as_posix()}")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{DEFAULT_MLFLOW_DB_PATH.as_posix()}")
SECRET_KEY = os.getenv("SECRET_KEY", "customer-support-platform-jwt-secret-key-2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Mock mode flag: active if no OpenAI API Key is provided
IS_MOCK_MODE = not bool(OPENAI_API_KEY)
