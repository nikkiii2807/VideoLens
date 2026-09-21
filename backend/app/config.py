import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

import shutil

# Ensure Winget packages are in os.environ["PATH"]
winget_bins = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Packages/Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.1-essentials_build/bin",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Packages/OpenJS.NodeJS.LTS_Microsoft.Winget.Source_8wekyb3d8bbwe/node-v24.19.0-win-arm64",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/node",
    Path(os.environ.get("LOCALAPPDATA", "")) / "bin",
    Path.home() / ".local/bin",
]
for p in winget_bins:
    if p.exists() and str(p) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{p};{os.environ.get('PATH', '')}"

FFMPEG_BIN = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE_BIN = shutil.which("ffprobe") or "ffprobe"

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"

# Load environment variables
load_dotenv(BASE_DIR / ".env")

class Settings(BaseModel):
    # App
    APP_NAME: str = "VideoLens — Multimodal Video Understanding & Grounded QA"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Limits & Validation
    MAX_DURATION_SECONDS: float = float(os.getenv("MAX_DURATION_SECONDS", "60.0"))
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    ALLOWED_EXTENSIONS: list[str] = [".mp4", ".mov", ".avi", ".webm", ".mkv"]

    # Frame Sampling & Redundancy Filtering
    SAMPLING_FPS: float = float(os.getenv("SAMPLING_FPS", "2.0"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.92")) # Cosine/SSIM redundancy threshold

    # Temporal Reasoning
    TEMPORAL_WINDOW_SECONDS: float = float(os.getenv("TEMPORAL_WINDOW_SECONDS", "4.0"))
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))

    # Modality Weights for Multimodal Scoring
    WEIGHT_TEXT: float = float(os.getenv("WEIGHT_TEXT", "0.35"))
    WEIGHT_VISUAL: float = float(os.getenv("WEIGHT_VISUAL", "0.35"))
    WEIGHT_OCR: float = float(os.getenv("WEIGHT_OCR", "0.20"))
    WEIGHT_TEMPORAL: float = float(os.getenv("WEIGHT_TEMPORAL", "0.10"))

    # Models & Providers
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini") # "gemini", "openai", "offline"
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Storage Dirs
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = DATA_DIR
    UPLOAD_DIR: Path = UPLOAD_DIR
    PROCESSED_DIR: Path = PROCESSED_DIR
    DB_PATH: Path = DATA_DIR / "videolens.db"

settings = Settings()

# Ensure directories exist
for directory in [DATA_DIR, UPLOAD_DIR, PROCESSED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
