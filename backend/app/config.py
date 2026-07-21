"""
Application configuration
"""

import os
from pathlib import Path


class Settings:
    """Application settings"""

    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # Paths — BASE_DIR is the monorepo root (parent of backend/)
    # In Docker the layout is /app/{backend,data,uploads} so BASE_DIR=/app
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads")))
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
    HAZARD_LAYERS_CSV: Path = Path(
        os.getenv("HAZARD_LAYERS_CSV", str(DATA_DIR / "hazard_layers.csv"))
    )

    # Ensure directories exist
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # File upload settings
    MAX_UPLOAD_SIZE: int = int(
        os.getenv("MAX_UPLOAD_SIZE", str(100 * 1024 * 1024))
    )  # 100 MB default
    ALLOWED_EXTENSIONS: set = {".shp", ".gpkg", ".zip", ".csv", ".geojson"}


settings = Settings()
