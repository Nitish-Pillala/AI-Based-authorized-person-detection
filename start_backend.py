"""Entry point to start the FastAPI backend server."""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from backend.config import settings


if __name__ == "__main__":
    print(f"Starting CCTV Backend on http://{settings.api_host}:{settings.api_port}")
    print(f"API Docs: http://localhost:{settings.api_port}/docs")

    uvicorn.run(
        "backend.api.fastapi_server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )
