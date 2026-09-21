import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.config import settings
from app.models.database import init_db
from app.utils.logger import logger
from app.api import video, query, evaluation

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Multimodal Video Understanding & Grounded QA with Temporal Reasoning"
)

# CORS configuration
# CORS configuration — allow all Vercel preview URLs + localhost
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.vercel\.app|http://localhost:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for video and extracted frame playback
app.mount("/data", StaticFiles(directory=str(settings.DATA_DIR)), name="data")

# Register API routers
app.include_router(video.router, prefix="/api/video", tags=["Video Processing"])
app.include_router(query.router, prefix="/api/video", tags=["Grounded QA"])
app.include_router(evaluation.router, prefix="/api/evaluation", tags=["Benchmark & Evaluation"])

@app.on_event("startup")
def on_startup():
    init_db()
    logger.info(f"VideoLens Backend started. Serving on {settings.HOST}:{settings.PORT}")

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "llm_provider": settings.LLM_PROVIDER,
        "sampling_fps": settings.SAMPLING_FPS,
        "temporal_window_sec": settings.TEMPORAL_WINDOW_SECONDS
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
