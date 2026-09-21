import sqlite3
from pathlib import Path
from contextlib import contextmanager
from app.config import settings
from app.utils.logger import logger

def init_db():
    db_path = settings.DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Videos
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                duration REAL NOT NULL,
                fps REAL NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'uploaded',
                stage TEXT NOT NULL DEFAULT 'ready',
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Sampled Frames
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS frames (
                frame_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                frame_idx INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                image_path TEXT NOT NULL,
                is_keyframe INTEGER NOT NULL DEFAULT 1,
                difference_score REAL DEFAULT 0.0,
                FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
            )
        """)
        
        # Audio Transcript Chunks
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transcripts (
                chunk_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL NOT NULL,
                text TEXT NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
            )
        """)
        
        # OCR Chunks
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ocr_chunks (
                ocr_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                frame_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                text TEXT NOT NULL,
                confidence REAL NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE,
                FOREIGN KEY (frame_id) REFERENCES frames(frame_id) ON DELETE CASCADE
            )
        """)
        
        # Grounded QA History
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                query_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                confidence REAL NOT NULL,
                evidence_json TEXT NOT NULL,
                timestamps_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
            )
        """)
        
        # Evaluation Runs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                eval_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                experiment_type TEXT NOT NULL,
                question TEXT NOT NULL,
                recall_at_k REAL NOT NULL,
                grounding_iou REAL NOT NULL,
                latency_seconds REAL NOT NULL,
                metadata_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        logger.info(f"Database initialized at {db_path}")

@contextmanager
def get_db():
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
