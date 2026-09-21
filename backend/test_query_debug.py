from app.models.database import get_db

with get_db() as conn:
    c = conn.cursor()
    c.execute('SELECT chunk_id, start_time, end_time, text FROM transcripts WHERE video_id = ?', ('cdc236fb-84c',))
    print("=== TRANSCRIPTS ===")
    for row in c.fetchall():
        print(dict(row))
    c.execute('SELECT ocr_id, timestamp, text FROM ocr_chunks WHERE video_id = ?', ('cdc236fb-84c',))
    print("=== OCR CHUNKS ===")
    for row in c.fetchall():
        print(dict(row))
