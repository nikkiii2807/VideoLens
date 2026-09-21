import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.frame_sampler import frame_sampler
from app.services.audio_processor import audio_processor
from app.services.transcription import transcription_service
from app.services.ocr_service import ocr_service
from app.services.embedding_service import embedding_service
from app.services.vector_store import VectorStore
from app.models.database import init_db

def test_embeddings_and_faiss():
    init_db()
    sample_path = Path(__file__).resolve().parent.parent.parent / "demo_data" / "sample_lecture.mp4"
    assert sample_path.exists()
    
    test_video_id = "test_faiss_vid_001"
    
    # 1. Sample Frames
    frames = frame_sampler.extract_frames(test_video_id, sample_path)
    assert len(frames) > 0
    
    # 2. Audio & Transcribe
    wav_path = audio_processor.extract_audio(test_video_id, sample_path)
    transcripts = transcription_service.transcribe(test_video_id, wav_path)
    
    # 3. OCR
    ocr_chunks = ocr_service.extract_text_from_frames(test_video_id, frames)
    
    # 4. Generate Embeddings
    print("Embedding visual frames with OpenCLIP...")
    frame_paths = [Path(f["image_path"]) for f in frames]
    visual_embeddings = embedding_service.embed_images_batch(frame_paths)
    assert visual_embeddings.shape == (len(frames), 512)
    
    # Check normalization
    norms = np.linalg.norm(visual_embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), "Visual embeddings must be L2-normalized"

    # Text Chunks (Transcript + OCR)
    text_chunks_meta = []
    text_strings = []
    
    for t in transcripts:
        text_chunks_meta.append({
            "type": "transcript",
            "id": t["chunk_id"],
            "timestamp": t["start_time"],
            "end_time": t["end_time"],
            "text": t["text"]
        })
        text_strings.append(t["text"])

    for o in ocr_chunks:
        text_chunks_meta.append({
            "type": "ocr",
            "id": o["ocr_id"],
            "frame_id": o["frame_id"],
            "timestamp": o["timestamp"],
            "text": o["text"],
            "confidence": o["confidence"]
        })
        text_strings.append(o["text"])

    print(f"Embedding {len(text_strings)} text chunks with SentenceTransformers...")
    text_embeddings = embedding_service.embed_texts_semantic_batch(text_strings)
    assert text_embeddings.shape == (len(text_strings), 384)

    # 5. Build FAISS Indices
    store = VectorStore(test_video_id)
    store.build_indices(visual_embeddings, frames, text_embeddings, text_chunks_meta)
    
    # 6. Test Cross-Modal Visual Search ("red object")
    q_clip = embedding_service.embed_text_clip("red object circular shape")
    vis_results = store.search_visual(q_clip, top_k=3)
    print("Top visual results for 'red object circular shape':")
    for r in vis_results:
        print(f"  Frame {r['frame_id']} @ {r['timestamp']}s (similarity: {r['similarity']:.3f})")
    assert len(vis_results) > 0
    # The first frame (0.0s) has the red object
    assert any(r["timestamp"] <= 3.0 for r in vis_results[:2]), "Red object query should match early frames"

    # 7. Test Semantic Text Search ("Gaussian filtering and convolution")
    q_text = embedding_service.embed_text_semantic("Gaussian filtering and convolution")
    text_results = store.search_text(q_text, top_k=3)
    print("Top text results for 'Gaussian filtering and convolution':")
    for r in text_results:
        print(f"  [{r['type']}] @ {r['timestamp']}s: '{r['text']}' (similarity: {r['similarity']:.3f})")
    assert len(text_results) > 0
    assert any("gaussian" in r["text"].lower() for r in text_results), "Expected Gaussian match"

    # 8. Test Reloading from Disk
    store_reloaded = VectorStore(test_video_id)
    loaded = store_reloaded.load_indices()
    assert loaded, "Should successfully load from disk"
    reloaded_results = store_reloaded.search_visual(q_clip, top_k=3)
    assert len(reloaded_results) == len(vis_results)
    assert np.isclose(reloaded_results[0]["similarity"], vis_results[0]["similarity"])

    print("ALL TEST_EMBEDDING_AND_FAISS CHECKS PASSED!")

if __name__ == "__main__":
    test_embeddings_and_faiss()
