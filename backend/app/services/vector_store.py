import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from app.config import settings
from app.utils.logger import logger

class VectorStore:
    def __init__(self, video_id: str):
        self.video_id = video_id
        self.index_dir = settings.DATA_DIR / "videos" / video_id / "index"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.visual_index_path = self.index_dir / "visual.faiss"
        self.text_index_path = self.index_dir / "text.faiss"
        self.metadata_path = self.index_dir / "metadata.json"

        # Indices: Inner Product (Cosine similarity for L2-normalized vectors)
        self.visual_index: Optional[faiss.IndexFlatIP] = None
        self.text_index: Optional[faiss.IndexFlatIP] = None

        # Metadata storage
        # visual_metadata: list of {frame_id, timestamp, image_path}
        # text_metadata: list of {type: "transcript"|"ocr", id, timestamp, end_time, text}
        self.visual_metadata: List[Dict[str, Any]] = []
        self.text_metadata: List[Dict[str, Any]] = []

    def build_indices(
        self,
        visual_embeddings: np.ndarray,
        visual_meta: List[Dict[str, Any]],
        text_embeddings: np.ndarray,
        text_meta: List[Dict[str, Any]]
    ):
        """Builds and persists visual and text FAISS indices."""
        # 1. Visual Index (dim: 512)
        if len(visual_embeddings) > 0:
            dim_vis = visual_embeddings.shape[1]
            self.visual_index = faiss.IndexFlatIP(dim_vis)
            self.visual_index.add(visual_embeddings)
            self.visual_metadata = visual_meta
            faiss.write_index(self.visual_index, str(self.visual_index_path))

        # 2. Text Index (dim: 384)
        if len(text_embeddings) > 0:
            dim_text = text_embeddings.shape[1]
            self.text_index = faiss.IndexFlatIP(dim_text)
            self.text_index.add(text_embeddings)
            self.text_metadata = text_meta
            faiss.write_index(self.text_index, str(self.text_index_path))

        # 3. Save metadata
        meta_payload = {
            "video_id": self.video_id,
            "visual_metadata": self.visual_metadata,
            "text_metadata": self.text_metadata
        }
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2)

        logger.info(
            f"FAISS indices built for {self.video_id}: "
            f"visual={len(self.visual_metadata)} vectors, text={len(self.text_metadata)} vectors."
        )

    def load_indices(self) -> bool:
        """Loads saved FAISS indices and metadata from disk."""
        if not self.metadata_path.exists():
            return False

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.visual_metadata = data.get("visual_metadata", [])
            self.text_metadata = data.get("text_metadata", [])

        if self.visual_index_path.exists() and len(self.visual_metadata) > 0:
            self.visual_index = faiss.read_index(str(self.visual_index_path))

        if self.text_index_path.exists() and len(self.text_metadata) > 0:
            self.text_index = faiss.read_index(str(self.text_index_path))

        return True

    def search_visual(self, query_clip_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Searches visual index using cross-modal CLIP text embedding."""
        if self.visual_index is None:
            self.load_indices()
        if self.visual_index is None or self.visual_index.ntotal == 0:
            return []

        k = min(top_k, self.visual_index.ntotal)
        query_vec = np.ascontiguousarray(query_clip_embedding.reshape(1, -1), dtype=np.float32)
        scores, indices = self.visual_index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.visual_metadata):
                continue
            item = dict(self.visual_metadata[idx])
            item["similarity"] = float(score)
            item["modality"] = "visual"
            results.append(item)
        return results

    def search_text(self, query_text_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Searches text index (transcripts + OCR) using SentenceTransformers embedding."""
        if self.text_index is None:
            self.load_indices()
        if self.text_index is None or self.text_index.ntotal == 0:
            return []

        k = min(top_k, self.text_index.ntotal)
        query_vec = np.ascontiguousarray(query_text_embedding.reshape(1, -1), dtype=np.float32)
        scores, indices = self.text_index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.text_metadata):
                continue
            item = dict(self.text_metadata[idx])
            item["similarity"] = float(score)
            item["modality"] = item.get("type", "text")
            results.append(item)
        return results
