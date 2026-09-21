import gc
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Union
from app.utils.logger import logger

class EmbeddingService:
    def __init__(self):
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._text_model = None

    def _get_text_model(self):
        """Loads SentenceTransformers all-MiniLM-L6-v2 on CPU."""
        if self._text_model is None:
            logger.info("Loading SentenceTransformers model 'all-MiniLM-L6-v2'...")
            from sentence_transformers import SentenceTransformer
            self._text_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            logger.info("SentenceTransformers model loaded (dim: 384).")
        return self._text_model

    def _get_clip_model(self):
        """Loads OpenCLIP ViT-B-32 model on CPU."""
        if self._clip_model is None:
            logger.info("Loading OpenCLIP model 'ViT-B-32' (openai)...")
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', device='cpu')
            tokenizer = open_clip.get_tokenizer('ViT-B-32')
            self._clip_model = model
            self._clip_preprocess = preprocess
            self._clip_tokenizer = tokenizer
            self._clip_model.eval()
            logger.info("OpenCLIP model loaded (dim: 512).")
        return self._clip_model, self._clip_preprocess, self._clip_tokenizer

    # --- SentenceTransformers Text Embeddings (dim: 384) ---
    def embed_text_semantic(self, text: str) -> np.ndarray:
        """Embeds a single text string into 384-dim normalized vector."""
        model = self._get_text_model()
        vec = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype(np.float32)

    def embed_texts_semantic_batch(self, texts: List[str]) -> np.ndarray:
        """Embeds multiple text strings into (N, 384) normalized matrix."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        model = self._get_text_model()
        vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        return vecs.astype(np.float32)

    # --- OpenCLIP Visual & Cross-Modal Embeddings (dim: 512) ---
    def embed_image(self, image_path: Path) -> np.ndarray:
        """Embeds an image into 512-dim normalized visual vector."""
        model, preprocess, _ = self._get_clip_model()
        with Image.open(image_path) as img:
            image_tensor = preprocess(img.convert("RGB")).unsqueeze(0)
        
        with torch.no_grad():
            vec = model.encode_image(image_tensor)
            vec = vec / vec.norm(dim=-1, keepdim=True)
            return vec.cpu().numpy().squeeze(0).astype(np.float32)

    def embed_images_batch(self, image_paths: List[Path]) -> np.ndarray:
        """Embeds multiple images into (N, 512) normalized matrix."""
        if not image_paths:
            return np.empty((0, 512), dtype=np.float32)

        model, preprocess, _ = self._get_clip_model()
        tensors = []
        for p in image_paths:
            with Image.open(p) as img:
                tensors.append(preprocess(img.convert("RGB")))
        batch = torch.stack(tensors)

        with torch.no_grad():
            vecs = model.encode_image(batch)
            vecs = vecs / vecs.norm(dim=-1, keepdim=True)
            return vecs.cpu().numpy().astype(np.float32)

    def embed_text_clip(self, text: str) -> np.ndarray:
        """Embeds text into 512-dim OpenCLIP space for cross-modal image matching."""
        model, _, tokenizer = self._get_clip_model()
        tokens = tokenizer([text])
        with torch.no_grad():
            vec = model.encode_text(tokens)
            vec = vec / vec.norm(dim=-1, keepdim=True)
            return vec.cpu().numpy().squeeze(0).astype(np.float32)

    def unload(self):
        """Release all embedding models from memory to free RAM after pipeline use."""
        if self._clip_model is not None:
            del self._clip_model
            del self._clip_preprocess
            del self._clip_tokenizer
            self._clip_model = None
            self._clip_preprocess = None
            self._clip_tokenizer = None
        if self._text_model is not None:
            del self._text_model
            self._text_model = None
        gc.collect()
        logger.info("Embedding models (CLIP + SentenceTransformers) unloaded from memory.")

embedding_service = EmbeddingService()
