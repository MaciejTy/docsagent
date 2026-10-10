from collections.abc import Callable
from functools import lru_cache

import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
DIMENSION = 384

Embedder = Callable[[list[str]], np.ndarray]


@lru_cache(maxsize=1)
def get_model():
    # Import w środku: ładowanie torch trwa kilka sekund, a testy go nie potrzebują.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def embed(texts: list[str]) -> np.ndarray:
    vectors = get_model().encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vectors.astype(np.float32)


def to_bytes(vector) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def from_bytes(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32)
