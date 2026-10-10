import numpy as np
from sqlmodel import Session, select

from embeddings import from_bytes
from models import Chunk, Document, SearchResult


def search(session: Session, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
    rows = session.exec(select(Chunk, Document.path).join(Document)).all()
    if not rows:
        return []

    matrix = np.stack([from_bytes(chunk.embedding) for chunk, _ in rows])
    scores = matrix @ np.asarray(query_vector, dtype=np.float32)
    best = np.argsort(-scores, kind="stable")[:top_k]

    return [
        SearchResult(
            chunk_id=rows[i][0].id,
            source=rows[i][1],
            heading_path=rows[i][0].heading_path,
            text=rows[i][0].text,
            score=float(scores[i]),
        )
        for i in best
    ]
