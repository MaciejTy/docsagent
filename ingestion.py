import hashlib
import sys
from pathlib import Path

from sqlmodel import Session, delete, select

from chunking import chunk_markdown
from db import create_db, engine
from embeddings import Embedder, embed, to_bytes
from models import Chunk, Document, utc_now


def ingest_file(session: Session, path: Path, embedder: Embedder) -> int | None:
    content = path.read_text(encoding="utf-8")
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    key = path.as_posix()

    document = session.exec(select(Document).where(Document.path == key)).first()
    if document is not None and document.content_hash == content_hash:
        return None

    drafts = chunk_markdown(content)
    vectors = embedder([d.embedding_input() for d in drafts]) if drafts else []

    if document is None:
        document = Document(path=key, content_hash=content_hash)
        session.add(document)
        session.flush()
    else:
        session.exec(delete(Chunk).where(Chunk.document_id == document.id))
        document.content_hash = content_hash
        document.ingested_at = utc_now()
        session.add(document)

    for draft, vector in zip(drafts, vectors):
        session.add(Chunk(
            document_id=document.id,
            heading_path=draft.heading_path,
            text=draft.text,
            embedding=to_bytes(vector),
        ))
    session.commit()
    return len(drafts)


def ingest_folder(session: Session, folder: Path, embedder: Embedder) -> dict[str, str]:
    report = {}
    for path in sorted(folder.rglob("*.md")):
        try:
            count = ingest_file(session, path, embedder)
        except (OSError, UnicodeDecodeError) as error:
            session.rollback()
            report[path.as_posix()] = f"error: {error}"
            continue
        report[path.as_posix()] = "unchanged" if count is None else f"indexed {count} chunks"
    return report


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python ingestion.py <folder>")
        return 2
    folder = Path(argv[1])
    if not folder.is_dir():
        print(f"error: {folder} is not a directory")
        return 1

    create_db()
    with Session(engine) as session:
        report = ingest_folder(session, folder, embed)
    for path, status in report.items():
        print(f"{path}: {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
