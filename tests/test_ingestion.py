from sqlmodel import select

from embeddings import DIMENSION, from_bytes
from ingestion import ingest_file, ingest_folder, main
from models import Chunk, Document


def write(folder, name, content):
    path = folder / name
    path.write_text(content, encoding="utf-8")
    return path


def chunks_of(session, path):
    return session.exec(
        select(Chunk).join(Document).where(Document.path == path.as_posix())
    ).all()


def test_ingest_file_stores_chunks(session, tmp_path, fake_embedder):
    path = write(tmp_path, "guide.md", "# Guide\n\nfirst\n\n## Install\n\nsecond\n")

    assert ingest_file(session, path, fake_embedder) == 2

    chunks = chunks_of(session, path)
    assert [c.heading_path for c in chunks] == ["Guide", "Guide > Install"]
    assert [c.text for c in chunks] == ["first", "second"]
    assert len(from_bytes(chunks[0].embedding)) == DIMENSION


def test_unchanged_file_is_skipped(session, tmp_path, fake_embedder):
    path = write(tmp_path, "guide.md", "# Guide\n\nfirst\n\n## Install\n\nsecond\n")
    ingest_file(session, path, fake_embedder)

    assert ingest_file(session, path, fake_embedder) is None
    assert fake_embedder.calls == 1
    assert len(chunks_of(session, path)) == 2


def test_changed_file_replaces_old_chunks(session, tmp_path, fake_embedder):
    path = write(tmp_path, "guide.md", "# Guide\n\nfirst\n\n## Install\n\nsecond\n")
    ingest_file(session, path, fake_embedder)
    write(tmp_path, "guide.md", "# Guide\n\nonly one now\n")

    assert ingest_file(session, path, fake_embedder) == 1
    assert [c.text for c in chunks_of(session, path)] == ["only one now"]
    assert len(session.exec(select(Document)).all()) == 1


def test_ingest_folder_reports_each_file_and_survives_bad_one(session, tmp_path, fake_embedder):
    good = write(tmp_path, "good.md", "# Good\n\ntext\n")
    bad = tmp_path / "bad.md"
    bad.write_bytes(b"\xff\xfe not utf-8")
    (tmp_path / "nested").mkdir()
    deep = write(tmp_path / "nested", "deep.md", "# Deep\n\ntext\n")
    write(tmp_path, "notes.txt", "not markdown")

    report = ingest_folder(session, tmp_path, fake_embedder)

    assert len(report) == 3
    assert report[good.as_posix()] == "indexed 1 chunks"
    assert report[deep.as_posix()] == "indexed 1 chunks"
    assert report[bad.as_posix()].startswith("error:")


def test_main_rejects_missing_folder(tmp_path, capsys):
    assert main(["ingestion.py", str(tmp_path / "nope")]) == 1
    assert "not a directory" in capsys.readouterr().out
