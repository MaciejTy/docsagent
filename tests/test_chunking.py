from pathlib import Path

import pytest

from chunking import MAX_CHUNK_CHARS, ChunkDraft, chunk_markdown, chunk_text


def test_heading_path_follows_nesting():
    text = "# A\n\nintro\n\n## B\n\nbody b\n\n### C\n\nbody c\n\n## D\n\nbody d\n"
    chunks = chunk_markdown(text)
    assert [(c.heading_path, c.text) for c in chunks] == [
        ("A", "intro"),
        ("A > B", "body b"),
        ("A > B > C", "body c"),
        ("A > D", "body d"),
    ]


def test_hash_inside_code_block_is_not_a_heading():
    text = "# Setup\n\n```bash\n# install deps\npip install fastapi\n```\n"
    chunks = chunk_markdown(text)
    assert len(chunks) == 1
    assert chunks[0].heading_path == "Setup"
    assert "# install deps" in chunks[0].text


def test_cleans_front_matter_html_comments_and_anchors():
    text = (
        "---\ninclude_yaml:\n  sponsors: data/sponsors.yml\n---\n\n"
        "# FastAPI { #fastapi }\n\n<!-- sponsors -->\n"
        '<p align="center"><em>Fast to code</em></p>\n\n'
        "<style>\nh1 { display: none; }\n</style>\n"
    )
    assert chunk_markdown(text) == [ChunkDraft(heading_path="FastAPI", text="Fast to code")]


def test_long_section_is_split_on_paragraphs():
    paragraphs = [f"Paragraph {i} " + "x" * 400 for i in range(5)]
    chunks = chunk_markdown("# Long\n\n" + "\n\n".join(paragraphs))
    assert len(chunks) == 3
    assert all(len(c.text) <= MAX_CHUNK_CHARS for c in chunks)
    assert all(c.heading_path == "Long" for c in chunks)
    assert chunks[0].text.startswith("Paragraph 0")


def test_oversized_paragraph_falls_back_to_char_split():
    chunks = chunk_markdown("# Big\n\n" + "y" * 2500)
    assert len(chunks) == 3
    assert all(len(c.text) <= MAX_CHUNK_CHARS for c in chunks)


def test_code_block_is_never_split():
    code = "```python\n" + "\n\n".join(f"value_{i:03d} = {i}" for i in range(150)) + "\n```"
    chunks = chunk_markdown("# Code\n\n" + code + "\n")
    assert chunks == [ChunkDraft(heading_path="Code", text=code)]


def test_file_without_headings_has_empty_heading_path():
    chunks = chunk_markdown("just some text")
    assert chunks == [ChunkDraft(heading_path="", text="just some text")]
    assert chunks[0].embedding_input() == "just some text"


def test_embedding_input_prefixes_heading_path():
    assert ChunkDraft("A > B", "body").embedding_input() == "A > B\n\nbody"


def test_empty_and_markup_only_input_gives_no_chunks():
    assert chunk_markdown("") == []
    assert chunk_markdown("# Title\n\n---\n\n<img src='x.png'>\n") == []


def test_sample_docs_are_chunked_sensibly():
    chunks = chunk_markdown(Path("sample_docs/fastapi.md").read_text(encoding="utf-8"))
    assert any(c.heading_path == "FastAPI > Example > Run it" for c in chunks)
    assert not any("<img" in c.text or "<a href" in c.text for c in chunks)


def test_chunk_text_does_not_emit_tail_made_only_of_overlap():
    assert chunk_text("a" * 1000, 1000, 100) == ["a" * 1000]


def test_chunk_text_empty_input():
    assert chunk_text("", 1000, 100) == []


def test_chunk_text_rejects_overlap_not_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("abc", 10, 10)


def test_code_block_right_after_text_line_is_not_split():
    code = "```bash\n" + "\n".join(f"echo step {i:03d}" for i in range(100)) + "\n```"
    chunks = chunk_markdown("# Run\n\nRun it:\n" + code + "\n")
    assert [c.text for c in chunks] == ["Run it:", code]


def test_inline_code_and_autolinks_survive_html_stripping():
    text = "# Types\n\nUse `List<String>` here, see <https://example.com> or <b>bold</b>.\n"
    assert chunk_markdown(text)[0].text == "Use `List<String>` here, see <https://example.com> or bold."
