import re
from dataclasses import dataclass

MAX_CHUNK_CHARS = 1000
FALLBACK_OVERLAP = 100

FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
STYLE_OR_SCRIPT = re.compile(r"<(style|script)\b.*?</\1>", re.DOTALL | re.IGNORECASE)
# Nazwa tagu, potem spacja z atrybutami albo od razu ">" — <https://...> to nie tag.
HTML_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(\s[^>]*)?/?>")
INLINE_CODE = re.compile(r"(`[^`]*`)")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
HEADING_ANCHOR = re.compile(r"\{\s*#[^}]*\}")
FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass
class ChunkDraft:
    heading_path: str
    text: str

    def embedding_input(self) -> str:
        if not self.heading_path:
            return self.text
        return f"{self.heading_path}\n\n{self.text}"


def chunk_markdown(text: str) -> list[ChunkDraft]:
    drafts = []
    for heading_path, lines in _split_sections(_strip_markup(text)):
        for chunk in _pack(_paragraphs(lines)):
            drafts.append(ChunkDraft(heading_path=heading_path, text=chunk))
    return drafts


def chunk_text(text, chunk_size, overlap):
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        if start + chunk_size >= len(text):
            break
        start = start + (chunk_size - overlap)
    return chunks


def _strip_markup(text):
    text = FRONT_MATTER.sub("", text)
    text = HTML_COMMENT.sub("", text)
    return STYLE_OR_SCRIPT.sub("", text)


def _clean_heading(title):
    title = HEADING_ANCHOR.sub("", title)
    title = _strip_tags(title)
    return title.replace("*", "").replace("`", "").strip()


def _strip_tags(line):
    """Usuwa tagi HTML, ale nie rusza kodu w `backtickach`."""
    parts = INLINE_CODE.split(line)
    return "".join(p if i % 2 else HTML_TAG.sub("", p) for i, p in enumerate(parts))


def _path(stack):
    return " > ".join(title for _, title in stack)


def _split_sections(text):
    """Lista (heading_path, linie treści) — jedna pozycja na sekcję."""
    sections = []
    stack = []  # (poziom nagłówka, tytuł)
    lines = []
    in_fence = False
    for line in text.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence:
            heading = HEADING.match(line)
            if heading:
                sections.append((_path(stack), lines))
                level = len(heading.group(1))
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, _clean_heading(heading.group(2))))
                lines = []
                continue
            line = _strip_tags(line)
        lines.append(line)
    sections.append((_path(stack), lines))
    return sections


def _paragraphs(lines):
    """Akapity rozdzielone pustą linią; blok kodu jest zawsze jednym akapitem."""
    paragraphs = []
    current = []
    in_fence = False
    for line in lines:
        if FENCE.match(line):
            if in_fence:
                current.append(line)
            # Blok kodu zaczyna i kończy akapit, nawet bez pustej linii obok.
            paragraphs.append("\n".join(current))
            current = [] if in_fence else [line]
            in_fence = not in_fence
        elif not line.strip() and not in_fence:
            paragraphs.append("\n".join(current))
            current = []
        else:
            current.append(line)
    paragraphs.append("\n".join(current))
    return [p.strip() for p in paragraphs if re.search(r"\w", p)]


def _pack(paragraphs):
    """Skleja akapity w fragmenty nie dłuższe niż MAX_CHUNK_CHARS."""
    chunks = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > MAX_CHUNK_CHARS:
            if current:
                chunks.append(current)
                current = ""
            if FENCE.match(paragraph):
                chunks.append(paragraph)
            else:
                chunks.extend(chunk_text(paragraph, MAX_CHUNK_CHARS, FALLBACK_OVERLAP))
        elif not current:
            current = paragraph
        elif len(current) + 2 + len(paragraph) <= MAX_CHUNK_CHARS:
            current += "\n\n" + paragraph
        else:
            chunks.append(current)
            current = paragraph
    if current:
        chunks.append(current)
    return chunks
