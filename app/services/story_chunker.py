import re

MIN_WORDS = 25
MAX_WORDS = 50
# The first chunk is kept short so the first audio is ready sooner; later chunks are
# generated while earlier ones play.
FIRST_MAX_WORDS = 20


def _word_count(text: str) -> int:
    return len(text.split())


def chunk_story(text: str) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    sentences = re.findall(
        r"[^.!?।]+[.!?।]+|[^.!?।]+$",
        cleaned,
    )

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_words = _word_count(sentence)

        if not current:
            current = [sentence]
            current_words = sentence_words
            continue

        candidate_words = current_words + sentence_words

        max_words = FIRST_MAX_WORDS if not chunks else MAX_WORDS
        if candidate_words <= max_words:
            current.append(sentence)
            current_words = candidate_words
            continue

        # If current chunk is too short, merge the next sentence instead of creating a tiny patch
        if current_words < MIN_WORDS:
            current.append(sentence)
            current_words = candidate_words
            continue

        chunks.append(" ".join(current))
        current = [sentence]
        current_words = sentence_words

    if current:
        chunks.append(" ".join(current))

    # Merge tiny last chunk into previous one
    if len(chunks) > 1 and _word_count(chunks[-1]) < MIN_WORDS:
        chunks[-2] = f"{chunks[-2]} {chunks[-1]}".strip()
        chunks.pop()

    return chunks


def chunk_story_map(text: str) -> dict[str, str]:
    return {
        f"chunk{index}": chunk
        for index, chunk in enumerate(chunk_story(text), start=1)
    }
