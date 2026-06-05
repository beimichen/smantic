"""Gradio 'try it live' demo for smantic, deployable as a Hugging Face Space.

Paste Markdown; smantic chunks it (structure-aware blocks + semantic prose
boundaries via ONNX all-MiniLM-L6-v2) and returns a per-chunk summary plus the
structured JSON.

The embedding model (~90 MB) downloads from the Hub on the first chunk, so the
very first request takes a few seconds on a CPU Space; the chunker is then kept
warm and later requests are fast. Tunables via env vars:

    SMANTIC_MAX_TOKENS=500  SMANTIC_OVERLAP_TOKENS=50  SMANTIC_DEMO_MAX_CHARS=20000

Run locally with:  pip install "smantic[onnx]" gradio && python demo/app.py
"""

import json
import os
import threading

import gradio as gr

from smantic import config, make_chunker
from smantic.adapters import from_markdown

MAX_CHARS = int(os.getenv("SMANTIC_DEMO_MAX_CHARS", "20000"))

_chunker = None
_chunker_lock = threading.Lock()

_EXAMPLE = """# Introduction

Retrieval-augmented generation depends on good chunks. A chunk that mixes two
unrelated ideas pollutes the embedding; a chunk split mid-thought loses context.
smantic tries to cut at the seams the document already has.

## How it works

It keeps code, tables, and formulas intact, and finds semantic boundaries inside
prose runs using sentence embeddings.

```python
def chunk(doc):
    return StructureAwareChunker().chunk_document(doc)
```

| Boundary | Trigger              |
|----------|----------------------|
| hard     | headings, block edge |
| soft     | meaning shifts       |

## References

Doe, J. 2021. Something that gets skipped as backmatter.
"""


def _get_chunker():
    """Build the chunker once and keep it warm across requests.

    The caller (run_chunk) already holds _chunker_lock, so this must NOT
    re-acquire it: threading.Lock is not reentrant and would self-deadlock.
    """
    global _chunker
    if _chunker is None:
        _chunker = make_chunker()
    return _chunker


def run_chunk(text: str, max_tokens: int, overlap: int):
    """Chunk Markdown text and return (summary markdown, pretty JSON)."""
    if not text or not text.strip():
        return "Paste some Markdown to begin.", "[]"
    if len(text) > MAX_CHARS:
        return f"**Input too long** (> {MAX_CHARS} chars for the demo).", "[]"

    with _chunker_lock:
        chunker = _get_chunker()
        chunker.max_tokens = int(max_tokens)
        chunker.overlap_tokens = int(overlap)
        try:
            chunks = chunker.chunk_document(from_markdown(text))
        except Exception as exc:  # surface errors in the UI instead of 500ing
            return f"**Chunking failed:** {exc}", "[]"

    if not chunks:
        return "_(no chunks produced)_", "[]"

    lines = [f"**{len(chunks)} chunks**", ""]
    for c in chunks:
        heading = (c.metadata or {}).get("nearest_heading", "")
        preview = " ".join(c.content.split())[:90]
        lines.append(
            f"- `[{c.sequence}]` **{c.dominant_type}** · {c.token_count} tokens"
            + (f" · _{heading}_" if heading else "")
            + f"\n    > {preview}"
        )
    summary = "\n".join(lines)
    data = json.dumps([c.to_dict() for c in chunks], indent=2, ensure_ascii=False)
    return summary, data


with gr.Blocks(title="smantic") as demo:
    gr.Markdown(
        "# smantic\n"
        "**Structure-aware semantic chunking, minus the heavyweight stack.** "
        "Paste Markdown and get retrieval-ready chunks back "
        "([source](https://github.com/beimichen/smantic)).\n\n"
        "_The first chunk downloads the embedding model (~90 MB), so give it a "
        "few seconds; later ones are fast._"
    )
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                label="Markdown", value=_EXAMPLE, lines=18, max_lines=30,
            )
            with gr.Row():
                max_tokens = gr.Slider(
                    100, 1000, value=config.DEFAULT_MAX_TOKENS, step=10, label="max tokens",
                )
                overlap = gr.Slider(
                    0, 200, value=config.DEFAULT_OVERLAP_TOKENS, step=10, label="overlap tokens",
                )
            run = gr.Button("Chunk", variant="primary")
        with gr.Column():
            with gr.Tab("Chunks"):
                summary_out = gr.Markdown()
            with gr.Tab("JSON"):
                json_out = gr.Code(language="json")
    run.click(run_chunk, inputs=[text, max_tokens, overlap], outputs=[summary_out, json_out])


if __name__ == "__main__":
    demo.launch()
