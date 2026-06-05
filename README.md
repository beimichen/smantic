# smantic

**Structure-aware semantic chunking, minus the heavyweight stack.**

[![PyPI](https://img.shields.io/pypi/v/smantic)](https://pypi.org/project/smantic/)
[![CI](https://github.com/beimichen/smantic/actions/workflows/ci.yml/badge.svg)](https://github.com/beimichen/smantic/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/Bei0001/smantic)

Good retrieval starts with good chunks. A chunk that staples two unrelated ideas
together poisons its embedding. A chunk that splits a thought in half loses the
context that made it useful. Most splitters cut every N characters and hope for
the best. smantic cuts at the seams the document already has.

It keeps code, tables, and formulas intact (splitting only the big ones, by AST
or row group or equation), finds real topic boundaries inside prose using
sentence embeddings, tracks the heading trail, and merges the runts. It does all
of that with no torch, no transformers, and no paddle. The semantic model is
all-MiniLM-L6-v2 running on plain `onnxruntime` plus the Rust `tokenizers`
library, so the whole thing installs small and runs on any CPU.

(Yes, the `e` fell out of "semantic". That is the joke. It is also a hint: this
is about **s**tructure plus se**mantic**s.)

It is the chunking half of a parse plus chunk pipeline. Its sibling
[NoPaddle](https://github.com/beimichen/nopaddle) turns a PDF into typed regions;
smantic turns those regions (or any Markdown) into chunks. They snap together.

## Install

```bash
pip install "smantic[onnx]"     # with semantic boundary detection (recommended)
pip install smantic             # core only: structural chunking, no model
```

The `[onnx]` extra pulls `onnxruntime`, `tokenizers`, and `huggingface_hub`. The
embedding model (~90 MB) is downloaded from the Hugging Face Hub the first time
you chunk, then cached on disk. Without the extra, smantic still chunks on
structural boundaries alone (headings, block edges, size limits); it just skips
the semantic ones.

## Quick start

Feed it a Markdown string (or file) and smantic works out the structure for you:

```python
import smantic

md = r"""
# Mixed Precision Training

Training in FP16 halves memory but underflows small gradients; loss scaling
multiplies the loss before backprop so they stay representable.

| Format | Bits | Exponent | Mantissa |
|--------|------|----------|----------|
| FP32   | 32   | 8        | 23       |
| FP16   | 16   | 5        | 10       |

The scaled loss keeps the same direction:

$$ \mathcal{L}_{scaled} = s \cdot \mathcal{L} $$
"""

chunks = smantic.chunk_markdown(md)

for c in chunks:
    print(c.sequence, c.dominant_type, f"{c.token_count}t", "|", c.metadata.get("nearest_heading"))
```

```text
0 prose 73t | Mixed Precision Training
1 table_block 39t | Mixed Precision Training
2 formula_block 28t | Mixed Precision Training
```

The heading, prose, table, and formula each became their own chunk: prose is cut
on semantic boundaries, while the table and formula stay intact as atomic blocks,
and every chunk carries the heading trail. Each item is a `Chunk`;
`chunk.to_dict()` gives a JSON-ready record (here the table chunk):

```python
>>> chunks[1].to_dict()
{
  "content": "| Format | Bits | Exponent | Mantissa |\n|--------|---...",
  "token_count": 39,
  "page_numbers": [1],
  "span_start": 0, "span_end": 159,
  "chunking_method": "atomic_block",
  "dominant_type": "table_block",
  "parent_chunk_id": None, "block_sequence": None,
  "has_code": False, "has_math": False, "has_table": True,
  "metadata": {
    "element_type": "table",
    "heading_trail": ["Mixed Precision Training"],
    "heading_level": 1,
    "nearest_heading": "Mixed Precision Training",
  },
  "sequence": 1,
}
```

Reading from a file is the same: `smantic.chunk_markdown(open("notes.md").read())`.

Already have a parsed document? Feed it straight in:

```python
import smantic

doc = smantic.from_docling_json(open("parsed.json").read())
chunks = smantic.chunk_document(doc, max_tokens=500, overlap_tokens=50)
```

### Pairs with NoPaddle

```python
import nopaddle, smantic

doc = nopaddle.parse_pdf("paper.pdf")          # PDF  -> typed regions
chunks = smantic.chunk_document(smantic.from_nopaddle(doc))   # regions -> chunks
```

`from_nopaddle` reads a NoPaddle `ParsedDocument` (object, dict, or JSON) with no
conversion step: the two projects share the same region shape, and smantic's IR
accepts NoPaddle's `regions` key directly.

A table NoPaddle detects survives intact: it lands in one `table_block` chunk
(or, if it is huge, splits by row group with the header repeated), never shredded
into prose:

```python
chunks = smantic.chunk_document(smantic.from_nopaddle(doc))

tables = [c for c in chunks if c.dominant_type == "table_block"]
print(tables[0].has_table)   # True
print(tables[0].content)     # the GFM table, kept whole
```

## Command line

```bash
smantic notes.md                       # JSONL, one chunk per line
smantic notes.md --format summary      # human-readable table
smantic parsed.json --input-format json --format json -o chunks.json
cat notes.md | smantic - --max-tokens 400 --overlap 40
```

`--input-format` defaults to `auto` (by file extension: `.json` is parsed JSON,
everything else is Markdown). `--format` is `jsonl` (default), `json`, or
`summary`.

## How it works

smantic walks the document once and classifies every element:

1. **Atomic blocks** (`code`, `table`, `formula`, `picture`, ...) stay intact.
   A block over the size limit is split into a parent plus children, by Python
   or JavaScript AST for code, by row group (headers repeated) for tables, and
   by environment or step for formula derivations. Visual blocks become a chunk
   only when they carry a caption or alt text worth retrieving.

2. **Prose** runs through three-tier boundary detection:
   - **hard** boundaries always cut: section headings, and transitions between
     incompatible element types.
   - **soft** boundaries cut when the chunk is already big enough: a drop in
     sentence-to-sentence cosine similarity (the semantic part), or a new
     paragraph.
   - **emergency** boundaries cut anywhere once a chunk would blow past
     `max_tokens`.

3. **Headings** are accumulated into a trail (`heading_trail`,
   `nearest_heading`, `heading_level`) and folded into the first chunk under
   them, so the heading is searchable without wasting a chunk on it.

4. **Cleanups**: consecutive prose chunks share a configurable token overlap for
   context continuity; chunks below a useful minimum get merged into a same-type
   neighbour; parser-artifact duplicates (a sentence repeated by a multi-column
   OCR pass, say) get collapsed; and low-value backmatter (References,
   Acknowledgments, Funding, and friends) is skipped.

When the embedding model is not installed, the soft semantic boundary is simply
skipped. Everything else still works.

## Input formats

| Source | Helper | Notes |
|--------|--------|-------|
| Markdown text | `from_markdown(text)` | headings, code fences, pipe tables, `$$`/`\[` math, lists, images, prose |
| NoPaddle output | `from_nopaddle(doc)` | object, dict, or JSON; reads the `regions` key natively |
| Docling-style JSON | `from_docling_json(text)` | `{"pages": [{"elements": [...]}]}` |
| The IR directly | `from_dict(d)` / `from_json(s)` | smantic's own shape |

All of them build a `smantic.Document`, which is what the chunker consumes.

## Output

`chunk_document` returns a list of `Chunk` objects. `chunk.to_dict()` gives a
JSON-ready dict:

```python
{
  "content": "...",              # the chunk text
  "token_count": 312,
  "page_numbers": [4],
  "span_start": 0, "span_end": 1840,
  "chunking_method": "semantic", # semantic | atomic_block | ast_split | row_group | ...
  "dominant_type": "prose",      # prose | code_block | table_block | formula_block | visual_block
  "has_code": false, "has_math": false, "has_table": false,
  "parent_chunk_id": null,       # set by you if you persist the parent->child links
  "block_sequence": null,        # order of a child within a split block
  "metadata": {                  # heading_trail, nearest_heading, section_type, timecodes, ...
    "heading_trail": ["Methods", "Training"],
    "nearest_heading": "Training"
  },
  "sequence": 7                  # position in the returned list
}
```

## Self-host (FastAPI + Docker)

```bash
pip install "smantic[onnx,serve]"
smantic-serve                                  # serves on 0.0.0.0:8000
```

```bash
curl -s localhost:8000/chunk \
  -H 'content-type: application/json' \
  -d '{"text": "# Title\n\nSome prose to chunk."}' | jq
```

`POST /chunk` takes either `text` (Markdown) or `document` (a parsed dict), plus
optional `max_tokens` / `overlap_tokens`. There is a `GET /healthz` and a
`GET /info`. One chunker is kept warm across requests; set
`SMANTIC_RELEASE_AFTER_REQUEST=1` to free the model after each call instead.

Or run the image:

```bash
docker build -f docker/Dockerfile -t smantic .
docker run -p 8000:8000 -v smantic-models:/models smantic
```

## Configuration

Every knob has an env var (`SMANTIC_*`) and a code path:

| Env | Default | Meaning |
|-----|---------|---------|
| `SMANTIC_MAX_TOKENS` | 500 | soft max tokens per chunk |
| `SMANTIC_OVERLAP_TOKENS` | 50 | token overlap between prose chunks |
| `SMANTIC_BOUNDARY_THRESHOLD` | 0.5 | cosine threshold for a soft semantic cut |
| `SMANTIC_EMBED_REPO` | `sentence-transformers/all-MiniLM-L6-v2` | embedding model repo |
| `MODEL_CACHE_DIR` | `~/.cache/smantic/models` | where the model is cached |

## Why it is light

No torch. No transformers. No paddle. The core is `numpy` plus stdlib; the
`[onnx]` extra adds `onnxruntime` and the Rust `tokenizers` library and nothing
else. The embedding model is the 384-dim all-MiniLM-L6-v2 ONNX graph, run with a
host-side mean-pool, so there is no deep-learning framework in the dependency
tree at all.

## Status

Alpha. The chunker is ported from a production ingestion pipeline and is well
covered by tests (the core suite runs offline against a graceful fallback, so it
is fast and needs no model download; the real-embedder path is exercised by the
`slow`-marked tests). The Markdown parser is a pragmatic block segmenter, not a
full CommonMark implementation: it handles the constructs that matter for
chunking and leaves inline formatting untouched.

## License

Apache-2.0.
