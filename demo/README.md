---
title: smantic
emoji: 🧩
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
suggested_hardware: cpu-basic
short_description: Structure-aware semantic chunking for RAG.
---

# smantic, live demo

A Gradio front end for [smantic](https://github.com/beimichen/smantic): paste
Markdown and get retrieval-ready chunks back (atomic code/table/formula blocks
kept intact, prose split on semantic boundaries, headings tracked, tiny chunks
merged).

## Deploy this as a Hugging Face Space

This folder *is* the Space. The files (`app.py`, `requirements.txt`, and this
`README.md`) belong at the **root** of the Space repo, not in a `demo/` subdir.

1. Create a Space: https://huggingface.co/new-space (SDK: **Gradio**, hardware:
   **CPU basic** is enough).
2. Copy this folder's three files to the Space repo root and push:
   ```bash
   cp demo/app.py demo/requirements.txt demo/README.md /path/to/space-repo/
   cd /path/to/space-repo && git add . && git commit -m "smantic demo" && git push
   ```
3. The Space builds, then runs `app.py`. The first chunk downloads the ~90 MB
   all-MiniLM-L6-v2 ONNX model from the Hub (a few seconds on CPU); the chunker
   is kept warm after that.

## Notes

- **Backend.** Uses the local `onnx` embedder (free, no key). With no model the
  chunker still runs on structural boundaries alone, but the demo installs the
  `[onnx]` extra so prose gets genuine semantic boundaries.
- **Input cap.** The demo chunks at most `SMANTIC_DEMO_MAX_CHARS` characters
  (default 20000) to keep CPU runs short. Raise it via a Space variable.
- **Local run.** `pip install "smantic[onnx]" gradio && python app.py`.
