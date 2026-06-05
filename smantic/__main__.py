"""CLI entry point: ``python -m smantic notes.md [--format jsonl]``."""

import argparse
import json
import sys
from pathlib import Path


def _read_input(path: Path) -> str:
    if str(path) == "-":
        return sys.stdin.read()
    return path.read_text(encoding="utf-8")


def _build_document(text: str, input_format: str, source: str):
    """Turn raw input text into a smantic Document per ``input_format``."""
    from smantic import from_json, from_markdown  # lazy: keep --help fast

    fmt = input_format
    if fmt == "auto":
        suffix = Path(source).suffix.lower()
        fmt = "json" if suffix == ".json" else "markdown"

    if fmt == "markdown":
        return from_markdown(text)
    # 'json' / 'nopaddle' share one permissive loader (regions/elements both ok).
    return from_json(text)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="smantic",
        description="Chunk a document (Markdown or parsed JSON) into semantic chunks.",
    )
    p.add_argument("input", type=Path, help="path to the document, or - for stdin")
    p.add_argument(
        "--input-format", default="auto",
        choices=["auto", "markdown", "json", "nopaddle"],
        help="how to read the input (default: auto by file extension)",
    )
    p.add_argument(
        "--format", default="jsonl", choices=["jsonl", "json", "summary"],
        help="output format (default: jsonl, one chunk per line)",
    )
    p.add_argument("--max-tokens", type=int, default=None,
                   help="soft max tokens per chunk (default: $SMANTIC_MAX_TOKENS or 500)")
    p.add_argument("--overlap", type=int, default=None,
                   help="token overlap between prose chunks (default: $SMANTIC_OVERLAP_TOKENS or 50)")
    p.add_argument("--model-dir", type=Path, default=None,
                   help="local all-MiniLM-L6-v2 ONNX dir (default: download + cache)")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="output file (default: stdout)")
    args = p.parse_args(argv)

    from smantic import chunk_document, config

    text = _read_input(args.input)
    doc = _build_document(text, args.input_format, str(args.input))

    chunks = chunk_document(
        doc,
        max_tokens=args.max_tokens if args.max_tokens is not None else config.DEFAULT_MAX_TOKENS,
        overlap_tokens=args.overlap if args.overlap is not None else config.DEFAULT_OVERLAP_TOKENS,
        model_dir=args.model_dir,
    )

    if args.format == "summary":
        lines = [f"{len(chunks)} chunks"]
        for c in chunks:
            heading = c.metadata.get("nearest_heading", "")
            preview = " ".join(c.content.split())[:70]
            lines.append(
                f"  [{c.sequence:>3}] {c.dominant_type:<12} {c.token_count:>4}t"
                f"  {heading[:30]:<30}  {preview}"
            )
        out = "\n".join(lines)
    elif args.format == "json":
        out = json.dumps([c.to_dict() for c in chunks], indent=2, ensure_ascii=False)
    else:  # jsonl
        out = "\n".join(json.dumps(c.to_dict(), ensure_ascii=False) for c in chunks)

    if args.output:
        args.output.write_text(out + "\n", encoding="utf-8")
    else:
        sys.stdout.write(out + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
