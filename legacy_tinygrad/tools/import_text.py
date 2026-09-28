#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Convierte texto libre a registros JSONL.")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--chunk-chars", type=int, default=1800)
    args = p.parse_args()

    text = Path(args.input).read_text(encoding="utf-8", errors="replace")
    text = text.replace("\r\n", "\n").strip()

    chunks = []
    step = max(200, int(args.chunk_chars))
    pos = 0
    while pos < len(text):
        end = min(len(text), pos + step)
        if end < len(text):
            split = text.rfind("\n", pos, end)
            if split > pos + step // 2:
                end = split
        chunk = text[pos:end].strip()
        if chunk:
            chunks.append(chunk)
        pos = max(end, pos + 1)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")

    print(f"Creado {out} con {len(chunks)} bloques.")

if __name__ == "__main__":
    main()
