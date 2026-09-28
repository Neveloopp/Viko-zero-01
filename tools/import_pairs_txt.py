#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser(
        description="Convierte un TXT de líneas alternadas pregunta/respuesta a JSONL."
    )
    p.add_argument("input")
    p.add_argument("output")
    args = p.parse_args()

    lines = [
        line.strip()
        for line in Path(args.input).read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]

    if len(lines) < 2:
        raise SystemExit("El TXT no contiene suficientes líneas.")

    if len(lines) % 2:
        print("Aviso: hay una línea impar; la última se ignorará.")
        lines = lines[:-1]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for i in range(0, len(lines), 2):
            f.write(json.dumps(
                {"prompt": lines[i], "response": lines[i+1]},
                ensure_ascii=False
            ) + "\n")

    print(f"Creado {out} con {len(lines)//2} conversaciones.")

if __name__ == "__main__":
    main()
