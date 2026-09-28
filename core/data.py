from pathlib import Path
import hashlib
import json
import numpy as np

from .tokenizer import ByteTokenizer

def _signature(paths):
    h = hashlib.sha256()
    for p in paths:
        path = Path(p)
        h.update(str(path.resolve()).encode())
        h.update(str(path.stat().st_size).encode())
        h.update(str(path.stat().st_mtime_ns).encode())
    return h.hexdigest()[:16]

def build_token_stream(paths, cache_dir="cache"):
    paths = [Path(p) for p in paths]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    sig = _signature(paths)
    cache_path = cache_dir / f"tokens_{sig}.npy"

    if cache_path.exists():
        return np.load(cache_path, mmap_mode="r"), cache_path

    tok = ByteTokenizer()
    tokens = []

    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSON inválido en {path}:{line_no}: {exc}") from exc

                encoded = tok.encode_record(rec)
                if encoded:
                    tokens.extend(encoded)

    if len(tokens) < 100:
        raise ValueError("Dataset demasiado pequeño.")

    arr = np.asarray(tokens, dtype=np.int32)
    np.save(cache_path, arr)
    return np.load(cache_path, mmap_mode="r"), cache_path

def sample_batch(stream, batch_size, context, rng):
    if len(stream) <= context + 1:
        raise ValueError(
            f"Se necesitan más de {context + 1} tokens; dataset tiene {len(stream)}."
        )

    starts = rng.integers(0, len(stream) - context - 1, size=batch_size)
    x = np.stack([stream[s:s+context] for s in starts]).astype(np.int32)
    y = np.stack([stream[s+1:s+context+1] for s in starts]).astype(np.int32)
    return x, y
