from pathlib import Path
import json
import math
import numpy as np

from tinygrad.nn.state import get_state_dict, safe_save, safe_load, load_state_dict

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parameter_count(model):
    return sum(int(v.numel()) for v in get_state_dict(model).values())

def human_number(n):
    n = int(n)
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.2f}K"
    return str(n)

def save_weights(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_save(get_state_dict(model), str(path))

def load_weights(model, path):
    state = safe_load(str(path))
    load_state_dict(model, state, strict=True, verbose=False)

def softmax_np(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits)
    e = np.exp(logits)
    s = e.sum()
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(e) / len(e)
    return e / s

def sample_token(logits, rng, temperature=0.8, top_k=40, top_p=0.92,
                 repetition_penalty=1.10, recent_tokens=None):
    logits = np.asarray(logits, dtype=np.float64).copy()

    if recent_tokens and repetition_penalty > 1.0:
        for token in set(recent_tokens[-96:]):
            if 0 <= token < len(logits):
                if logits[token] >= 0:
                    logits[token] /= repetition_penalty
                else:
                    logits[token] *= repetition_penalty

    temperature = max(0.05, float(temperature))
    logits /= temperature

    if top_k and 0 < int(top_k) < len(logits):
        k = int(top_k)
        threshold = np.partition(logits, -k)[-k]
        logits[logits < threshold] = -np.inf

    probs = softmax_np(logits)

    if top_p and 0 < float(top_p) < 1:
        order = np.argsort(probs)[::-1]
        sorted_probs = probs[order]
        cumulative = np.cumsum(sorted_probs)
        cut = cumulative > float(top_p)
        if len(cut):
            cut[0] = False
        sorted_probs[cut] = 0
        total = sorted_probs.sum()
        if total > 0:
            sorted_probs /= total
            probs = np.zeros_like(probs)
            probs[order] = sorted_probs

    return int(rng.choice(len(probs), p=probs))
