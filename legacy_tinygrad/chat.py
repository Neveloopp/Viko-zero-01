#!/usr/bin/env python3
from pathlib import Path
import argparse
import json

import numpy as np
from tinygrad import Tensor, Context, Device

from core.model import VikoGPT
from core.tokenizer import ByteTokenizer, EOS, USR, AST
from core.utils import load_config, load_weights, sample_token, parameter_count, human_number

def load_history(path):
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return list(data.get("turns", []))
    except Exception:
        return []

def save_history(path, turns):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"turns": turns}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def generate(model, tokenizer, prompt_ids, cfg, rng):
    g = cfg["generate"]
    context = int(cfg["model"]["context"])
    max_new = int(g.get("max_new_tokens", 180))
    temperature = float(g.get("temperature", 0.8))
    top_k = int(g.get("top_k", 40))
    top_p = float(g.get("top_p", 0.92))
    rep = float(g.get("repetition_penalty", 1.10))

    tokens = list(prompt_ids)
    generated = []

    with Context(TRAINING=0):
        for _ in range(max_new):
            window = tokens[-context:]
            x = Tensor(np.asarray([window], dtype=np.int32))
            logits = model(x).numpy()[0, -1]

            nxt = sample_token(
                logits, rng,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=rep,
                recent_tokens=tokens
            )

            if nxt in (EOS, USR):
                break

            # Evita que el modelo emita un marcador de rol extra.
            if nxt == AST and generated:
                continue

            tokens.append(nxt)
            generated.append(nxt)

    return tokenizer.decode_text(generated).strip()

def main():
    parser = argparse.ArgumentParser(description="Chat local con VikoZero.")
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    name = cfg.get("name", "viko")
    weights = Path("checkpoints") / f"{name}_latest.safetensors"

    if not weights.exists():
        raise SystemExit(
            f"No existe {weights}.\n"
            f"Primero entrena: ./viko train {name}"
        )

    print(f"[VikoZero] Device: {Device.DEFAULT}")
    print(f"[VikoZero] Cargando {weights}...")

    model = VikoGPT(cfg)
    load_weights(model, weights)

    print(f"[VikoZero] Parámetros: {parameter_count(model):,}")
    print("[VikoZero] Lista. /help para comandos.\n")

    tokenizer = ByteTokenizer()
    history_path = Path("memory") / f"{name}_history.json"
    history = load_history(history_path)
    history_turns = int(cfg["generate"].get("history_turns", 5))
    rng = np.random.default_rng(args.seed)

    while True:
        try:
            user = input("Tú > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user:
            continue

        if user == "/exit":
            break

        if user == "/help":
            print(
                "/clear           borra historial\n"
                "/temp 0.8        temperatura\n"
                "/topk 40         top-k\n"
                "/topp 0.92       top-p\n"
                "/exit            salir"
            )
            continue

        if user == "/clear":
            history = []
            save_history(history_path, history)
            print("[historial borrado]")
            continue

        if user.startswith("/temp "):
            try:
                cfg["generate"]["temperature"] = float(user.split(None, 1)[1])
                print(f"[temperature={cfg['generate']['temperature']}]")
            except Exception:
                print("[valor inválido]")
            continue

        if user.startswith("/topk "):
            try:
                cfg["generate"]["top_k"] = int(user.split(None, 1)[1])
                print(f"[top_k={cfg['generate']['top_k']}]")
            except Exception:
                print("[valor inválido]")
            continue

        if user.startswith("/topp "):
            try:
                cfg["generate"]["top_p"] = float(user.split(None, 1)[1])
                print(f"[top_p={cfg['generate']['top_p']}]")
            except Exception:
                print("[valor inválido]")
            continue

        recent = history[-history_turns:]
        prompt_ids = tokenizer.encode_chat_prompt(
            recent,
            user,
            max_tokens=max(8, int(cfg["model"]["context"]) - 8)
        )

        answer = generate(model, tokenizer, prompt_ids, cfg, rng)
        if not answer:
            answer = "..."

        print(f"Viko > {answer}\n")

        history.append({"user": user, "assistant": answer})
        history = history[-100:]
        save_history(history_path, history)

if __name__ == "__main__":
    main()
