#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import signal
import sys
import time

import numpy as np
from tinygrad import Tensor, Context, nn, Device

from core.model import VikoGPT
from core.data import build_token_stream, sample_batch
from core.utils import load_config, parameter_count, human_number, save_weights, load_weights

def main():
    parser = argparse.ArgumentParser(description="Entrena VikoZero desde pesos aleatorios.")
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--data", nargs="+", default=["data/viko_seed.jsonl"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["train"]
    name = cfg.get("name", "viko")

    steps = int(args.steps or tcfg["steps"])
    batch_size = int(args.batch_size or tcfg["batch_size"])
    lr = float(args.lr or tcfg["learning_rate"])
    weight_decay = float(tcfg.get("weight_decay", 0.0))
    log_every = int(tcfg.get("log_every", 10))
    save_every = int(tcfg.get("save_every", 250))
    seed = int(tcfg.get("seed", 1337))
    context = int(cfg["model"]["context"])

    Tensor.manual_seed(seed)
    rng = np.random.default_rng(seed)

    print(f"[VikoZero] Device: {Device.DEFAULT}")
    print("[VikoZero] Construyendo/cargando stream de entrenamiento...")
    stream, cache_path = build_token_stream(args.data)
    print(f"[VikoZero] Tokens: {len(stream):,}")
    print(f"[VikoZero] Cache: {cache_path}")

    model = VikoGPT(cfg)
    params = parameter_count(model)
    print(f"[VikoZero] Parámetros: {params:,} ({human_number(params)})")
    print(f"[VikoZero] Contexto: {context} bytes/tokens")
    print(f"[VikoZero] Batch: {batch_size}")
    print(f"[VikoZero] LR: {lr}")

    latest = Path("checkpoints") / f"{name}_latest.safetensors"
    state_path = Path("checkpoints") / f"{name}_state.json"
    start_step = 0

    if args.resume and latest.exists():
        print(f"[VikoZero] Cargando {latest}...")
        load_weights(model, latest)
        if state_path.exists():
            try:
                start_step = int(json.loads(state_path.read_text())["step"])
            except Exception:
                start_step = 0
        print(f"[VikoZero] Continuando desde paso {start_step}.")
    elif args.resume:
        print("[VikoZero] No existe checkpoint: comenzando desde cero.")
    else:
        print("[VikoZero] Pesos aleatorios nuevos: entrenamiento DESDE CERO.")

    optim = nn.optim.AdamW(
        nn.state.get_parameters(model),
        lr=lr,
        weight_decay=weight_decay
    )

    stop = {"value": False}

    def request_stop(sig, frame):
        stop["value"] = True
        print("\n[VikoZero] Detención solicitada. Guardaré al terminar este paso...")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    last_time = time.time()
    running_loss = 0.0
    running_count = 0
    completed_step = start_step

    try:
        with Context(TRAINING=1):
            for step in range(start_step + 1, steps + 1):
                x_np, y_np = sample_batch(stream, batch_size, context, rng)
                x = Tensor(x_np)
                y = Tensor(y_np)

                optim.zero_grad()
                logits = model(x)
                loss = logits.sparse_categorical_crossentropy(y)
                loss.backward()
                optim.step()

                loss_value = float(loss.item())
                running_loss += loss_value
                running_count += 1
                completed_step = step

                if step % log_every == 0 or step == start_step + 1:
                    now = time.time()
                    elapsed = max(now - last_time, 1e-6)
                    avg = running_loss / max(running_count, 1)
                    tok_s = (running_count * batch_size * context) / elapsed
                    ppl = min(1e9, float(np.exp(min(avg, 20))))
                    print(
                        f"paso {step:>7}/{steps} | "
                        f"loss {avg:.4f} | ppl {ppl:.2f} | "
                        f"{tok_s:,.0f} tok/s"
                    )
                    running_loss = 0.0
                    running_count = 0
                    last_time = now

                if step % save_every == 0:
                    save_weights(model, latest)
                    state_path.write_text(
                        json.dumps({"step": step, "config": args.config}, indent=2),
                        encoding="utf-8"
                    )
                    snap = Path("checkpoints") / f"{name}_step_{step}.safetensors"
                    save_weights(model, snap)
                    print(f"[VikoZero] Checkpoint guardado: {latest}")

                if stop["value"]:
                    break

    finally:
        if completed_step > start_step:
            save_weights(model, latest)
            state_path.write_text(
                json.dumps({"step": completed_step, "config": args.config}, indent=2),
                encoding="utf-8"
            )
            print(f"[VikoZero] Guardado final: {latest}")
            print(f"[VikoZero] Paso alcanzado: {completed_step}")

if __name__ == "__main__":
    main()
