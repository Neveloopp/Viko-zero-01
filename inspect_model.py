#!/usr/bin/env python3
import argparse
from tinygrad import Device
from core.model import VikoGPT
from core.utils import load_config, parameter_count, human_number

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.json")
    args = p.parse_args()

    cfg = load_config(args.config)
    model = VikoGPT(cfg)
    n = parameter_count(model)
    m = cfg["model"]

    print("VikoZero")
    print(f"Preset:      {cfg.get('name', '-')}")
    print(f"Device:      {Device.DEFAULT}")
    print(f"Parámetros:  {n:,} ({human_number(n)})")
    print(f"Capas:       {m['layers']}")
    print(f"Dimensión:   {m['dim']}")
    print(f"Cabezas:     {m['heads']}")
    print(f"FF:          {m['ff_dim']}")
    print(f"Contexto:    {m['context']}")
    print(f"Vocabulario: {m['vocab_size']} (byte-level UTF-8 + especiales)")

if __name__ == "__main__":
    main()
