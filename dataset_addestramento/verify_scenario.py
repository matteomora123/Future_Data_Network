#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_scenario.py — Controlla consistenza di scenario JSON (posizioni, range, U2U, distanze).
Legge il percorso del dataset direttamente da config_env.json.
"""

import json
import numpy as np
from pathlib import Path


def verify(path: Path):
    scn = json.loads(path.read_text(encoding="utf-8"))
    slots = scn["slots"]
    T = len(slots)
    print(f"\nScenario: {scn['meta'].get('name','?')} | slot totali: {T}")

    # --- range coordinate ---
    all_u = np.array([[u["x"], u["y"], u["z"]] for s in slots for u in s["uav"]])
    all_g = np.array([[g["x"], g["y"], g["z"]] for g in slots[0]["gs"]])
    print("\n[1] RANGE COORDINATE (m)")
    print(f"UAV: x∈[{all_u[:,0].min():.2f},{all_u[:,0].max():.2f}], "
          f"y∈[{all_u[:,1].min():.2f},{all_u[:,1].max():.2f}], "
          f"z∈[{all_u[:,2].min():.2f},{all_u[:,2].max():.2f}]")
    print(f"GS : x∈[{all_g[:,0].min():.2f},{all_g[:,0].max():.2f}], "
          f"y∈[{all_g[:,1].min():.2f},{all_g[:,1].max():.2f}], "
          f"z∈[{all_g[:,2].min():.2f},{all_g[:,2].max():.2f}]")

    # --- distanze medie U2G slot 0 ---
    s0 = slots[0]
    D = []
    for lk in s0["links"]:
        u = next(u for u in s0["uav"] if u["id"] == lk["uav_id"])
        g = next(g for g in s0["gs"] if g["id"] == lk["gs_id"])
        d_real = np.linalg.norm([u["x"] - g["x"], u["y"] - g["y"], u["z"] - g["z"]])
        D.append(abs(d_real - lk["w1"]))
    print("\n[2] VERIFICA DISTANZE U2G (slot 0)")
    print(f"Errore medio distanza (m): {np.mean(D):.3f}")

    # --- U2U links ---
    u2u = s0.get("u2u", [])
    if u2u:
        d = np.array([e["d"] for e in u2u])
        r = np.array([e["r"] for e in u2u])
        print(f"\n[3] LINK U2U: {len(u2u)} collegamenti, "
              f"distanza media {d.mean():.2f} m, max {d.max():.2f} m")
        print(f"   Rate medio: {r.mean():.1f}, min: {r.min():.1f}")
    else:
        print("\n[3] Nessun link U2U nel dataset.")


if __name__ == "__main__":
    CONFIG_PATH = Path(__file__).resolve().parents[1] / "config_env.json"
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("File config_env.json non trovato nella directory principale.")

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    dataset_path = cfg.get("dataset", {}).get("path_train", "dataset_addestramento/dataset.json")
    SCN_PATH = Path(dataset_path)

    if not SCN_PATH.exists():
        raise FileNotFoundError(f"Dataset non trovato: {SCN_PATH.resolve()}")

    verify(SCN_PATH)
