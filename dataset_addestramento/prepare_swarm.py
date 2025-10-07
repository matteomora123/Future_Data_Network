#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_swarm_model.py — genera scenario 3D realistico tipo stormo dinamico (swarm)
usando il modello Flocking del pacchetto vmodel (EPFL LIS).
Compatibile con datacom_full.py e verifica con verify_scenario.py.

Dipendenze:
    pip install -e ./vmodel  (repo ufficiale: https://github.com/lis-epfl/vmodel)

Legge i parametri da config_env.json → sezione "swarm_generation".
"""

import json
import numpy as np
from pathlib import Path
from vmodel.examples.example_flocking import Flocking
from vmodel.simulator import Simulator


# === Carica configurazione globale ===
CONFIG_PATH = Path("config_env.json")
if not CONFIG_PATH.exists():
    raise FileNotFoundError("File config_env.json non trovato nella directory principale.")

cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
swarm_cfg = cfg.get("swarm_generation", {})
dataset_cfg = cfg.get("dataset", {})

OUT_PATH = Path(dataset_cfg.get("path_train", "dataset_addestramento/dataset.json"))
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

slots = swarm_cfg.get("slots", 300)
n_uav = swarm_cfg.get("n_uav", 20)
n_gs = swarm_cfg.get("n_gs", 4)
area = swarm_cfg.get("area", 100)
height = swarm_cfg.get("height", 10)
seed = swarm_cfg.get("seed", 42)


# === Inizializza simulazione vmodel ===
np.random.seed(seed)
sim = Simulator()
env = Flocking(n_agents=n_uav)
sim.set_environment(env)

print(f"[INIT] Avvio simulazione Flocking con {n_uav} droni, {slots} slot...")

sim.run(steps=slots)

# === Costruzione scenario JSON compatibile ===
scene = {
    "meta": {
        "name": "dataset_vmodel.json",
        "flight": "Flying_swarm_3D_vmodel",
        "slots": slots,
        "uav": n_uav,
        "gs": n_gs,
        "area_m": area,
        "height_m": height,
        "generator": "vmodel.Flocking"
    },
    "slots": []
}

# Posizioni GS agli angoli dell'area
gs_positions = [
    [0, 0, 0.5],
    [area, 0, 0.5],
    [area, area, 0.5],
    [0, area, 0.5],
][:n_gs]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


for t in range(slots):
    slot_data = {"t": t, "uav": [], "gs": [], "links": [], "u2u": []}

    # === UAV ===
    for i, a in enumerate(env.agents):
        pos = a.position
        vel = a.velocity
        backlog = max(0.0, np.random.normal(150.0, 60.0))
        deadline = max(1.0, np.random.uniform(1.0, 6.0))
        relay_budget = max(0.0, np.random.normal(160.0, 40.0))
        slot_data["uav"].append({
            "id": f"u{i}",
            "f1": backlog,
            "f2": deadline,
            "f3": float(np.linalg.norm(vel)),
            "r1": relay_budget,
            "x": float(clamp(pos[0] * area, 0, area)),
            "y": float(clamp(pos[1] * area, 0, area)),
            "z": float(clamp(abs(pos[2]) * height, 1.0, height))
        })

    # === Ground Stations ===
    for j, g in enumerate(gs_positions):
        cap = max(30.0, np.random.normal(120.0, 30.0))
        lat = max(5.0, np.random.normal(20.0, 8.0))
        q = max(0.0, np.random.normal(20.0, 10.0))
        slot_data["gs"].append({
            "id": f"g{j}",
            "c1": cap,
            "c2": lat,
            "c3": q,
            "x": g[0],
            "y": g[1],
            "z": g[2]
        })

    # === Link UAV → GS ===
    alpha = 2.0
    for u in slot_data["uav"]:
        upos = np.array([u["x"], u["y"], u["z"]])
        for g in slot_data["gs"]:
            gpos = np.array([g["x"], g["y"], g["z"]])
            d = float(np.linalg.norm(upos - gpos))
            snr = 1.0 / (d ** alpha + 1e-6)
            rate = clamp(320 * np.log2(1 + snr), 1, 320)
            slot_data["links"].append({
                "uav_id": u["id"],
                "gs_id": g["id"],
                "w1": d,
                "w2": rate
            })

    # === Link U2U ===
    for i in range(n_uav):
        for j in range(i + 1, n_uav):
            pi = np.array([slot_data["uav"][i]["x"], slot_data["uav"][i]["y"], slot_data["uav"][i]["z"]])
            pj = np.array([slot_data["uav"][j]["x"], slot_data["uav"][j]["y"], slot_data["uav"][j]["z"]])
            d = float(np.linalg.norm(pi - pj))
            if d < 25.0:
                r = max(1.0, 240.0 - 0.20 * d + np.random.randn() * 3.0)
                slot_data["u2u"].append({"a_id": f"u{i}", "b_id": f"u{j}", "d": d, "r": r})
                slot_data["u2u"].append({"a_id": f"u{j}", "b_id": f"u{i}", "d": d, "r": r})

    scene["slots"].append(slot_data)

# === Salva output ===
OUT_PATH.write_text(json.dumps(scene, indent=2))
print(f"[OK] Scenario generato e salvato in {OUT_PATH}")
