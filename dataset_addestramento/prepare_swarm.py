#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_swarm.py — genera scenario 3D tipo stormo orbitante con respiro dinamico
Compatibile con datacom_full.py, verify_scenario.py, viz_infer.py.
"""

import json, math, random
from pathlib import Path
import numpy as np


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# === Caricamento configurazione ===
CONFIG_PATH = Path("config_env.json")
if not CONFIG_PATH.exists():
    raise FileNotFoundError("File config_env.json non trovato.")

cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
swarm_cfg = cfg.get("swarm_generation", {})
env_cfg = cfg.get("environment", {})
dataset_cfg = cfg.get("dataset", {})

OUT_PATH = Path(dataset_cfg.get("path_train", "dataset_addestramento/dataset.json"))
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# === Parametri ===
n_uav = swarm_cfg.get("n_uav", 20)
n_gs = swarm_cfg.get("n_gs", 4)
slots = swarm_cfg.get("slots", 300)
area = swarm_cfg.get("area", 100)
height = swarm_cfg.get("height", 10)
seed = swarm_cfg.get("seed", 42)
alpha = env_cfg.get("alpha", 2.0)

max_speed = swarm_cfg.get("max_speed", 4.0)
vision = swarm_cfg.get("vision", 30.0)
u2u_radius = swarm_cfg.get("behavior", {}).get("u2u_radius", 25.0)


# === Generatore stormo ===
def make_swarm_3d():
    random.seed(seed)
    np.random.seed(seed)

    scene = {
        "meta": {
            "name": "dataset.json",
            "flight": "Flying_swarm_3D_breathing",
            "slots": slots,
            "uav": n_uav,
            "gs": n_gs,
            "area_m": area,
            "height_m": height,
            "path_loss_alpha": alpha
        },
        "slots": []
    }

    # 4 GS ai vertici
    gs_positions = [
        [0, 0, 0.5],
        [area, 0, 0.5],
        [area, area, 0.5],
        [0, area, 0.5],
    ][:n_gs]

    # Posizioni iniziali: cerchio 3D con fasi casuali
    phases = np.linspace(0, 2 * math.pi, n_uav, endpoint=False)
    base_radii = np.random.uniform(12, 20, size=n_uav)
    heights = np.random.uniform(3, height - 2, size=n_uav)
    center = np.array([area / 2, area / 2, height / 2])
    positions = np.zeros((n_uav, 3))
    velocities = np.zeros_like(positions)

    for i in range(n_uav):
        positions[i] = [
            center[0] + base_radii[i] * math.cos(phases[i]),
            center[1] + base_radii[i] * math.sin(phases[i]),
            heights[i]
        ]

    # --- simulazione ---
    for t in range(slots):
        slot_data = {"t": t, "uav": [], "gs": [], "links": [], "u2u": []}

        # Orbita del baricentro: ellisse con oscillazione verticale
        cx = area / 2 + 15 * math.cos(2 * math.pi * t / slots)
        cy = area / 2 + 12 * math.sin(2 * math.pi * t / slots)
        cz = height / 2 + 2.5 * math.sin(2 * math.pi * t / (slots / 2))
        center = np.array([cx, cy, cz])

        # "Respiro" dello stormo → raggio orbitale si espande e contrae
        breath = 1.0 + 0.35 * math.sin(2 * math.pi * t / (slots / 4))

        new_positions = []
        for i in range(n_uav):
            phi = phases[i] + 2 * math.pi * t / slots
            # raggio individuale dinamico (respiro globale + piccola variazione personale)
            r = base_radii[i] * breath * (1.0 + 0.05 * math.sin(phi + i))
            # quota oscillante
            z = center[2] + 2.5 * math.sin(phi + i / 3)
            x = center[0] + r * math.cos(phi + i / 5)
            y = center[1] + r * math.sin(phi + i / 5)
            pos = np.array([x, y, z])

            # velocità (approssimata come delta)
            vel = (pos - positions[i]) * 0.8
            speed = float(np.linalg.norm(vel))
            if speed > max_speed:
                vel = vel / speed * max_speed

            new_positions.append(pos)
            velocities[i] = vel

            backlog = max(0.0, np.random.normal(150.0, 60.0))
            deadline = max(1.0, np.random.uniform(1.0, 6.0))
            relay_budget = max(0.0, np.random.normal(160.0, 40.0))

            slot_data["uav"].append({
                "id": f"u{i}",
                "f1": backlog,
                "f2": deadline,
                "f3": speed,
                "r1": relay_budget,
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2])
            })

        positions = np.array(new_positions)

        # GS
        for j, g in enumerate(gs_positions):
            cap = max(30.0, np.random.normal(120.0, 30.0))
            lat = max(5.0, np.random.normal(20.0, 8.0))
            q = max(0.0, np.random.normal(20.0, 10.0))
            slot_data["gs"].append({
                "id": f"g{j}",
                "c1": cap, "c2": lat, "c3": q,
                "x": g[0], "y": g[1], "z": g[2]
            })

        # UAV→GS link
        for u in slot_data["uav"]:
            upos = np.array([u["x"], u["y"], u["z"]])
            for g in slot_data["gs"]:
                gpos = np.array([g["x"], g["y"], g["z"]])
                d = float(np.linalg.norm(upos - gpos))
                snr = 1.0 / (d ** alpha + 1e-6)
                rate = clamp(320 * math.log2(1 + snr), 1, 320)
                slot_data["links"].append({
                    "uav_id": u["id"], "gs_id": g["id"],
                    "w1": d, "w2": rate
                })

        # UAV→UAV link
        for i in range(n_uav):
            for j in range(i + 1, n_uav):
                d = float(np.linalg.norm(positions[i] - positions[j]))
                if d < u2u_radius:
                    r = max(1.0, 240.0 - 0.20 * d + np.random.randn() * 3.0)
                    slot_data["u2u"].append({"a_id": f"u{i}", "b_id": f"u{j}", "d": d, "r": r})
                    slot_data["u2u"].append({"a_id": f"u{j}", "b_id": f"u{i}", "d": d, "r": r})

        scene["slots"].append(slot_data)

    return scene


# === MAIN ===
if __name__ == "__main__":
    scn = make_swarm_3d()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(scn, f, indent=2)
    print(f"[OK] Scenario 3D con respiro dinamico generato in {OUT_PATH}")
