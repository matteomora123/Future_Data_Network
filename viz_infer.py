#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_infer.py — Visualizzazione 3D del flusso UAV–GS (Swarm 3D)
Legge automaticamente percorsi e parametri da config_env.json.
Compatibile con datacom_full.py e dataset_addestramento/dataset.json.
"""

import json, torch, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datacom_full import (
    load_scenario, PPOPolicy, compute_augmented_edges,
    safety_project_aug, get_device
)

# === Carica configurazione globale ===
CONFIG_PATH = Path("config_env.json")
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"File di configurazione mancante: {CONFIG_PATH}")

config = json.loads(CONFIG_PATH.read_text())

SCN_PATH = Path(config["paths"]["inference_input"])
CKPT_PATH = Path(config["paths"]["checkpoint_file"])

RELAY_FACTOR = config["training"].get("relay_factor", 0.8)
CAP_SCALE = config["training"].get("cap_scale", 30.0)
RELAY_SCALE = config["training"].get("relay_scale", 30.0)

# === Funzioni principali ===
def run_inference(policy, scn):
    """Esegue la policy su tutti gli slot dello scenario e restituisce mapping e relay"""
    policy.eval()
    assignments = []
    with torch.no_grad():
        for slot in scn.slots:
            aug, overrides = compute_augmented_edges(slot, relay_factor=RELAY_FACTOR) if slot.u2u else ({}, {})
            probs, _ = policy.forward_policy(slot, overrides=overrides if overrides else None)
            mapping, relay_plan = safety_project_aug(slot, probs, aug,
                                                     cap_scale=CAP_SCALE, relay_scale=RELAY_SCALE)
            assignments.append((mapping, relay_plan))
    return assignments


def draw_slot(ax, slot, mapping, idx, total):
    """Disegna la scena per un singolo slot"""
    ax.clear()
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_zlim(0, 10)
    ax.set_title(f"Slot {slot.t}/{total}")

    # Ground Stations
    gs_coords = np.array([[g.x, g.y, g.z] for g in slot.gs])
    ax.scatter(gs_coords[:, 0], gs_coords[:, 1], gs_coords[:, 2],
               c='green', s=80, depthshade=True, label="GS")

    # UAV
    u_coords = np.array([[u.x, u.y, u.z] for u in slot.uav])
    ax.scatter(u_coords[:, 0], u_coords[:, 1], u_coords[:, 2],
               c='dodgerblue', s=50, depthshade=True, label="UAV")

    # Link attivi
    for u in slot.uav:
        dest = mapping.get(u.id, "NO_TX")
        if dest.startswith("g"):
            g = next(g for g in slot.gs if g.id == dest)
            ax.plot([u.x, g.x], [u.y, g.y], [u.z, g.z],
                    color='cyan', linewidth=2.0, alpha=0.9)
        elif dest.startswith("u"):
            v = next(v for v in slot.uav if v.id == dest)
            ax.plot([u.x, v.x], [u.y, v.y], [u.z, v.z],
                    color='orange', linewidth=1.5, alpha=0.8)

    ax.legend(loc='upper right')


def visualize():
    """Esegue inferenza e visualizza il comportamento del modello addestrato"""
    print(f"[Config] Dataset base: {SCN_PATH}")
    print(f"[Config] Checkpoint:   {CKPT_PATH}")

    device = get_device()
    scn = load_scenario(SCN_PATH)
    policy = PPOPolicy().to(device)
    policy.load_state_dict(torch.load(CKPT_PATH, map_location=device))

    assignments = run_inference(policy, scn)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    for slot, (mapping, _) in zip(scn.slots, assignments):
        draw_slot(ax, slot, mapping, slot.t, len(scn.slots))
        plt.pause(0.05)

    plt.show()


if __name__ == "__main__":
    visualize()
