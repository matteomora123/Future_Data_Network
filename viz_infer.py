#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_infer.py - Animazione 3D del flusso dati in scenario NWPU.
Mostra i link attivi e l'illuminazione dei collegamenti scelti.
"""

import argparse, torch, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datacom_full import (
    load_scenario, PPOPolicy, compute_augmented_edges,
    safety_project_aug, get_device
)

# === Parametri principali ===
CKPT_DIR = Path(r"C:\Users\matte\PycharmProjects\Future_Data_Network\Checkpoint")
SCN_PATH = Path(r"C:\Users\matte\PycharmProjects\Future_Data_Network\dataset_addestramento\scenario_env0_ch5.json")
RELAY_FACTOR = 0.9
CAP_SCALE = 30.0
RELAY_SCALE = 30.0


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


def draw_slot(ax, slot, mapping, epoch_idx, total_epochs):
    """Disegna la scena per un singolo slot"""
    ax.clear()
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    ax.set_xlim(0, 20); ax.set_ylim(0, 18); ax.set_zlim(0, 8)
    ax.set_title(f"Epoca {epoch_idx}/{total_epochs} — Slot {slot.t}")

    # GS
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


def replay_training():
    """Riproduce i checkpoint in sequenza"""
    device = get_device()
    scn = load_scenario(SCN_PATH)

    ckpts = sorted(list(CKPT_DIR.glob("policy_epoch_*.pt")))
    if not ckpts:
        print(f"Nessun checkpoint trovato in {CKPT_DIR}")
        return

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    for i, ckpt in enumerate(ckpts, 1):
        print(f"[Replay] Epoca {i}/{len(ckpts)} – {ckpt.name}")
        policy = PPOPolicy().to(device)
        policy.load_state_dict(torch.load(ckpt, map_location=device))

        assignments = run_inference(policy, scn)

        for slot, (mapping, _) in zip(scn.slots, assignments):
            draw_slot(ax, slot, mapping, i, len(ckpts))
            plt.pause(0.05)  # velocità animazione

        plt.pause(0.5)  # pausa tra epoche

    plt.show()


if __name__ == "__main__":
    replay_training()
