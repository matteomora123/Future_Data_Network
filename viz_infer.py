#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_infer.py — Visualizzazione 3D del flusso UAV–GS (Swarm 3D)
Legge automaticamente percorsi e parametri da config_env.json.
Compatibile con datacom_full.py e dataset_addestramento/dataset.json.
"""

import json, torch, numpy as np, sys
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


def draw_slot(ax, slot, mapping, relay_plan, idx, total):
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

    # Link attivi (con relay evidenziati)
    for u in slot.uav:
        dest = mapping.get(u.id, "NO_TX")
        relay_info = relay_plan.get(u.id, {"relay": "", "use_relay": False})
        relay_id = relay_info["relay"]
        use_relay = relay_info["use_relay"]

        if use_relay and relay_id:
            # UAV → Relay (arancione)
            try:
                v = next(v for v in slot.uav if v.id == relay_id)
                ax.plot([u.x, v.x], [u.y, v.y], [u.z, v.z],
                        color='orange', linewidth=2.5, alpha=0.9, label=None)
            except StopIteration:
                continue

            # Relay → GS (ciano chiaro)
            relay_dest = mapping.get(relay_id, "NO_TX")
            if relay_dest.startswith("g"):
                g = next(g for g in slot.gs if g.id == relay_dest)
                ax.plot([v.x, g.x], [v.y, g.y], [v.z, g.z],
                        color='deepskyblue', linewidth=2.0, alpha=0.9, label=None)

        elif dest.startswith("g"):
            # Collegamento diretto U2G (ciano)
            g = next(g for g in slot.gs if g.id == dest)
            ax.plot([u.x, g.x], [u.y, g.y], [u.z, g.z],
                    color='cyan', linewidth=2.0, alpha=0.8, label=None)

        elif dest.startswith("u"):
            # Collegamento U2U non relay (fallback)
            v = next(v for v in slot.uav if v.id == dest)
            ax.plot([u.x, v.x], [u.y, v.y], [u.z, v.z],
                    color='orange', linewidth=1.5, alpha=0.7, linestyle='--', label=None)

    ax.legend(loc='upper right')


def visualize(save_mp4=False):
    """Esegue inferenza e visualizza il comportamento del modello addestrato"""
    print(f"[Config] Dataset base: {SCN_PATH}")
    print(f"[Config] Checkpoint:   {CKPT_PATH}")

    device = get_device()
    scn = load_scenario(SCN_PATH)
    policy = PPOPolicy().to(device)
    policy.load_state_dict(torch.load(CKPT_PATH, map_location=device))

    assignments = run_inference(policy, scn)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    if save_mp4:
        import cv2
        out_path = Path("viz_output.mp4")
        fps = 10
        frame_size = (800, 700)
        writer = cv2.VideoWriter(str(out_path),
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, frame_size)
        print(f"[Video] Registrazione attiva → {out_path}")

    for slot, (mapping, relay_plan) in zip(scn.slots, assignments):
        draw_slot(ax, slot, mapping, relay_plan, slot.t, len(scn.slots))
        plt.pause(0.05)

        if save_mp4:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            frame = np.frombuffer(renderer.buffer_rgba(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            frame = frame[:, :, :3]  # elimina canale alfa
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, frame_size)
            writer.write(frame)

    if save_mp4:
        writer.release()
        print(f"[Video] Salvataggio completato: {out_path.resolve()}")

    plt.show()


if __name__ == "__main__":
    save_mp4 = "--mp4" in sys.argv
    visualize(save_mp4=save_mp4)
