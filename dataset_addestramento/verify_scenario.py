#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verifica completa di coerenza per scenario_env0_*.json
- Controlla range coordinate
- Confronta distanze UAV–GS con w1
- Verifica numero e range dei link U2U
- Mostra grafico 2D iniziale
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCN_PATH = Path("scenario_env0_concat.json")  # o scenario_env0_ch5.json

# --- Carica file JSON ---
d = json.load(open(SCN_PATH, "r"))
slots = d["slots"]
print(f"\nScenario: {d['meta']['name']} | slot totali: {len(slots)}")

# --- 1. Verifica range posizioni ---
u0 = slots[0]["uav"]
g0 = slots[0]["gs"]

u_coords = np.array([[u["x"], u["y"], u["z"]] for u in u0])
g_coords = np.array([[g["x"], g["y"], g["z"]] for g in g0])

print("\n[1] RANGE COORDINATE (m)")
print(f"UAV: x∈[{u_coords[:,0].min():.2f}, {u_coords[:,0].max():.2f}], "
      f"y∈[{u_coords[:,1].min():.2f}, {u_coords[:,1].max():.2f}], "
      f"z∈[{u_coords[:,2].min():.2f}, {u_coords[:,2].max():.2f}]")
print(f"GS : x∈[{g_coords[:,0].min():.2f}, {g_coords[:,0].max():.2f}], "
      f"y∈[{g_coords[:,1].min():.2f}, {g_coords[:,1].max():.2f}], "
      f"z∈[{g_coords[:,2].min():.2f}, {g_coords[:,2].max():.2f}]")

# --- 2. Verifica distanze U2G (slot 0) ---
print("\n[2] VERIFICA DISTANZE U2G (slot 0)")
links = slots[0]["links"]
diffs = []
for lk in links:
    u = next(u for u in u0 if u["id"] == lk["uav_id"])
    g = next(g for g in g0 if g["id"] == lk["gs_id"])
    d_calc = np.linalg.norm(np.array([u["x"], u["y"], u["z"]]) - np.array([g["x"], g["y"], g["z"]]))
    diffs.append(abs(d_calc - lk["w1"]))
print(f"Errore medio distanza (m): {np.mean(diffs):.3f}")

# --- 3. Verifica link U2U ---
if "u2u" in slots[0]:
    u2u = slots[0]["u2u"]
    dists = [lk["d"] for lk in u2u]
    print(f"\n[3] LINK U2U: {len(u2u)} collegamenti, distanza media {np.mean(dists):.2f} m, max {np.max(dists):.2f} m")
    print(f"   Rate medio: {np.mean([lk['r'] for lk in u2u]):.1f}, min: {np.min([lk['r'] for lk in u2u]):.1f}")
else:
    print("\n[3] Nessun link U2U trovato in slot[0].")

# --- 4. Visualizzazione 2D ---
plt.figure(figsize=(6, 5))
plt.scatter(g_coords[:, 0], g_coords[:, 1], c='green', s=80, label='GS (anchors)')
plt.scatter(u_coords[:, 0], u_coords[:, 1], c='dodgerblue', s=50, label='UAV')
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Distribuzione spaziale iniziale (slot 0)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
