#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizza in modo estetico il grafo UAV–GS per uno slot
--------------------------------------------------------
Uso:
    python visualize_slot_graph.py --slot 0
    python visualize_slot_graph.py --slot 5 --out slot5.png
"""

import json
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def nice_layout(G, radius=5.0):
    """Posiziona UAV in cerchio e GS ai bordi."""
    uavs = [n for n, d in G.nodes(data=True) if d["type"] == "uav"]
    gss = [n for n, d in G.nodes(data=True) if d["type"] == "gs"]

    pos = {}
    # UAV in cerchio
    for i, n in enumerate(uavs):
        angle = 2 * np.pi * i / len(uavs)
        pos[n] = (radius * np.cos(angle), radius * np.sin(angle))
    # GS disposti ai vertici di un rettangolo esterno
    r2 = radius * 2.2
    for i, n in enumerate(gss):
        angle = 2 * np.pi * i / len(gss)
        pos[n] = (r2 * np.cos(angle), r2 * np.sin(angle))
    return pos


def visualize_slot(data, slot_id, out_path=None):
    slots = data["slots"]
    if slot_id >= len(slots):
        raise IndexError(f"Slot {slot_id} fuori range (max {len(slots)-1})")

    slot = slots[slot_id]
    G = nx.Graph()

    # UAV nodes
    for u in slot["uav"]:
        uid = u["id"]
        label = f"{uid}\n(f1={u['f1']:.0f}, f2={u['f2']:.1f}, f3={u['f3']:.1f})"
        G.add_node(uid, type="uav", label=label)

    # GS nodes
    for g in slot["gs"]:
        gid = g["id"]
        label = f"{gid}\n(c1={g['c1']:.0f}, c2={g['c2']:.1f}, c3={g['c3']:.1f})"
        G.add_node(gid, type="gs", label=label)

    # Links
    if "links" in slot:
        for link in slot["links"]:
            u = link.get("u") or link.get("uav") or link.get("src")
            g = link.get("g") or link.get("gs") or link.get("dst")
            if not (u and g):
                continue
            w1, w2 = link.get("w1", 0), link.get("w2", 0)
            lbl = f"w1={w1:.0f}, w2={w2:.0f}"
            G.add_edge(u, g, label=lbl)

    pos = nice_layout(G)
    color_map = ["#4F9DFF" if d["type"] == "uav" else "#FFA84F" for _, d in G.nodes(data=True)]

    plt.figure(figsize=(11, 8))
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=1800,
                           edgecolors="black", linewidths=0.8, alpha=0.95)
    nx.draw_networkx_labels(G, pos,
                            labels={n: d["label"] for n, d in G.nodes(data=True)},
                            font_size=8.5, font_family="DejaVu Sans", font_weight="bold")

    nx.draw_networkx_edges(G, pos, width=1.2, alpha=0.4, edge_color="gray", connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_edge_labels(G, pos,
                                 edge_labels=nx.get_edge_attributes(G, "label"),
                                 font_size=7, label_pos=0.5, rotate=False)

    title = f"Slot {slot_id} – UAV–GS Graph ({data['meta'].get('flight','')})"
    plt.title(title, fontsize=13, weight="bold", pad=20)
    plt.axis("off")
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
        print(f"[✔] Immagine salvata in: {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualizza grafo UAV–GS (layout estetico)")
    parser.add_argument("--dataset", default="dataset_addestramento/dataset.json",
                        help="Percorso al file dataset.json (default: dataset_addestramento/dataset.json)")
    parser.add_argument("--slot", type=int, required=True, help="Indice dello slot da visualizzare")
    parser.add_argument("--out", type=str, default=None, help="File immagine di output (es. slot0.png)")
    args = parser.parse_args()

    path = Path(args.dataset)
    if not path.exists():
        raise FileNotFoundError(f"Dataset non trovato: {path}")

    data = load_dataset(path)
    visualize_slot(data, args.slot, args.out)


if __name__ == "__main__":
    main()
