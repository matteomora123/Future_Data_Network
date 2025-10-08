#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_relays.py — Analizza il file di submission e verifica la presenza di collegamenti U2U (relay).
Legge automaticamente i percorsi da config_env.json.
Mostra statistiche per slot e riepilogo complessivo.
"""

import json
from pathlib import Path
from collections import Counter

# === Caricamento config ===
CONFIG_PATH = Path("config_env.json")
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"File di configurazione mancante: {CONFIG_PATH.resolve()}")

cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
SUB_PATH = Path(cfg.get("paths", {}).get("inference_output", "submission.json"))

if not SUB_PATH.exists():
    raise FileNotFoundError(f"File submission non trovato: {SUB_PATH.resolve()}")

print(f"\n[Config] Caricato {CONFIG_PATH}")
print(f"[Input]  Submission: {SUB_PATH}\n")

# === Caricamento submission ===
data = json.loads(SUB_PATH.read_text(encoding="utf-8"))
assignments = data.get("assignments", [])

if not assignments:
    raise ValueError("⚠️ Nessun record 'assignments' trovato nel file submission.")

print(f"[OK] Submission caricata con {len(assignments)} slot.\n")

# === Analisi relay ===
total_relays = 0
relay_uavs = Counter()
slots_with_relay = 0
slots_total = len(assignments)

for rec in assignments:
    t = rec.get("t", "?")
    relay_map = rec.get("relay", {})
    n_relay = 0

    for uid, info in relay_map.items():
        if info.get("use_relay", False) and info.get("relay", ""):
            total_relays += 1
            n_relay += 1
            relay_uavs[info["relay"]] += 1

    if n_relay > 0:
        slots_with_relay += 1
        print(f"[Slot {t:>3}] {n_relay} UAV con relay attivo")

# === Riepilogo globale ===
print("\n=== STATISTICHE GLOBALI ===")
print(f"Slot totali:              {slots_total}")
print(f"Slot con almeno un relay: {slots_with_relay}")
print(f"Totale relay usati:       {total_relays}")

if slots_total > 0:
    perc_slots = (slots_with_relay / slots_total) * 100
    print(f"Percentuale slot con relay: {perc_slots:.1f}%")

if total_relays > 0:
    top_relays = relay_uavs.most_common(5)
    print("\nTop 5 UAV più usati come relay:")
    for uid, cnt in top_relays:
        print(f"  {uid:<6} → {cnt} volte")
else:
    print("\n⚠️ Nessun collegamento U2U (relay) trovato nel submission.")

print()
