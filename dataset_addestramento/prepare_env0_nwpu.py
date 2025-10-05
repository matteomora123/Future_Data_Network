#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_env0_nwpu.py — costruisce uno Scenario JSON (Environment0) per datacom_full.py

Input:
  --root    = cartella del dataset scompattato (contiene Environment0/, Raw_data/, ...)
Scelte:
  --flight  ∈ {Flying_straight, Flying_climb}
  --channel ∈ {2,3,4,5}  (oppure --channels 2,3,4,...)
  --merge   ∈ {per_ch, concat, average}
Output:
  --out     = file JSON (o più file) con (meta, slots[t].{uav,gs,links})

Note:
- Converte esplicitamente le distanze da cm → metri.
- w2 (bitrate sintetico) è funzione decrescente della distanza in metri (clamp [1,320]).
- Filename robusti: gestisce sia Distance_anchor_label_* che Dis_anchor_label_*,
  e sia Position_label_* che position_label_*.
- Modalità:
    • per_ch  : genera un file per ogni canale (suffisso _chN)
    • concat  : concatena gli slot dei canali (T = 30 * #ch)
    • average : media P e D sui canali in parallelo (T = 30)

Dipendenze: numpy, scipy (scipy.io); fallback h5py per .mat v7.3 (HDF5)
"""

import argparse, json
from pathlib import Path
import numpy as np


# --------------------- util ---------------------
def load_mat(fp: Path) -> dict:
    """Carica un .mat (v5 o v7.3) e restituisce un dict numpy-ready."""
    if not fp.exists():
        raise FileNotFoundError(fp)
    try:
        from scipy.io import loadmat
        mat = loadmat(str(fp))
        return {k: v for k, v in mat.items() if not k.startswith("__")}
    except Exception:
        import h5py
        out = {}
        with h5py.File(fp, "r") as f:
            def to_np(obj):
                a = np.array(obj)
                # HDF5 salva per colonne; se è 2D spesso va trasposto
                if a.ndim == 2 and 1 < a.shape[0] < 64 and 1 < a.shape[1] < 64:
                    return a.T
                return a
            for k in f.keys():
                out[k] = to_np(f[k])
        return out


def find_array_2d(d: dict, target_shapes=((8,3),(3,8))) -> np.ndarray:
    for v in d.values():
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.size > 0:
            if v.shape in target_shapes:
                return v if v.shape == (8,3) else v.T
    # prova trasposizioni
    for v in d.values():
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.size > 0:
            if v.T.shape in ((8,3),(3,8)):
                return v.T if v.T.shape == (8,3) else v
    raise RuntimeError("Anchors.mat: non trovo matrice 8x3 (coordinate GS)")


def permute_to_shape(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    if arr.shape == target_shape:
        return arr
    if arr.ndim != len(target_shape):
        raise RuntimeError(f"Atteso array {len(target_shape)}D, trovato {arr.ndim}D")
    import itertools
    for perm in itertools.permutations(range(arr.ndim)):
        if arr.transpose(perm).shape == target_shape:
            return arr.transpose(perm)
    raise RuntimeError(f"Impossibile permutare array da {arr.shape} a {target_shape}")


def find_array_3d(d: dict, target_shape: tuple) -> np.ndarray:
    # cerca array già conforme
    for v in d.values():
        if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape == target_shape:
            return v
    # prova a permutare
    for v in d.values():
        if isinstance(v, np.ndarray) and v.ndim == 3:
            try:
                return permute_to_shape(v, target_shape)
            except RuntimeError:
                pass
    # ultima spiaggia
    for v in d.values():
        if isinstance(v, np.ndarray) and v.ndim == 3:
            return permute_to_shape(v, target_shape)
    raise RuntimeError(f"Non trovo array 3D riconducibile a {target_shape}")


# --------------------- loader posizioni & distanze ---------------------
def load_pos_dist(root: Path, flight: str, ch: int) -> tuple[np.ndarray, np.ndarray]:
    base = root / "Raw_data" / "Environment0" / flight
    # posizione (7,3,30) — gestisci maiuscole/minuscole nel filename
    pos_fp1 = base / "Position_Label" / f"Position_label_ch{ch}.mat"
    pos_fp2 = base / "Position_Label" / f"position_label_ch{ch}.mat"
    pos_fp = pos_fp1 if pos_fp1.exists() else pos_fp2
    if not pos_fp.exists():
        raise FileNotFoundError(f"Position_Label non trovato: {pos_fp1.name} | {pos_fp2.name}")
    P = find_array_3d(load_mat(pos_fp), (7,3,30)).astype(np.float32)

    # distanza (8,7,30) — filename robusto (Distance_* o Dis_*)
    dis_dir = base / "Distance_Anchor_Label"
    cand1 = dis_dir / f"Distance_anchor_label_ch{ch}.mat"
    cand2 = dis_dir / f"Dis_anchor_label_ch{ch}.mat"
    dis_fp = cand1 if cand1.exists() else cand2
    if not dis_fp or not dis_fp.exists():
        raise FileNotFoundError(f"Distance_Anchor_Label non trovato: {cand1.name} | {cand2.name}")
    D_cm = find_array_3d(load_mat(dis_fp), (8,7,30)).astype(np.float32)
    D_m = D_cm / 100.0  # cm → metri
    return P, D_m


# --------------------- builder da P,D ---------------------
def build_from_PD(P: np.ndarray, D_m: np.ndarray, flight: str, ch: int, seed: int) -> dict:
    rng = np.random.RandomState(seed)
    T, U, G = 30, 7, 8
    gs_ids = [f"g{i}" for i in range(G)]

    # GS features fisse nello scenario
    gs_feat = []
    for _ in range(G):
        c1 = float(max(30.0, rng.normal(120.0, 30.0)))
        c2 = float(max(5.0,  rng.normal(20.0, 8.0)))
        c3 = float(max(0.0,  rng.normal(20.0, 10.0)))
        gs_feat.append((c1, c2, c3))

    slots = []
    for t in range(T):
        # UAV list
        uav = []
        for u in range(U):
            speed = 0.0 if t == 0 else float(np.linalg.norm(P[u, :, t] - P[u, :, t-1]))
            f1 = float(max(0.0, rng.normal(150.0, 60.0)))   # backlog (astratto)
            f2 = float(max(1.0,  rng.uniform(1.0, 6.0)))    # deadline (s, astratto)
            f3 = speed
            uav.append({"id": f"u{u}", "f1": f1, "f2": f2, "f3": f3})

        # GS list
        gs = [{"id": gs_ids[g], "c1": gs_feat[g][0], "c2": gs_feat[g][1], "c3": gs_feat[g][2]} for g in range(G)]

        # Links da distanze reali; bitrate w2 sintetico da distanza (metri), clamp [1,320]
        links = []
        for u in range(U):
            for g in range(G):
                w1 = float(D_m[g, u, t])  # metri
                w2_raw = 320.0 - 25.0 * w1 + rng.randn() * 3.0
                w2 = float(min(320.0, max(1.0, w2_raw)))
                links.append({"uav_id": f"u{u}", "gs_id": gs_ids[g], "w1": w1, "w2": w2})

        slots.append({"t": t, "uav": uav, "gs": gs, "links": links})

    meta = {
        "name": "nwpu_env0",
        "flight": flight,
        "channel": f"ch{ch}",
        "channels_note": "pos/dist reali; bitrate sintetico; GS feat sintetiche (seed riproducibile)",
        "units": {"pos": "m", "w1": "m", "w2": "arb_bitrate"},
        "note": "w1 in metri (da dataset); w2 ~ f(distanza) con clamp [1,320]"
    }
    return {"meta": meta, "slots": slots}


# --------------------- CLI ---------------------
def main():
    ap = argparse.ArgumentParser(description="NWPU Env0 → Scenario JSON (multi-channel)")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--flight", type=str, default="Flying_straight",
                    choices=["Flying_straight", "Flying_climb"])
    ap.add_argument("--channel", type=int, default=None, help="canale singolo (2..5)")
    ap.add_argument("--channels", type=str, default=None, help="lista canali, es. '2,3,4'")
    ap.add_argument("--merge", type=str, default="per_ch", choices=["per_ch", "concat", "average"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pretty", action="store_true", help="salva JSON con indent=2 (debug)")
    ap.add_argument("--include-anchors", action="store_true", help="inserisci coordinates GS in meta.anchors")
    args = ap.parse_args()

    root = args.root
    flight = args.flight
    seed = args.seed

    # Anchors (per meta opzionale)
    anchors_mat = load_mat(root / "Environment0" / "Anchors.mat")
    A = find_array_2d(anchors_mat, target_shapes=((8,3),(3,8))).astype(np.float32)  # (8,3)

    # Normalizza canali
    if args.channels:
        ch_list = [int(x.strip()) for x in args.channels.split(",") if x.strip()]
        ch_list = sorted(set(ch_list))
    else:
        if args.channel is None:
            raise SystemExit("Specificare --channel N oppure --channels 2,3,4,5")
        ch_list = [args.channel]

    indent = 2 if args.pretty else None

    # BUILD in funzione della modalità
    if args.merge == "per_ch":
        # un file per canale: out funge da prefisso, aggiungo _chN.json
        for ch in ch_list:
            P, D = load_pos_dist(root, flight, ch)
            scn = build_from_PD(P, D, flight, ch, seed)
            out_path = args.out.with_name(f"{args.out.stem}_ch{ch}{args.out.suffix}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(scn, ensure_ascii=False, indent=indent))
            print(f"OK → {out_path}  (slots={len(scn['slots'])}, "
                  f"uav/slot={len(scn['slots'][0]['uav'])}, "
                  f"gs/slot={len(scn['slots'][0]['gs'])}, "
                  f"links/slot={len(scn['slots'][0]['links'])})")

    elif args.merge == "concat":
        # concatena gli slot: T = 30 * #ch
        big_slots = []
        slot_channels = []
        for ch in ch_list:
            P, D = load_pos_dist(root, flight, ch)
            scn = build_from_PD(P, D, flight, ch, seed + ch)  # seed variato per diversificare i sintetici
            for s in scn["slots"]:
                s["t"] = len(big_slots)  # indice progressivo
                big_slots.append(s)
                slot_channels.append(f"ch{ch}")
        meta = {
            "name": "nwpu_env0_concat",
            "flight": flight,
            "channels": [f"ch{c}" for c in ch_list],
            "slot_channel": slot_channels,
            "units": {"pos": "m", "w1": "m", "w2": "arb_bitrate"},
            "note": "Concatenazione degli slot dei canali; pos/dist reali, bitrate sintetico"
        }
        obj = {"meta": meta, "slots": big_slots}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(obj, ensure_ascii=False, indent=indent))
        print(f"OK → {args.out}  (slots={len(big_slots)}, "
              f"uav/slot={len(big_slots[0]['uav'])}, "
              f"gs/slot={len(big_slots[0]['gs'])}, "
              f"links/slot={len(big_slots[0]['links'])})")

    else:  # average
        # media per-t su canali: P e D mediati
        P_list, D_list = [], []
        for ch in ch_list:
            P, D = load_pos_dist(root, flight, ch)
            P_list.append(P); D_list.append(D)
        P_avg = np.mean(np.stack(P_list, axis=0), axis=0).astype(np.float32)
        D_avg = np.mean(np.stack(D_list, axis=0), axis=0).astype(np.float32)
        scn = build_from_PD(P_avg, D_avg, flight, 0, seed)
        scn["meta"]["channel"] = "avg(" + ",".join(f"ch{c}" for c in ch_list) + ")"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(scn, ensure_ascii=False, indent=indent))
        print(f"OK → {args.out}  (slots={len(scn['slots'])}, "
              f"uav/slot={len(scn['slots'][0]['uav'])}, "
              f"gs/slot={len(scn['slots'][0]['gs'])}, "
              f"links/slot={len(scn['slots'][0]['links'])})")


if __name__ == "__main__":
    main()
