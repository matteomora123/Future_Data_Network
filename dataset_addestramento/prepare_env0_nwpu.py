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
  --out     = file JSON (o più file) con (meta, slots[t].{uav,gs,links[,u2u]})

Note:
- Converte esplicitamente le distanze da cm → metri.
- U2G: w2 (bitrate sintetico) funzione decrescente della distanza (clamp [1,320]).
- U2U (opzionale con --u2u): r (bitrate sintetico) funzione decrescente della distanza
  con parametri configurabili (Rmax, k, noise, range), clamp [1, Rmax].
- Filename robusti per .mat: gestisce Distance_anchor_label_* / Dis_anchor_label_*
  e Position_label_* / position_label_*.
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
    P = find_array_3d(load_mat(pos_fp), (7,3,30)).astype(np.float32) / 100.0  # cm → metri

    # distanza U2G (8,7,30) — filename robusto (Distance_* o Dis_*)
    dis_dir = base / "Distance_Anchor_Label"
    cand1 = dis_dir / f"Distance_anchor_label_ch{ch}.mat"
    cand2 = dis_dir / f"Dis_anchor_label_ch{ch}.mat"
    dis_fp = cand1 if cand1.exists() else cand2
    if not dis_fp or not dis_fp.exists():
        raise FileNotFoundError(f"Distance_Anchor_Label non trovato: {cand1.name} | {cand2.name}")
    D_cm = find_array_3d(load_mat(dis_fp), (8,7,30)).astype(np.float32)
    D_m = D_cm / 100.0  # cm → metri
    return P, D_m


# --------------------- U2U helper ---------------------
def build_u2u_links_from_positions(P_t: np.ndarray,
                                   u_ids: list[str],
                                   rng: np.random.RandomState,
                                   rmax: float,
                                   k: float,
                                   noise: float,
                                   max_range: float | None,
                                   directed: bool) -> list[dict]:
    """
    Genera link U2U per lo slot corrente a partire dalle posizioni 3D in metri.
    P_t: (U, 3) posizioni all'istante t.
    """
    U = P_t.shape[0]
    links: list[dict] = []
    for i in range(U):
        for j in range(i+1, U):
            d = float(np.linalg.norm(P_t[i] - P_t[j]))
            # Ignora solo distanze palesemente fuori scala
            if (max_range is not None and d > max_range) or d < 0.01:
                continue
            # bitrate sintetico: calo lineare dolce + rumore
            r_raw = rmax - k * d + rng.randn() * noise
            r = float(np.clip(r_raw, 5.0, rmax))  # min 5.0 → mai annullato
            # salva i→j e (se directed) anche j→i
            links.append({"a_id": u_ids[i], "b_id": u_ids[j], "d": d, "r": r})
            if directed:
                links.append({"a_id": u_ids[j], "b_id": u_ids[i], "d": d, "r": r})
    return links




# --------------------- builder da P,D ---------------------
def build_from_PD(P: np.ndarray,
                  D_m: np.ndarray,
                  flight: str,
                  ch: int,
                  seed: int,
                  A: np.ndarray,
                  include_u2u: bool = False,
                  u2u_rmax: float = 160.0,
                  u2u_k: float = 35.0,
                  u2u_noise: float = 2.0,
                  u2u_range: float | None = None,
                  u2u_directed: bool = True,
                  r1_mean: float = 160.0,
                  r1_std: float = 40.0) -> dict:
    rng = np.random.RandomState(seed)
    T, U, G = 30, 7, 8
    gs_ids = [f"g{i}" for i in range(G)]
    u_ids = [f"u{i}" for i in range(U)]

    # GS features fisse nello scenario
    gs_feat = []
    for _ in range(G):
        c1 = float(max(30.0, rng.normal(120.0, 30.0)))
        c2 = float(max(5.0,  rng.normal(20.0, 8.0)))
        c3 = float(max(0.0,  rng.normal(20.0, 10.0)))
        gs_feat.append((c1, c2, c3))

    slots = []
    for t in range(T):
        # UAV list (aggiungo r1 = budget relay)
        uav = []
        for u in range(U):
            # posizione reale in metri (dal dataset)
            x, y, z = map(float, P[u, :, t])
            speed = 0.0 if t == 0 else float(np.linalg.norm(P[u, :, t] - P[u, :, t - 1]))
            f1 = float(max(0.0, rng.normal(150.0, 60.0)))  # backlog astratto
            f2 = float(max(1.0, rng.uniform(1.0, 6.0)))  # deadline
            f3 = speed
            r1 = float(max(0.0, rng.normal(r1_mean, r1_std)))  # budget relay

            # aggiungi coordinate fisiche nel JSON
            uav.append({
                "id": u_ids[u],
                "f1": f1, "f2": f2, "f3": f3, "r1": r1,
                "x": x, "y": y, "z": z
            })

        # GS list
        # per le GS uso le coordinate dagli anchors (A)
        gs = []
        for g in range(G):
            gx, gy, gz = (float(A[g, 0]) / 100.0,
                          float(A[g, 1]) / 100.0,
                          float(A[g, 2]) / 100.0)  # cm → m

            gs.append({
                "id": gs_ids[g],
                "c1": gs_feat[g][0], "c2": gs_feat[g][1], "c3": gs_feat[g][2],
                "x": gx, "y": gy, "z": gz
            })

        # Links U2G da distanze reali; bitrate w2 sintetico da distanza (metri), clamp [1,320]
        links = []
        for u in range(U):
            for g in range(G):
                w1 = float(D_m[g, u, t])  # metri
                # forma razionale più dolce, con rumore; clamp [1,320]
                # decay più forte per range 0–20 m (UWB realistico)
                w2_raw = 300.0 / (1.0 + 3.0 * w1) + rng.randn() * 8.0

                w2 = float(min(320.0, max(1.0, w2_raw)))
                links.append({"uav_id": u_ids[u], "gs_id": gs_ids[g], "w1": w1, "w2": w2})

        slot_obj = {"t": t, "uav": uav, "gs": gs, "links": links}

        # Links U2U opzionali da posizioni reali (P[:, :, t]) — chiave "u2u"
        if include_u2u:
            P_t = P[:, :, t].astype(np.float32)  # (U,3)
            u2u_links = build_u2u_links_from_positions(
                P_t=P_t,
                u_ids=u_ids,
                rng=rng,
                rmax=u2u_rmax,
                k=u2u_k,
                noise=u2u_noise,
                max_range=u2u_range,
                directed=u2u_directed
            )
            slot_obj["u2u"] = u2u_links

        slots.append(slot_obj)

    meta = {
        "name": "nwpu_env0",
        "flight": flight,
        "channel": f"ch{ch}",
        "channels_note": "pos/dist reali; bitrate U2G sintetico; GS feat sintetiche (seed riproducibile)",
        "units": {"pos": "m", "w1": "m", "w2": "arb_bitrate", "u2u_d": "m", "u2u_r": "arb_bitrate"},
        "note": "w1 in metri (da dataset); w2 ~ f(distanza) con clamp [1,320]",
        "has_u2u": bool(include_u2u),
        "u2u_model": {
            "enabled": bool(include_u2u),
            "directed": bool(u2u_directed),
            "range_m": None if u2u_range is None else float(u2u_range),
            "rate": {"type": "affine", "rmax": float(u2u_rmax), "k": float(u2u_k), "noise_sigma": float(u2u_noise)},
            "formula": "r = clamp(1, rmax - k * d + N(0, noise_sigma))"
        }
    }
    return {"meta": meta, "slots": slots}


# --------------------- CLI ---------------------
def main():
    ap = argparse.ArgumentParser(description="NWPU Env0 → Scenario JSON (multi-channel, opzionale U2U)")
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

    # ---- Opzioni U2U ----
    ap.add_argument("--u2u", action="store_true", help="includi link U2U negli slot")
    ap.add_argument("--u2u_max", type=float, default=160.0, help="Rmax U2U (clamp superiore)")
    ap.add_argument("--u2u_k", type=float, default=35.0, help="pendenza (bitrate cala con distanza)")
    ap.add_argument("--u2u_noise", type=float, default=2.0, help="sigma rumore sul rate U2U")
    ap.add_argument("--u2u_range", type=float, default=None, help="raggio massimo (m) per creare un link U2U")
    ap.add_argument("--u2u_undirected", action="store_true", help="salva un solo link per coppia (u<v)")

    # ---- Budget relay UAV (per abilitare i path 2-hop) ----
    ap.add_argument("--r1_mean", type=float, default=160.0, help="media budget relay per UAV")
    ap.add_argument("--r1_std", type=float, default=40.0, help="deviazione std budget relay per UAV")

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

    # Parametri U2U condivisi
    include_u2u = bool(args.u2u)
    u2u_directed = not args.u2u_undirected

    # BUILD in funzione della modalità
    if args.merge == "per_ch":
        # un file per canale: out funge da prefisso, aggiungo _chN.json
        for ch in ch_list:
            P, D = load_pos_dist(root, flight, ch)
            scn = build_from_PD(P, D, flight, ch, seed, A,
                                include_u2u=include_u2u,
                                u2u_rmax=args.u2u_max,
                                u2u_k=args.u2u_k,
                                u2u_noise=args.u2u_noise,
                                u2u_range=args.u2u_range,
                                u2u_directed=u2u_directed,
                                r1_mean=args.r1_mean,
                                r1_std=args.r1_std)
            if args.include_anchors:
                scn["meta"]["anchors"] = A.tolist()
            out_path = args.out.with_name(f"{args.out.stem}_ch{ch}{args.out.suffix}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(scn, ensure_ascii=False, indent=indent))
            u2u_per_slot = len(scn["slots"][0].get("u2u", []))
            print(f"OK → {out_path}  (slots={len(scn['slots'])}, "
                  f"uav/slot={len(scn['slots'][0]['uav'])}, "
                  f"gs/slot={len(scn['slots'][0]['gs'])}, "
                  f"u2g_links/slot={len(scn['slots'][0]['links'])}, "
                  f"u2u_links/slot={u2u_per_slot})")

    elif args.merge == "concat":
        # concatena gli slot: T = 30 * #ch
        big_slots = []
        slot_channels = []
        for ch in ch_list:
            P, D = load_pos_dist(root, flight, ch)
            scn = build_from_PD(P, D, flight, ch, seed + ch, A,
                                include_u2u=include_u2u,
                                u2u_rmax=args.u2u_max,
                                u2u_k=args.u2u_k,
                                u2u_noise=args.u2u_noise,
                                u2u_range=args.u2u_range,
                                u2u_directed=u2u_directed,
                                r1_mean=args.r1_mean,
                                r1_std=args.r1_std)
            for s in scn["slots"]:
                s["t"] = len(big_slots)  # indice progressivo
                big_slots.append(s)
                slot_channels.append(f"ch{ch}")
        meta = {
            "name": "nwpu_env0_concat",
            "flight": flight,
            "channels": [f"ch{c}" for c in ch_list],
            "slot_channel": slot_channels,
            "units": {"pos": "m", "w1": "m", "w2": "arb_bitrate", "u2u_d": "m", "u2u_r": "arb_bitrate"},
            "note": "Concatenazione degli slot dei canali; pos/dist reali, bitrate sintetico",
            "has_u2u": include_u2u
        }
        obj = {"meta": meta, "slots": big_slots}
        if args.include_anchors:
            obj["meta"]["anchors"] = A.tolist()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(obj, ensure_ascii=False, indent=indent))
        u2u_per_slot = len(big_slots[0].get("u2u", []))
        print(f"OK → {args.out}  (slots={len(big_slots)}, "
              f"uav/slot={len(big_slots[0]['uav'])}, "
              f"gs/slot={len(big_slots[0]['gs'])}, "
              f"u2g_links/slot={len(big_slots[0]['links'])}, "
              f"u2u_links/slot={u2u_per_slot})")

    else:  # average
        # media per-t su canali: P e D mediati
        P_list, D_list = [], []
        for ch in ch_list:
            P, D = load_pos_dist(root, flight, ch)
            P_list.append(P); D_list.append(D)
        P_avg = np.mean(np.stack(P_list, axis=0), axis=0).astype(np.float32)
        D_avg = np.mean(np.stack(D_list, axis=0), axis=0).astype(np.float32)
        scn = build_from_PD(P_avg, D_avg, flight, 0, seed, A,
                            include_u2u=include_u2u,
                            u2u_rmax=args.u2u_max,
                            u2u_k=args.u2u_k,
                            u2u_noise=args.u2u_noise,
                            u2u_range=args.u2u_range,
                            u2u_directed=u2u_directed,
                            r1_mean=args.r1_mean,
                            r1_std=args.r1_std)
        scn["meta"]["channel"] = "avg(" + ",".join(f"ch{c}" for c in ch_list) + ")"
        if args.include_anchors:
            scn["meta"]["anchors"] = A.tolist()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(scn, ensure_ascii=False, indent=indent))
        u2u_per_slot = len(scn["slots"][0].get("u2u", []))
        print(f"OK → {args.out}  (slots={len(scn['slots'])}, "
              f"uav/slot={len(scn['slots'][0]['uav'])}, "
              f"gs/slot={len(scn['slots'][0]['gs'])}, "
              f"u2g_links/slot={len(scn['slots'][0]['links'])}, "
              f"u2u_links/slot={u2u_per_slot})")


if __name__ == "__main__":
    main()
