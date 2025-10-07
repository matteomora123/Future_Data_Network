#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_env1_nwpu.py — costruisce scenario JSON (Environment1: Flying_disperse o Flying_circle)
"""

import argparse, json
from pathlib import Path
import numpy as np

# --------------------- util ---------------------
def load_mat(fp: Path) -> dict:
    """Carica .mat v5/v7.3 → dict numpy-ready"""
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
                if a.ndim == 2 and 1 < a.shape[0] < 64 and 1 < a.shape[1] < 64:
                    return a.T
                return a
            for k in f.keys():
                out[k] = to_np(f[k])
        return out


def find_array_3d(d: dict, target_shape: tuple) -> np.ndarray:
    for v in d.values():
        if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape == target_shape:
            return v
    for v in d.values():
        if isinstance(v, np.ndarray) and v.ndim == 3:
            try:
                return np.reshape(v, target_shape)
            except Exception:
                pass
    raise RuntimeError("array 3D non trovato")


# --------------------- caricamento posizioni ---------------------
def load_pos_dist(root: Path, flight: str, ch: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Carica posizioni UAV e distanze U2G (se disponibili).
    Per Environment1 (disperse/circle) genera D_m dummy.
    """
    # -------- Environment 1 --------
    if "disperse" in flight.lower() or "circle" in flight.lower():
        base = root / "Environment1" / "Flying_path" / flight
        files = sorted(base.glob("Position_UAV*.mat"))
        if not files:
            raise FileNotFoundError(f"Nessun file Position_UAV*.mat in {base}")

        P_list = []
        for fp in files:
            d = load_mat(fp)
            arr = None
            for v in d.values():
                if isinstance(v, np.ndarray) and v.ndim == 2 and 3 in v.shape:
                    arr = v if v.shape[1] == 3 else v.T
                    break
            if arr is None:
                raise RuntimeError(f"{fp.name}: matrice posizioni non trovata")
            P_list.append(arr.T)  # (3,T)
        P = np.stack(P_list, axis=0).astype(np.float32)
        G = 8
        D_m = np.zeros((G, P.shape[0], P.shape[2]), dtype=np.float32)
        return P, D_m

    # -------- Environment 0 --------
    base = root / "Raw_data" / "Environment0" / flight
    pos_fp = base / "Position_Label" / f"Position_label_ch{ch}.mat"
    if not pos_fp.exists():
        pos_fp = base / "Position_Label" / f"position_label_ch{ch}.mat"
    P = find_array_3d(load_mat(pos_fp), (7, 3, 30)).astype(np.float32)
    dis_dir = base / "Distance_Anchor_Label"
    cand = dis_dir / f"Distance_anchor_label_ch{ch}.mat"
    if not cand.exists():
        cand = dis_dir / f"Dis_anchor_label_ch{ch}.mat"
    D_m = find_array_3d(load_mat(cand), (8, 7, 30)).astype(np.float32)
    return P, D_m


# --------------------- U2U helper ---------------------
def build_u2u_links_from_positions(P_t, u_ids, rng, rmax, k, noise, max_range, directed):
    U = P_t.shape[0]
    links = []
    for i in range(U):
        for j in range(i + 1, U):
            d = float(np.linalg.norm(P_t[i] - P_t[j]))
            if (max_range is not None and d > max_range) or d < 0.01:
                continue
            r_raw = rmax - k * d + rng.randn() * noise
            r = float(np.clip(r_raw, 5.0, rmax))
            links.append({"a_id": u_ids[i], "b_id": u_ids[j], "d": d, "r": r})
            if directed:
                links.append({"a_id": u_ids[j], "b_id": u_ids[i], "d": d, "r": r})
    return links


# --------------------- builder ---------------------
def build_from_PD(P, D_m, flight, ch, seed, A,
                  include_u2u=False, u2u_rmax=160.0, u2u_k=35.0,
                  u2u_noise=2.0, u2u_range=None, u2u_directed=True,
                  r1_mean=160.0, r1_std=40.0):
    rng = np.random.RandomState(seed)
    T, U, G = P.shape[2], P.shape[0], 8
    gs_ids = [f"g{i}" for i in range(G)]
    u_ids = [f"u{i}" for i in range(U)]

    gs_feat = [(float(max(30.0, rng.normal(120.0, 30.0))),
                float(max(5.0, rng.normal(20.0, 8.0))),
                float(max(0.0, rng.normal(20.0, 10.0)))) for _ in range(G)]

    slots = []
    for t in range(T):
        uav = []
        for u in range(U):
            x, y, z = map(float, P[u, :, t])
            speed = 0.0 if t == 0 else float(np.linalg.norm(P[u, :, t] - P[u, :, t - 1]))
            f1 = float(max(0.0, rng.normal(150.0, 60.0)))
            f2 = float(max(1.0, rng.uniform(1.0, 6.0)))
            f3 = speed
            r1 = float(max(0.0, rng.normal(r1_mean, r1_std)))
            uav.append({"id": u_ids[u], "f1": f1, "f2": f2, "f3": f3,
                        "r1": r1, "x": x, "y": y, "z": z})

        gs = [{"id": gs_ids[g], "c1": gs_feat[g][0], "c2": gs_feat[g][1],
               "c3": gs_feat[g][2], "x": float(A[g, 0]), "y": float(A[g, 1]), "z": float(A[g, 2])}
              for g in range(G)]

        links = []
        for u in range(U):
            for g in range(G):
                w1 = float(D_m[g, u, t]) if np.any(D_m) else float(np.linalg.norm(P[u, :, t] - A[g]))
                w2_raw = 300.0 / (1.0 + 3.0 * w1) + rng.randn() * 8.0
                w2 = float(np.clip(w2_raw, 1.0, 320.0))
                links.append({"uav_id": u_ids[u], "gs_id": gs_ids[g], "w1": w1, "w2": w2})

        slot = {"t": t, "uav": uav, "gs": gs, "links": links}

        if include_u2u:
            u2u_links = build_u2u_links_from_positions(P[:, :, t], u_ids, rng,
                                                       u2u_rmax, u2u_k, u2u_noise,
                                                       u2u_range, u2u_directed)
            slot["u2u"] = u2u_links
        slots.append(slot)

    meta = {
        "name": "nwpu_env1" if "disperse" in flight.lower() else "nwpu_env0",
        "flight": flight,
        "channel": f"ch{ch}",
        "note": "Dataset NWPU – Environment1 con GS sintetiche e link U2U opzionali"
    }
    return {"meta": meta, "slots": slots}


# --------------------- main ---------------------
def main():
    ap = argparse.ArgumentParser(description="NWPU Environment1 → Scenario JSON (multi-channel, opzionale U2U)")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--flight", required=True)
    ap.add_argument("--channel", type=int, default=None)
    ap.add_argument("--channels", type=str, default=None)
    ap.add_argument("--merge", type=str, default="concat", choices=["per_ch", "concat", "average"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pretty", action="store_true")
    ap.add_argument("--u2u", action="store_true")
    ap.add_argument("--u2u_max", type=float, default=160.0)
    ap.add_argument("--u2u_k", type=float, default=35.0)
    ap.add_argument("--u2u_noise", type=float, default=2.0)
    ap.add_argument("--u2u_range", type=float, default=None)
    ap.add_argument("--u2u_undirected", action="store_true")
    ap.add_argument("--r1_mean", type=float, default=160.0)
    ap.add_argument("--r1_std", type=float, default=40.0)
    args = ap.parse_args()

    root, flight = args.root, args.flight
    seed = args.seed
    include_u2u = args.u2u
    u2u_directed = not args.u2u_undirected
    indent = 2 if args.pretty else None

    # Anchors (GS)
    if "disperse" in flight.lower() or "circle" in flight.lower():
        A = np.array([
            [0, 0, 0],
            [18, 0, 0],
            [0, 15, 0],
            [18, 15, 0],
            [0, 0, 8],
            [18, 0, 8],
            [0, 15, 8],
            [18, 15, 8],
        ], dtype=np.float32)
    else:
        anchors_mat = load_mat(root / "Raw_data" / "Environment0" / flight / "Anchors.mat")
        A = anchors_mat.get("Anchors", np.zeros((8, 3), dtype=np.float32)).astype(np.float32) / 100.0

    # Canali
    if args.channels:
        ch_list = [int(x.strip()) for x in args.channels.split(",") if x.strip()]
    elif args.channel is not None:
        ch_list = [args.channel]
    else:
        raise SystemExit("Specificare --channel N oppure --channels 2,3,...")

    # Merge concat
    big_slots = []
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
            s["t"] = len(big_slots)
            big_slots.append(s)

    obj = {
        "meta": {
            "name": "nwpu_env1_concat",
            "flight": flight,
            "channels": [f"ch{c}" for c in ch_list],
            "units": {"pos": "m", "w1": "m", "w2": "arb_bitrate"},
            "note": "Environment1 indoor 18×15×8 m",
            "has_u2u": include_u2u
        },
        "slots": big_slots
    }
    obj["meta"]["anchors"] = A.tolist()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(obj, ensure_ascii=False, indent=indent))
    print(f"✅ OK → {args.out}  (slots={len(big_slots)}, "
          f"uav/slot={len(big_slots[0]['uav'])}, gs/slot={len(big_slots[0]['gs'])})")


if __name__ == "__main__":
    main()
