#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Future Network – Data Communications (Huawei Tech Arena 2025)
ESTESO: relay U2U 1-hop come add-on (compatibile con l'output UAV→GS).
- U2U opzionale in JSON (altrimenti disattivo).
- Augment U2G con cammino u->v->g (fattore λ) e safety su budget di relay.
- EdgeNet/teacher/safety/env lavorano con feature "augmentate" quando disponibili.
Dipendenze: Python 3.9+, numpy, torch.
================================================================================
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import argparse, json, math, random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Velocità extra su GPU Ampere+
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# =======================
# = CAPITOLO 1: I/O JSON =
# =======================

@dataclass
class UAV:
    id: str
    f1: float
    f2: float
    f3: float
    r1: float = 0.0  # capacità/budget per fungere da relay U2U (opzionale)
    x: float = 0.0   # coordinate fisiche opzionali
    y: float = 0.0
    z: float = 0.0

@dataclass
class GS:
    id: str
    c1: float
    c2: float
    c3: float
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

@dataclass
class Link:
    uav_id: str
    gs_id: str
    w1: float
    w2: float

# arco U2U opzionale (diretto fra UAV)
@dataclass
class U2ULink:
    a_id: str
    b_id: str
    d: float   # distanza u2u
    r: float   # bitrate u2u

@dataclass
class Slot:
    t: int
    uav: List[UAV]
    gs: List[GS]
    links: List[Link]
    u2u: Optional[List[U2ULink]] = None  # opzionale

@dataclass
class Scenario:
    meta: Dict[str, str]
    slots: List[Slot]

def _coerce_u2u_list(raw_list) -> List[U2ULink]:
    """Accetta sia record {a_id,b_id,d,r} che {src_id,dst_id,d,r} (retrocompat)."""
    out = []
    for x in raw_list:
        if "a_id" in x and "b_id" in x:
            out.append(U2ULink(a_id=x["a_id"], b_id=x["b_id"], d=float(x["d"]), r=float(x["r"])))
        elif "src_id" in x and "dst_id" in x:
            out.append(U2ULink(a_id=x["src_id"], b_id=x["dst_id"], d=float(x["d"]), r=float(x["r"])))
        else:
            # ignora record non riconosciuti
            continue
    return out

def load_scenario(path: Path) -> Scenario:
    obj = json.loads(path.read_text())
    slots: List[Slot] = []
    for s in obj["slots"]:
        # UAV: supporta assenza di r1 (usa default dataclass=0.0)
        u = [UAV(**x) if "r1" in x else UAV(id=x["id"], f1=x["f1"], f2=x["f2"], f3=x["f3"])
             for x in s["uav"]]
        g = [GS(**x) for x in s["gs"]]
        l = [Link(**x) for x in s["links"]]
        # parse opzionale u2u (chiave "u2u" o retrocompat "u2u_links")
        u2u = None
        if "u2u" in s and s["u2u"] is not None:
            u2u = _coerce_u2u_list(s["u2u"])
        elif "u2u_links" in s and s["u2u_links"] is not None:
            u2u = _coerce_u2u_list(s["u2u_links"])
        slots.append(Slot(t=s["t"], uav=u, gs=g, links=l, u2u=u2u))
    return Scenario(meta=obj.get("meta", {}), slots=slots)

def save_submission(meta: Dict[str, str], assignments: List[Dict[str, str]], out: Path):
    out.write_text(json.dumps({
        "meta": meta,
        "assignments": [{"t": t, "map": m} for t, m in enumerate(assignments)]
    }, ensure_ascii=False, indent=2))


# ================================================
# = CAPITOLO 2: Generatore sintetico + valid util =
# ================================================

def gen_synth(T=32, U=12, G=5, seed=7) -> Scenario:
    random.seed(seed)
    slots: List[Slot] = []
    rs_u = np.random.RandomState(seed)
    rs_g = np.random.RandomState(seed+1)
    u_pos = rs_u.rand(U, 2) * 1000.0
    g_pos = rs_g.rand(G, 2) * 1000.0
    for t in range(T):
        uav = []
        for u in range(U):
            u_pos[u] += np.random.randn(2) * 15.0
            backlog = max(0.0, np.random.normal(150.0, 60.0))
            deadline = max(1.0, np.random.uniform(1.0, 6.0))
            speed = float(np.linalg.norm(np.random.randn(2) * 5.0))
            relay_budget = max(0.0, np.random.normal(160.0, 40.0))  # budget relay sintetico
            uav.append(UAV(id=f"u{u}", f1=backlog, f2=deadline, f3=speed, r1=relay_budget))
        gs = []
        for g in range(G):
            cap = max(30.0, np.random.normal(120.0, 30.0))
            lat = max(5.0, np.random.normal(20.0, 8.0))
            q   = max(0.0, np.random.normal(20.0, 10.0))
            gs.append(GS(id=f"g{g}", c1=cap, c2=lat, c3=q))
        links = []
        for u in range(U):
            for gg in range(G):
                dist = float(np.linalg.norm(u_pos[u] - g_pos[gg]) + 1.0)
                rate = max(1.0, 320.0 - 0.25 * dist + np.random.randn()*3.0)
                links.append(Link(uav_id=f"u{u}", gs_id=f"g{gg}", w1=dist, w2=rate))
        # U2U sintetico (con raggio ragionevole)
        u2u = []
        for i in range(U):
            for j in range(i+1, U):
                d = float(np.linalg.norm(u_pos[i] - u_pos[j]) + 1.0)
                r = max(1.0, 240.0 - 0.20 * d + np.random.randn()*3.0)
                if d < 20.0 and r > 5.0:
                    u2u.append(U2ULink(a_id=f"u{i}", b_id=f"u{j}", d=d, r=r))
                    u2u.append(U2ULink(a_id=f"u{j}", b_id=f"u{i}", d=d, r=r))
        slots.append(Slot(t=t, uav=uav, gs=gs, links=links, u2u=u2u))
    return Scenario(meta={"name": "synth"}, slots=slots)

def validate_slot(slot: Slot) -> Optional[str]:
    uids = {u.id for u in slot.uav}
    gids = {g.id for g in slot.gs}
    for lk in slot.links:
        if lk.uav_id not in uids:
            return f"link con uav_id inesistente: {lk.uav_id} @t={slot.t}"
        if lk.gs_id not in gids:
            return f"link con gs_id inesistente: {lk.gs_id} @t={slot.t}"
    if slot.u2u:
        for e in slot.u2u:
            if e.a_id not in uids or e.b_id not in uids:
                return f"u2u con id inesistente: {e.a_id}->{e.b_id} @t={slot.t}"
    return None


# =====================================
# = CAPITOLO 3: Ambiente / Reward     =
# =====================================

class DataComEnv:
    def __init__(self, alpha=1.2, beta=0.015, gamma=0.02, delta=0.0, penalty=30.0,
                 relay_latency_extra=5.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.penalty = penalty
        self.prev_choice: Dict[str, str] = {}
        self.relay_latency_extra = relay_latency_extra  # costo extra se 2 hop

    # aggiunto: aug_info e relay_plan per distinguere diretto/relay
    def step(self, slot: Slot, mapping: Dict[str, str],
             aug_info=None, relay_plan=None, cap_scale: float = 30.0) -> Tuple[float, Dict[str, float]]:
        """
        Step ambientale con supporto a relay U2U.
        Ogni UAV deve terminare su una GS. Se sceglie un relay, il percorso U2U→U2G
        è valutato solo se il costo complessivo è minore del collegamento diretto.
        """

        # --- Mappe di supporto ---
        gs_capacity = {g.id: max(1.0, g.c1) for g in slot.gs}
        link_map = {(lk.uav_id, lk.gs_id): lk for lk in slot.links}
        u2u_map = {}
        if hasattr(slot, "u2u") and slot.u2u:
            for lk in slot.u2u:
                a, b = lk.a_id, lk.b_id
                u2u_map[(a, b)] = {"d": lk.d, "r": lk.r}

        # --- Metriche cumulative ---
        throughput = 0.0
        latency = 0.0
        dist_sum = 0.0
        handovers = 0.0
        violations = 0.0

        for u in slot.uav:
            dest = mapping.get(u.id, "NO_TX")

            # 1️⃣ Collegamento diretto U2G
            if dest.startswith("g"):
                lk = link_map.get((u.id, dest))
                if lk is None:
                    violations += 1.0
                    continue
                cons = lk.w2 / 30.0
                if gs_capacity[dest] - cons < -1e-6:
                    violations += 1.0
                    continue
                gs_capacity[dest] -= cons
                throughput += min(u.f1, lk.w2) * (1.0 + np.random.randn() * 0.05)
                latency += max(1.0, lk.w1 * 0.05)
                dist_sum += lk.w1

            # 2️⃣ Collegamento via relay U2U → U2G
            elif dest.startswith("u"):
                lk1 = u2u_map.get((u.id, dest))
                relay_dest = mapping.get(dest, "NO_TX")

                # valido solo se il relay ha GS finale
                if lk1 and relay_dest.startswith("g"):
                    lk2 = link_map.get((dest, relay_dest))
                    if lk2 is None:
                        violations += 1.0
                        continue

                    # calcola costi
                    cost_relay = lk1["d"] + lk2.w1
                    direct_links = [lk for lk in slot.links if lk.uav_id == u.id]
                    cost_direct = min((lk.w1 for lk in direct_links), default=1e9)

                    # il relay è accettato se è entro il 20% del costo diretto
                    if cost_relay * 0.8 < cost_direct:
                        cons = (lk1["r"] + lk2.w2) / 60.0
                        if gs_capacity[relay_dest] - cons < -1e-6:
                            violations += 1.0
                            continue
                        gs_capacity[relay_dest] -= cons
                        throughput += min(u.f1, lk2.w2 * 0.8) * (1.0 + np.random.randn() * 0.05)
                        latency += max(1.0, cost_relay * 0.05)
                        dist_sum += cost_relay
                    else:
                        # relay inutile
                        violations += 0.5
                        latency += 10.0

                else:
                    # relay orfano (non termina su GS)
                    violations += 1.0
                    latency += 15.0

            # 3️⃣ Nessuna trasmissione
            else:
                latency += 5.0

            # 4️⃣ Handover (rispetto slot precedente)
            if self.prev_choice.get(u.id, dest) != dest and u.id in self.prev_choice:
                handovers += 1.0

        # --- Reward aggregato ---
        reward = (self.alpha * throughput
                  - self.beta * latency
                  - self.gamma * dist_sum
                  - self.delta * handovers
                  - self.penalty * violations)

        # aggiorna stato
        self.prev_choice = {u.id: mapping.get(u.id, "NO_TX") for u in slot.uav}

        info = dict(throughput=throughput, latency=latency, distance=dist_sum,
                    handover=handovers, violations=violations, reward=reward)
        return reward, info

# =========================================
# = CAPITOLO 4: Baseline greedy + Isteresi =
# =========================================

def _edge_w12(lk: Link, overrides: Optional[Dict[Tuple[str,str], Tuple[float,float]]]):
    if overrides is None:
        return lk.w1, lk.w2
    key = (lk.uav_id, lk.gs_id)
    if key in overrides:
        w1, w2 = overrides[key]
        return float(w1), float(w2)
    return lk.w1, lk.w2

def greedy_policy(slot: Slot, cap_scale: float = 30.0,
                  overrides: Optional[Dict[Tuple[str,str], Tuple[float,float]]] = None) -> Dict[str, str]:
    links_by_u: Dict[str, List[Link]] = {}
    for lk in slot.links:
        links_by_u.setdefault(lk.uav_id, []).append(lk)
    gs_left = {g.id: max(1.0, g.c1)/cap_scale for g in slot.gs}
    decision: Dict[str, str] = {}
    for u in slot.uav:
        cands = links_by_u.get(u.id, [])
        if not cands:
            decision[u.id] = "NO_TX"; continue
        # ordina per efficienza (w2/(w1+1)) usando override se disponibile
        cands.sort(key=lambda e: (_edge_w12(e, overrides)[1] / (_edge_w12(e, overrides)[0] + 1.0)), reverse=True)
        chosen = "NO_TX"
        for e in cands:
            w1, w2 = _edge_w12(e, overrides)
            need = w2 / cap_scale
            if gs_left.get(e.gs_id, 0.0) >= need:
                gs_left[e.gs_id] -= need
                chosen = e.gs_id
                break
        decision[u.id] = chosen
    return decision

def greedy_with_hysteresis(slot: Slot, prev_choice: Dict[str, str],
                           cap_scale: float = 30.0, lambda_h: float = 0.15,
                           overrides: Optional[Dict[Tuple[str,str], Tuple[float,float]]] = None) -> Dict[str, str]:
    """
    Capacity-aware + bonus di inerzia se si resta sulla stessa GS dello slot precedente.
    """
    links_by_u: Dict[str, List[Link]] = {}
    for lk in slot.links:
        links_by_u.setdefault(lk.uav_id, []).append(lk)
    gs_left = {g.id: max(1.0, g.c1)/cap_scale for g in slot.gs}
    decision: Dict[str, str] = {}
    for u in slot.uav:
        cands = links_by_u.get(u.id, [])
        if not cands:
            decision[u.id] = "NO_TX"; continue
        scores = []
        for e in cands:
            w1, w2 = _edge_w12(e, overrides)
            base = w2 / (w1 + 1.0)
            if prev_choice.get(u.id, None) == e.gs_id:
                base += lambda_h * base
            scores.append((e, base))
        scores.sort(key=lambda x: x[1], reverse=True)
        chosen = "NO_TX"
        for e, _sc in scores:
            _w1, w2 = _edge_w12(e, overrides)
            need = w2 / cap_scale
            if gs_left.get(e.gs_id, 0.0) >= need:
                gs_left[e.gs_id] -= need
                chosen = e.gs_id
                break
        decision[u.id] = chosen
    return decision


# =====================================================
# = CAPITOLO 4bis: Augment U2G con relay U2U (1-hop)  =
# =====================================================

AugInfo = Dict[str, float]  # alias

def _build_u2u_index(slot: Slot) -> Dict[str, List[Tuple[str, float, float]]]:
    """Ritorna adiacenza U2U: u_id -> [(v_id, d_uv, r_uv), ...]"""
    idx: Dict[str, List[Tuple[str,float,float]]] = {}
    if not slot.u2u:
        return idx
    for e in slot.u2u:
        idx.setdefault(e.a_id, []).append((e.b_id, float(e.d), float(e.r)))
    return idx

def compute_augmented_edges(slot: Slot, relay_factor: float = 0.9
                            ) -> Tuple[Dict[Tuple[str,str], AugInfo],
                                       Dict[Tuple[str,str], Tuple[float,float]]]:
    """
    Per ogni (u,g) valuta migliore opzione: diretto o via un vicino v.
    Ritorna:
    - aug_info[(u,g)] con dettagli (diretto/effettivo/relay/id).
    - overrides[(u,g)] = (w1_eff, w2_eff) da dare alla GNN/safety.
    """
    link_map = {(lk.uav_id, lk.gs_id): lk for lk in slot.links}
    u2u_idx = _build_u2u_index(slot)

    aug_info: Dict[Tuple[str,str], AugInfo] = {}
    overrides: Dict[Tuple[str,str], Tuple[float,float]] = {}

    for (u_id, g_id), lk in link_map.items():
        best_rate = float(lk.w2)
        best_dist = float(lk.w1)
        best_v: Optional[str] = None
        best_need = 0.0

        # prova tutti i vicini v di u
        for (v_id, d_uv, r_uv) in u2u_idx.get(u_id, []):
            lk_vg = link_map.get((v_id, g_id))
            if lk_vg is None:
                continue
            # rate via relay = λ * min(r_uv, w2(v,g))
            via_rate = relay_factor * min(float(r_uv), float(lk_vg.w2))
            if via_rate > best_rate * 0.9:
                best_rate = via_rate
                best_dist = float(d_uv) + float(lk_vg.w1)
                best_v = v_id
                best_need = via_rate  # consumo "lordo" da far pagare al relay

        ai: AugInfo = {
            "w1_direct": float(lk.w1),
            "w2_direct": float(lk.w2),
            "w1_eff": best_dist,
            "w2_eff": best_rate,
            "relay": best_v if best_v is not None else "",
            "relay_need": float(best_need),
            "direct_ok": 1.0
        }
        aug_info[(u_id, g_id)] = ai
        overrides[(u_id, g_id)] = (ai["w1_eff"], ai["w2_eff"])

    return aug_info, overrides


# =========================================
# = CAPITOLO 5: GNN / EdgeNet / PPOPolicy =
# =========================================

def pack_features(slot: Slot,
                  overrides: Optional[Dict[Tuple[str,str], Tuple[float,float]]] = None
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                             Dict[str,int], Dict[str,int], List[Tuple[int,int]]]:
    u_ids = {u.id: i for i, u in enumerate(slot.uav)}
    g_ids = {g.id: i for i, g in enumerate(slot.gs)}
    u_feat = torch.tensor([[u.f1, u.f2, u.f3] for u in slot.uav], dtype=torch.float32)
    g_feat = torch.tensor([[g.c1, g.c2, g.c3] for g in slot.gs], dtype=torch.float32)

    # e_feat usa eventuali override (w1,w2 effettivi)
    e_rows = []
    edge_index = []
    for lk in slot.links:
        w1, w2 = _edge_w12(lk, overrides)
        e_rows.append([w1 / 20.0, w2 / 320.0])  # normalizzati (campo 20m, bitrate max ~320)
        edge_index.append((u_ids[lk.uav_id], g_ids[lk.gs_id]))
    e_feat = torch.tensor(e_rows, dtype=torch.float32)

    return u_feat, g_feat, e_feat, u_ids, g_ids, edge_index

class EdgeNet(nn.Module):
    def __init__(self, dU=3, dG=3, dE=2, dh=128):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(dU + dG + dE, dh), nn.ReLU(),
            nn.Linear(dh, dh), nn.ReLU(),
            nn.Linear(dh, 1)
        )
        self.notx_mlp = nn.Sequential(
            nn.Linear(dU, dh//2), nn.ReLU(),
            nn.Linear(dh//2, 1)
        )

    def forward(self, slot: Slot,
                overrides: Optional[Dict[Tuple[str,str], Tuple[float,float]]] = None
                ) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, torch.Tensor]]:
        device = next(self.parameters()).device
        u_feat, g_feat, e_feat, u_ids, g_ids, edge_index = pack_features(slot, overrides)
        u_feat = u_feat.to(device); g_feat = g_feat.to(device); e_feat = e_feat.to(device)

        if len(edge_index) == 0:
            return {uid: [("NO_TX", 1.0)] for uid in u_ids.keys()}, {}

        u_idx = torch.tensor([ui for ui, _ in edge_index], dtype=torch.long, device=device)
        g_idx = torch.tensor([gi for _, gi in edge_index], dtype=torch.long, device=device)

        x = torch.cat([u_feat[u_idx], g_feat[g_idx], e_feat], dim=-1)
        edge_logits = self.edge_mlp(x).squeeze(-1)

        per_u_pos: Dict[int, List[int]] = {}
        per_u_gids: Dict[int, List[int]] = {}
        for k, (ui, gi) in enumerate(edge_index):
            per_u_pos.setdefault(ui, []).append(k)
            per_u_gids.setdefault(ui, []).append(gi)

        logit_no = self.notx_mlp(u_feat).squeeze(-1)

        probs_dict: Dict[str, List[Tuple[str, float]]] = {}
        raw_logits_store: Dict[str, torch.Tensor] = {}

        rev_gid = {v:k for k,v in g_ids.items()}
        for uid, ui in u_ids.items():
            pos_list = per_u_pos.get(ui, [])
            gs_list = per_u_gids.get(ui, [])
            if pos_list:
                e_log = edge_logits[pos_list]
                logits = torch.cat([e_log, logit_no[ui:ui+1]], dim=0)
                labels = [rev_gid[gi] for gi in gs_list] + ["NO_TX"]
            else:
                logits = logit_no[ui:ui+1]
                labels = ["NO_TX"]
            raw_logits_store[uid] = logits
            p = torch.softmax(logits, dim=0).detach().cpu().tolist()
            probs_dict[uid] = list(zip(labels, p))
        return probs_dict, raw_logits_store

class PPOPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = EdgeNet()
        self.v_mlp = nn.Sequential(
            nn.Linear(3+3, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward_policy(self, slot: Slot,
                       overrides: Optional[Dict[Tuple[str,str], Tuple[float,float]]] = None):
        return self.net(slot, overrides)

    def forward_value(self, slot: Slot) -> torch.Tensor:
        device = next(self.parameters()).device
        u_feat, g_feat, _, _, _, _ = pack_features(slot, overrides=None)
        u_feat = u_feat.to(device); g_feat = g_feat.to(device)
        u_mean = u_feat.mean(dim=0)
        g_mean = g_feat.mean(dim=0)
        z = torch.cat([u_mean, g_mean], dim=-1)
        return self.v_mlp(z).squeeze(-1)


# ============================================================
# = CAPITOLO 6: Training sintetico (fallback, completo PPO)  =
# ============================================================

def ppo_train(env,
              episodes=200,
              gamma=0.98,
              lam=0.95,
              clip=0.2,
              lr=3e-4,
              curriculum=(8,3,16,6),
              il_steps=200,
              relay_factor=0.8,          # >>>
              cap_scale=30.0,            # >>>
              relay_scale=30.0,          # >>>
              device: torch.device = get_device(),
              seed=42) -> PPOPolicy:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    policy = PPOPolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    # IL su sintetico con teacher greedy (su link augmentati)
    for step in range(il_steps):
        scn = gen_synth(T=1, U=curriculum[0], G=curriculum[1], seed=seed+step)
        slot = scn.slots[0]
        aug, overrides = compute_augmented_edges(slot, relay_factor=relay_factor)
        target = greedy_policy(slot, cap_scale=cap_scale, overrides=overrides)
        probs, raw = policy.forward_policy(slot, overrides=overrides)
        loss = torch.tensor(0.0, device=device)
        for uid, choices in probs.items():
            labels = [lab for lab, _ in choices]
            y = target.get(uid, "NO_TX")
            if y not in labels: y = "NO_TX"
            yi = labels.index(y)
            logits = raw[uid]
            loss = loss + F.cross_entropy(logits.unsqueeze(0), torch.tensor([yi], device=device))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if (step+1) % 50 == 0:
            print(f"[IL-synth] {step+1}/{il_steps} loss={float(loss):.4f}")

    U0, G0, U1, G1 = curriculum
    for ep in range(episodes):
        frac = ep / max(1, episodes-1)
        U = int(U0 + (U1-U0) * frac)
        G = int(G0 + (G1-G0) * frac)
        scn = gen_synth(T=8, U=U, G=G, seed=seed+1000+ep)
        env.prev_choice = {}
        traj, values = [], []

        # Rollout
        for slot in scn.slots:
            aug, overrides = compute_augmented_edges(slot, relay_factor=relay_factor)
            probs, _ = policy.forward_policy(slot, overrides=overrides)
            mapping, relay_plan = safety_project_aug(slot, probs, aug,
                                                     cap_scale=cap_scale, relay_scale=relay_scale)
            # logp aggregata
            logp = 0.0
            for uid, choices in probs.items():
                d = dict(choices)
                chosen = mapping.get(uid, "NO_TX")
                p = max(1e-8, d.get(chosen, d.get("NO_TX", 1e-8)))
                logp += math.log(p)
            r, _info = env.step(slot, mapping, aug_info=aug, relay_plan=relay_plan,
                                cap_scale=cap_scale)
            v = policy.forward_value(slot)
            traj.append((slot, aug, overrides, mapping, relay_plan, logp, r))
            values.append(v)

        # GAE
        with torch.no_grad():
            vals = torch.stack(values + [torch.tensor(0.0, device=device)])
        rewards = torch.tensor([tr[6] for tr in traj], dtype=torch.float32, device=device)
        adv = torch.zeros_like(rewards, device=device)
        gae = 0.0
        for t in reversed(range(len(traj))):
            delta = rewards[t] + gamma * float(vals[t+1]) - float(vals[t])
            gae = delta + gamma * lam * gae
            adv[t] = gae
        returns = adv + vals[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Aggiornamento
        pol_loss = torch.tensor(0.0, device=device)
        val_loss = torch.tensor(0.0, device=device)
        ent_bonus = torch.tensor(0.0, device=device)
        for (slot, aug, overrides, mapping, relay_plan, old_logp, _r), A, R in zip(traj, adv, returns):
            probs, _ = policy.forward_policy(slot, overrides=overrides)
            new_logp = 0.0; ent = 0.0
            for _, choices in probs.items():
                d = dict(choices)
                ent -= sum(p*math.log(max(p,1e-8)) for p in d.values())
            for uid, choices in probs.items():
                d = dict(choices)
                chosen = mapping.get(uid, "NO_TX")
                p = max(1e-8, d.get(chosen, d.get("NO_TX", 1e-8)))
                new_logp += math.log(p)
            ratio = math.exp(new_logp - old_logp)
            surr1 = ratio * float(A)
            surr2 = max(min(ratio, 1.0+clip), 1.0-clip) * float(A)
            pol_loss += -min(surr1, surr2)
            v = policy.forward_value(slot)
            val_loss += F.mse_loss(v, R)
            ent_bonus += ent
        n = len(traj)
        loss = (pol_loss/n) + 0.5*(val_loss/n) - 0.001*(ent_bonus/n)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if (ep+1) % 10 == 0:
            avg_R = float(torch.mean(rewards).item())
            print(f"[PPO-synth] ep {ep+1}/{episodes}  U={U} G={G}  Ravg={avg_R:.2f}  loss={float(loss):.3f}")

    return policy


# ===================================================================
# = CAPITOLO 7: Training REALE (LOCO) completo con PPO operativo    =
# ===================================================================

def load_many(paths: List[str]) -> List[Scenario]:
    scns = []
    for p in paths or []:
        scns.append(load_scenario(Path(p)))
    return scns

def ppo_train_with_loco(env: DataComEnv,
                        train_scn: List[Scenario],
                        val_scn: List[Scenario],
                        episodes=60,
                        il_steps=150,
                        gamma=0.98,
                        lam=0.95,
                        clip=0.2,
                        lr=3e-4,
                        relay_factor=0.8,
                        cap_scale=30.0,
                        relay_scale=30.0,
                        device: torch.device = get_device(),
                        seed=123) -> PPOPolicy:
    """
    PPO su dati REALI:
    - IL warm-start contro teacher greedy+isteresi (capacity-aware) su link augmentati (se u2u presenti).
    - Rollout PPO scorrendo gli slot reali (continuità per l’handover).
    - Validazione LOCO: media reward su scenari di validazione; tiene best checkpoint.
    """
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    policy = PPOPolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    # ----------- IL su REALI ----------
    all_train_slots: List[Tuple[Slot, int]] = []
    for si, sc in enumerate(train_scn):
        for sl in sc.slots:
            all_train_slots.append((sl, si))
    prev_choice_teacher: Dict[int, Dict[str, str]] = {i: {} for i in range(len(train_scn))}

    for step in range(il_steps):
        slot, scen_idx = random.choice(all_train_slots)
        aug, overrides = compute_augmented_edges(slot, relay_factor=relay_factor) if slot.u2u else ({}, {})
        teacher_map = greedy_with_hysteresis(slot, prev_choice_teacher[scen_idx],
                                             cap_scale=cap_scale, overrides=overrides)
        prev_choice_teacher[scen_idx] = {u.id: teacher_map.get(u.id, "NO_TX") for u in slot.uav}

        probs, raw = policy.forward_policy(slot, overrides=overrides if overrides else None)
        loss = torch.tensor(0.0, device=device)
        for uid, choices in probs.items():
            labels = [lab for lab, _ in choices]
            y = teacher_map.get(uid, "NO_TX")
            if y not in labels: y = "NO_TX"
            yi = labels.index(y)
            logits = raw[uid]
            loss = loss + F.cross_entropy(logits.unsqueeze(0), torch.tensor([yi], device=logits.device))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if (step+1) % 50 == 0:
            print(f"[IL-real] {step+1}/{il_steps} loss={float(loss):.4f}")

    # ----------------------------- PPO su REALI ------------------------------
    best_val = float("-inf")
    best_state = None

    for ep in range(episodes):
        random.shuffle(train_scn)
        total_rew = 0.0
        total_cnt = 0

        for sc in train_scn:
            env.prev_choice = {}
            traj, values = [], []

            for slot in sc.slots:
                aug, overrides = compute_augmented_edges(slot, relay_factor=relay_factor) if slot.u2u else ({}, {})
                probs, _ = policy.forward_policy(slot, overrides=overrides if overrides else None)
                mapping, relay_plan = safety_project_aug(slot, probs, aug,
                                                         cap_scale=cap_scale, relay_scale=relay_scale)
                logp = 0.0
                for uid, choices in probs.items():
                    d = dict(choices)
                    chosen = mapping.get(uid, "NO_TX")
                    p = max(1e-8, d.get(chosen, d.get("NO_TX", 1e-8)))
                    logp += math.log(p)

                r, _info = env.step(slot, mapping, aug_info=aug if aug else None,
                                    relay_plan=relay_plan, cap_scale=cap_scale)
                v = policy.forward_value(slot)

                traj.append((slot, aug, overrides, mapping, relay_plan, logp, r))
                values.append(v)
                total_rew += r
                total_cnt += 1

            # ---- Update PPO per scenario ----
            with torch.no_grad():
                vals = torch.stack(values + [torch.tensor(0.0, device=device)])
            rewards = torch.tensor([tr[6] for tr in traj], dtype=torch.float32, device=device)
            adv = torch.zeros_like(rewards, device=device)
            gae = 0.0
            for t in reversed(range(len(traj))):
                delta = rewards[t] + gamma * float(vals[t+1]) - float(vals[t])
                gae = delta + gamma * lam * gae
                adv[t] = gae
            returns = adv + vals[:-1]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            pol_loss = torch.tensor(0.0, device=device)
            val_loss = torch.tensor(0.0, device=device)
            ent_bonus = torch.tensor(0.0, device=device)
            for (slot, aug, overrides, mapping, relay_plan, old_logp, _r), A, R in zip(traj, adv, returns):
                probs, _ = policy.forward_policy(slot, overrides=overrides if overrides else None)

                new_logp = 0.0; ent = 0.0
                for _, choices in probs.items():
                    d = dict(choices)
                    ent -= sum(p*math.log(max(p,1e-8)) for p in d.values())
                for uid, choices in probs.items():
                    d = dict(choices)
                    chosen = mapping.get(uid, "NO_TX")
                    p = max(1e-8, d.get(chosen, d.get("NO_TX", 1e-8)))
                    new_logp += math.log(p)

                ratio = math.exp(new_logp - old_logp)
                surr1 = ratio * float(A)
                surr2 = max(min(ratio, 1.0+clip), 1.0-clip) * float(A)
                pol_loss += -min(surr1, surr2)

                v = policy.forward_value(slot)
                val_loss += F.mse_loss(v, R)
                ent_bonus += ent

            n = len(traj)
            if n > 0:
                loss = (pol_loss/n) + 0.5*(val_loss/n) - 0.001*(ent_bonus/n)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()

        avg_train = total_rew / max(1, total_cnt)
        print(f"[PPO-real][train] ep {ep+1}/{episodes} avg_reward={avg_train:.3f}")

        # ------------------- Validazione LOCO (se presente) -------------------
        if val_scn:
            vtot = 0.0; vcnt = 0
            for scv in val_scn:
                env.prev_choice = {}
                for slot in scv.slots:
                    aug, overrides = compute_augmented_edges(slot, relay_factor=relay_factor) if slot.u2u else ({}, {})
                    probs, _ = policy.forward_policy(slot, overrides=overrides if overrides else None)
                    mapping, relay_plan = safety_project_aug(slot, probs, aug,
                                                             cap_scale=cap_scale, relay_scale=relay_scale)
                    r, _info = env.step(slot, mapping, aug_info=aug if aug else None,
                                        relay_plan=relay_plan, cap_scale=cap_scale)
                    vtot += r; vcnt += 1
            avg_val = vtot / max(1, vcnt)
            print(f"[PPO-real][VAL] ep {ep+1}/{episodes} avg_val_reward={avg_val:.3f}")
            if avg_val > best_val:
                best_val = avg_val
                best_state = {k: v.clone().detach().cpu() for k, v in policy.state_dict().items()}
            # --- Salva checkpoint ogni epoca ---
            ckpt_dir = Path(r"C:\Users\matte\PycharmProjects\Future_Data_Network\Checkpoint")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(policy.state_dict(), ckpt_dir / f"policy_epoch_{ep + 1:03d}.pt")

    if best_state is not None:
        policy.load_state_dict(best_state)
        print(f"[PPO-real] ripristinato best checkpoint (val reward={best_val:.3f})")
    return policy


# =================================================
# = CAPITOLO 6.5: Safety con budget relay U2U     =
# =================================================

def safety_project_aug(slot: Slot,
                       probs: Dict[str, List[Tuple[str, float]]],
                       aug_info: Dict[Tuple[str,str], AugInfo],
                       cap_scale: float = 30.0,
                       relay_scale: float = 30.0
                       ) -> Tuple[Dict[str, str], Dict[str, Dict]]:
    """
    Proietta la distribuzione delle scelte nel rispetto sia della capacità GS
    sia del budget di relay dei nodi. Restituisce:
    - mapping u->gs
    - relay_plan[u] = {"relay": v or "", "use_relay": bool}
    """
    gs_left = {g.id: max(1.0, g.c1)/cap_scale for g in slot.gs}
    relay_left = {u.id: max(0.0, getattr(u, "r1", 0.0))/relay_scale for u in slot.uav}
    decision: Dict[str, str] = {}
    relay_plan: Dict[str, Dict] = {}
    link_map = {(lk.uav_id, lk.gs_id): lk for lk in slot.links}

    for u in slot.uav:
        ranked = sorted(probs[u.id], key=lambda x: x[1], reverse=True)
        chosen_g = "NO_TX"
        use_relay = False
        chosen_relay = ""

        for gs_id, _p in ranked:
            if gs_id == "NO_TX":
                chosen_g = "NO_TX"; use_relay = False; chosen_relay = ""; break

            key = (u.id, gs_id)
            ai = aug_info.get(key)
            if ai is None:
                # fallback: usa diretto da link_map
                lk = link_map.get(key)
                if lk is None:
                    continue
                need_gs = lk.w2 / cap_scale
                if gs_left.get(gs_id, 0.0) >= need_gs:
                    gs_left[gs_id] -= need_gs
                    chosen_g = gs_id; use_relay = False; chosen_relay = ""
                    break
                else:
                    continue

            # prova prima relay (se esiste) altrimenti diretto
            relay_id = ai["relay"]
            w2_dir = ai["w2_direct"]; w2_eff = ai["w2_eff"]
            # path 1: relay
            if relay_id:
                need_gs = w2_eff / cap_scale
                need_relay = ai["relay_need"] / relay_scale
                if gs_left.get(gs_id, 0.0) >= need_gs and relay_left.get(relay_id, 0.0) >= need_relay:
                    gs_left[gs_id] -= need_gs
                    relay_left[relay_id] -= need_relay
                    chosen_g = gs_id; use_relay = True; chosen_relay = relay_id
                    break
            # path 2: diretto
            lk = link_map.get(key)
            if lk is None:
                continue
            need_gs = w2_dir / cap_scale
            if gs_left.get(gs_id, 0.0) >= need_gs:
                gs_left[gs_id] -= need_gs
                chosen_g = gs_id; use_relay = False; chosen_relay = ""
                break

        decision[u.id] = chosen_g
        relay_plan[u.id] = {"relay": chosen_relay, "use_relay": use_relay}

    return decision, relay_plan


# =================================================
# = CAPITOLO 8: Inferenza e produzione submission =
# =================================================

def run_inference(policy: PPOPolicy, scenario: Scenario, device: torch.device,
                  relay_factor: float = 0.8, cap_scale: float = 30.0, relay_scale: float = 30.0
                  ) -> List[Dict[str,str]]:
    policy.to(device)
    env = DataComEnv(gamma=0.02)
    assignments: List[Dict[str, str]] = []
    env.prev_choice = {}
    for slot in scenario.slots:
        err = validate_slot(slot)
        if err:
            raise ValueError(err)
        aug, overrides = compute_augmented_edges(slot, relay_factor=relay_factor) if slot.u2u else ({}, {})
        probs, _ = policy.forward_policy(slot, overrides=overrides if overrides else None)
        mapping, relay_plan = safety_project_aug(slot, probs, aug,
                                                 cap_scale=cap_scale, relay_scale=relay_scale)
        assignments.append(mapping)
        _r, _info = env.step(slot, mapping, aug_info=aug if aug else None,
                             relay_plan=relay_plan, cap_scale=cap_scale)
    return assignments


# ====================================
# = CAPITOLO 9: Entry-point /  CLI   =
# ====================================

def main():
    ap = argparse.ArgumentParser(description="Future Network – DataCom (esteso: LOCO, reali, hybrid, U2U relay)")
    ap.add_argument("--train", action="store_true", help="train (sintetico fallback o reali se specificati)")
    ap.add_argument("--eval", type=Path, help="valuta il modello su uno Scenario JSON (stampa metriche)")
    ap.add_argument("--per_slot", action="store_true", help="stampa le metriche per ogni slot durante --eval")
    ap.add_argument("--train_scenarios", type=str, nargs="*", default=None,
                    help="file JSON reali usati per training")
    ap.add_argument("--val_scenarios", type=str, nargs="*", default=None,
                    help="file JSON reali usati per validazione (LOCO)")
    ap.add_argument("--use_loco", action="store_true", help="(flag informativo) validazione LOCO quando val_scenarios è fornito")
    ap.add_argument("--episodes", type=int, default=100, help="episodi PPO")
    ap.add_argument("--il_steps", type=int, default=150, help="passi di imitation learning")
    ap.add_argument("--ckpt", type=Path, default=Path("policy.pt"), help="checkpoint .pt")
    ap.add_argument("--infer", type=Path, help="inferenzia su input Scenario JSON")
    ap.add_argument("--out", type=Path, help="salva submission JSON")
    # hyper per relay
    ap.add_argument("--relay_factor", type=float, default=0.8, help="fattore di efficienza per il relay u2u (λ)")
    ap.add_argument("--cap_scale", type=float, default=30.0, help="scala normalizzazione capacità GS")
    ap.add_argument("--relay_scale", type=float, default=30.0, help="scala normalizzazione budget relay UAV")
    args = ap.parse_args()

    device = get_device()
    print(f"[Device] Using {device}")

    if args.train:
        env = DataComEnv()
        if args.train_scenarios:
            # TRAIN su REALI + VAL LOCO (se fornita)
            train_scn = load_many(args.train_scenarios)
            val_scn   = load_many(args.val_scenarios) if args.val_scenarios else []
            policy = ppo_train_with_loco(env,
                                         train_scn=train_scn,
                                         val_scn=val_scn,
                                         episodes=args.episodes,
                                         il_steps=args.il_steps,
                                         relay_factor=args.relay_factor,
                                         cap_scale=args.cap_scale,
                                         relay_scale=args.relay_scale,
                                         device=device)
        else:
            # Fallback: training sintetico
            policy = ppo_train(env,
                               episodes=args.episodes,
                               il_steps=args.il_steps,
                               relay_factor=args.relay_factor,
                               cap_scale=args.cap_scale,
                               relay_scale=args.relay_scale,
                               device=device)
        torch.save(policy.state_dict(), args.ckpt)
        print(f"Checkpoint salvato -> {args.ckpt}")

    if args.infer:
        scn = load_scenario(args.infer)
        policy = PPOPolicy().to(device)
        if args.ckpt.exists():
            policy.load_state_dict(torch.load(args.ckpt, map_location=device))
        else:
            print("ATTENZIONE: nessun checkpoint trovato, policy random.")
        assignments = run_inference(policy, scn, device,
                                    relay_factor=args.relay_factor,
                                    cap_scale=args.cap_scale,
                                    relay_scale=args.relay_scale)
        out = args.out or Path("submission.json")
        save_submission({"solver": "gnn_ppo_ctde_loco_u2u"}, assignments, out)
        print(f"Submission scritta -> {out}")
    if args.eval:
        scn = load_scenario(args.eval)
        policy = PPOPolicy().to(device)
        if args.ckpt.exists():
            policy.load_state_dict(torch.load(args.ckpt, map_location=device))
        else:
            print("ATTENZIONE: nessun checkpoint trovato, policy random.")
        _ = evaluate_scenario(policy, scn, device=device, per_slot=args.per_slot,
                              relay_factor=args.relay_factor,
                              cap_scale=args.cap_scale,
                              relay_scale=args.relay_scale)

def evaluate_scenario(policy: PPOPolicy, scenario: Scenario, device: torch.device,
                      per_slot: bool = False, relay_factor: float = 0.8,
                      cap_scale: float = 30.0, relay_scale: float = 30.0
                      ) -> Dict[str, float]:
    """
    Valuta la policy sullo scenario con relay opzionale (U2U).
    """
    policy.to(device)
    env = DataComEnv()
    env.prev_choice = {}

    agg = {
        "throughput": 0.0,
        "latency": 0.0,
        "distance": 0.0,
        "handover": 0.0,
        "violations": 0.0,
        "reward": 0.0
    }

    n = 0
    for slot in scenario.slots:
        err = validate_slot(slot)
        if err:
            raise ValueError(err)

        aug, overrides = compute_augmented_edges(slot, relay_factor=relay_factor) if slot.u2u else ({}, {})
        probs, _ = policy.forward_policy(slot, overrides=overrides if overrides else None)
        mapping, relay_plan = safety_project_aug(slot, probs, aug,
                                                 cap_scale=cap_scale, relay_scale=relay_scale)
        _r, info = env.step(slot, mapping, aug_info=aug if aug else None,
                            relay_plan=relay_plan, cap_scale=cap_scale)

        if per_slot:
            print(f"[t={slot.t:02d}] "
                  f"R={info['reward']:.2f}  thr={info['throughput']:.2f}  "
                  f"lat={info['latency']:.2f}  dist={info['distance']:.2f}  "
                  f"ho={info['handover']:.0f}  viol={info['violations']:.0f}")

        for k in agg.keys():
            agg[k] += float(info[k])
        n += 1

    # medie per slot
    for k in agg.keys():
        agg[k] = agg[k] / max(1, n)

    # stampa riepilogo
    name = scenario.meta.get("name", "scenario")
    print("\n=== EVALUATION SUMMARY ===")
    print(f"scenario: {name}")
    if "flight" in scenario.meta:
        print(f"flight:   {scenario.meta['flight']}")
    if "channel" in scenario.meta:
        print(f"channel:  {scenario.meta['channel']}")
    print(f"slots:    {n}")
    print(f"avg_reward:    {agg['reward']:.2f}")
    print(f"avg_throughput:{agg['throughput']:.2f}")
    print(f"avg_latency:   {agg['latency']:.2f}")
    print(f"avg_distance:  {agg['distance']:.2f}")
    print(f"avg_handover:  {agg['handover']:.2f}")
    print(f"avg_violations:{agg['violations']:.2f}\n")

    return agg

if __name__ == "__main__":
    main()
