#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Future Network – Data Communications (Huawei Tech Arena 2025)
VERSIONE ESTESA: supporto addestramento su dataset reali (LOCO / HOLD-OUT / K-FOLD),
validazione reale, fusioni synth + reali, teacher con isteresi anti-handover.
Dipendenze: Python 3.9+, numpy, torch.
================================================================================

USO TIPICO
----------
# 1) Prepara scenari reali (uno o più file JSON) con prepare_env0_nwpu.py
#    - per_ch:   scenario_env0_ch2.json, scenario_env0_ch3.json, ...
#    - concat:   scenario_env0_concat.json (slot concatenati di più canali)

# 2) Addestramento su REALI con validazione LOCO:
python datacom_full.py --train \
  --train_scenarios dataset_addestramento/scenario_env0_ch2.json dataset_addestramento/scenario_env0_ch3.json \
  --val_scenarios   dataset_addestramento/scenario_env0_ch4.json \
  --use_loco \
  --episodes 60 --il_steps 150 --ckpt policy.pt

# 3) Inferenza su un qualsiasi scenario JSON:
python datacom_full.py --infer dataset_addestramento/scenario_env0_ch5.json --ckpt policy.pt --out submission.json

NOTE
----
- Policy EdgeNet “edge-centric” (MLP sugli archi) + NO_TX logit.
- CTDE: critic su embedding globale (medie di feature).
- Safety-layer (projection) per rispettare capacità GS.
- Teacher “capacity-aware + isteresi anti-handover” per IL.
- Tre strategie di validazione (commentate in codice):
  * LOCO (Leave-One-Channel-Out)    [implementata]
  * HOLD-OUT (split per slot/per canale)
  * K-FOLD (cross-validation)       [traccia commentata]
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

@dataclass
class GS:
    id: str
    c1: float
    c2: float
    c3: float

@dataclass
class Link:
    uav_id: str
    gs_id: str
    w1: float
    w2: float

@dataclass
class Slot:
    t: int
    uav: List[UAV]
    gs: List[GS]
    links: List[Link]

@dataclass
class Scenario:
    meta: Dict[str, str]
    slots: List[Slot]

def load_scenario(path: Path) -> Scenario:
    obj = json.loads(path.read_text())
    slots: List[Slot] = []
    for s in obj["slots"]:
        u = [UAV(**x) for x in s["uav"]]
        g = [GS(**x) for x in s["gs"]]
        l = [Link(**x) for x in s["links"]]
        slots.append(Slot(t=s["t"], uav=u, gs=g, links=l))
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
    u_pos = np.random.RandomState(seed).rand(U, 2) * 1000.0
    g_pos = np.random.RandomState(seed+1).rand(G, 2) * 1000.0
    for t in range(T):
        uav = []
        for u in range(U):
            u_pos[u] += np.random.randn(2) * 15.0
            backlog = max(0.0, np.random.normal(150.0, 60.0))
            deadline = max(1.0, np.random.uniform(1.0, 6.0))
            speed = float(np.linalg.norm(np.random.randn(2) * 5.0))
            uav.append(UAV(id=f"u{u}", f1=backlog, f2=deadline, f3=speed))
        gs = []
        for g in range(G):
            cap = max(30.0, np.random.normal(120.0, 30.0))
            lat = max(5.0, np.random.normal(20.0, 8.0))
            q   = max(0.0, np.random.normal(20.0, 10.0))
            gs.append(GS(id=f"g{g}", c1=cap, c2=lat, c3=q))
        links = []
        for u in range(U):
            for g in range(G):
                dist = float(np.linalg.norm(u_pos[u] - g_pos[g]) + 1.0)
                rate = max(1.0, 320.0 - 0.25 * dist + np.random.randn()*3.0)
                links.append(Link(uav_id=f"u{u}", gs_id=f"g{g}", w1=dist, w2=rate))
        slots.append(Slot(t=t, uav=uav, gs=gs, links=links))
    return Scenario(meta={"name": "synth"}, slots=slots)

def validate_slot(slot: Slot) -> Optional[str]:
    uids = {u.id for u in slot.uav}
    gids = {g.id for g in slot.gs}
    for lk in slot.links:
        if lk.uav_id not in uids:
            return f"link con uav_id inesistente: {lk.uav_id} @t={slot.t}"
        if lk.gs_id not in gids:
            return f"link con gs_id inesistente: {lk.gs_id} @t={slot.t}"
    return None


# =====================================
# = CAPITOLO 3: Ambiente / Reward     =
# =====================================

class DataComEnv:
    def __init__(self, alpha=1.0, beta=0.02, gamma=0.0015, delta=0.3, penalty=50.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.penalty = penalty
        self.prev_choice: Dict[str, str] = {}

    def step(self, slot: Slot, mapping: Dict[str, str]) -> Tuple[float, Dict[str, float]]:
        gs_capacity = {g.id: max(1.0, g.c1) for g in slot.gs}
        link_map = {(lk.uav_id, lk.gs_id): lk for lk in slot.links}

        throughput = 0.0
        latency = 0.0
        dist_sum = 0.0
        handovers = 0.0
        violations = 0.0

        for u in slot.uav:
            ch = mapping.get(u.id, "NO_TX")
            if ch == "NO_TX":
                continue
            lk = link_map.get((u.id, ch), None)
            if lk is None:
                violations += 1.0
                continue
            cons = lk.w2 / 30.0
            if gs_capacity[ch] - cons < -1e-6:
                violations += 1.0
                continue
            gs_capacity[ch] -= cons
            served = min(u.f1, lk.w2)
            throughput += served
            g_obj = next(g for g in slot.gs if g.id == ch)
            latency += (max(1.0, g_obj.c2) + max(0.0, u.f1 - served) * 0.05)
            dist_sum += lk.w1
            if self.prev_choice.get(u.id, ch) != ch and u.id in self.prev_choice:
                handovers += 1.0

        reward = (self.alpha * throughput
                  - self.beta * latency
                  - self.gamma * dist_sum
                  - self.delta * handovers
                  - self.penalty * violations)
        self.prev_choice = {u.id: mapping.get(u.id, "NO_TX") for u in slot.uav}
        info = dict(throughput=throughput, latency=latency, distance=dist_sum,
                    handover=handovers, violations=violations, reward=reward)
        return reward, info


# =========================================
# = CAPITOLO 4: Baseline greedy + Isteresi =
# =========================================

def greedy_policy(slot: Slot, cap_scale: float = 30.0) -> Dict[str, str]:
    links_by_u: Dict[str, List[Link]] = {}
    for lk in slot.links:
        links_by_u.setdefault(lk.uav_id, []).append(lk)
    gs_left = {g.id: max(1.0, g.c1)/cap_scale for g in slot.gs}
    decision: Dict[str, str] = {}
    for u in slot.uav:
        cands = links_by_u.get(u.id, [])
        if not cands:
            decision[u.id] = "NO_TX"; continue
        cands.sort(key=lambda e: (e.w2 / (e.w1 + 1.0)), reverse=True)
        chosen = "NO_TX"
        for e in cands:
            need = e.w2 / cap_scale
            if gs_left.get(e.gs_id, 0.0) >= need:
                gs_left[e.gs_id] -= need
                chosen = e.gs_id
                break
        decision[u.id] = chosen
    return decision

def greedy_with_hysteresis(slot: Slot, prev_choice: Dict[str, str],
                           cap_scale: float = 30.0, lambda_h: float = 0.15) -> Dict[str, str]:
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
            base = e.w2 / (e.w1 + 1.0)
            if prev_choice.get(u.id, None) == e.gs_id:
                base += lambda_h * base
            scores.append((e, base))
        scores.sort(key=lambda x: x[1], reverse=True)
        chosen = "NO_TX"
        for e, _sc in scores:
            need = e.w2 / cap_scale
            if gs_left.get(e.gs_id, 0.0) >= need:
                gs_left[e.gs_id] -= need
                chosen = e.gs_id
                break
        decision[u.id] = chosen
    return decision

def safety_project(slot: Slot,
                   probs: Dict[str, List[Tuple[str, float]]],
                   cap_scale: float = 30.0) -> Dict[str, str]:
    gs_left = {g.id: max(1.0, g.c1)/cap_scale for g in slot.gs}
    link_map = {(lk.uav_id, lk.gs_id): lk for lk in slot.links}
    decision: Dict[str, str] = {}
    for u in slot.uav:
        ranked = sorted(probs[u.id], key=lambda x: x[1], reverse=True)
        chosen = "NO_TX"
        for gs_id, _p in ranked:
            if gs_id == "NO_TX":
                chosen = "NO_TX"; break
            lk = link_map.get((u.id, gs_id))
            if not lk:
                continue
            need = lk.w2 / cap_scale
            if gs_left.get(gs_id, 0.0) >= need:
                gs_left[gs_id] -= need
                chosen = gs_id
                break
        decision[u.id] = chosen
    return decision


# =========================================
# = CAPITOLO 5: GNN / EdgeNet / PPOPolicy =
# =========================================

def pack_features(slot: Slot) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                       Dict[str,int], Dict[str,int], List[Tuple[int,int]]]:
    u_ids = {u.id: i for i, u in enumerate(slot.uav)}
    g_ids = {g.id: i for i, g in enumerate(slot.gs)}
    u_feat = torch.tensor([[u.f1, u.f2, u.f3] for u in slot.uav], dtype=torch.float32)
    g_feat = torch.tensor([[g.c1, g.c2, g.c3] for g in slot.gs], dtype=torch.float32)
    e_feat = torch.tensor([[lk.w1, lk.w2] for lk in slot.links], dtype=torch.float32)
    edge_index = [(u_ids[lk.uav_id], g_ids[lk.gs_id]) for lk in slot.links]
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

    def forward(self, slot: Slot) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, torch.Tensor]]:
        device = next(self.parameters()).device
        u_feat, g_feat, e_feat, u_ids, g_ids, edge_index = pack_features(slot)
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

        for uid, ui in u_ids.items():
            pos_list = per_u_pos.get(ui, [])
            gs_list = per_u_gids.get(ui, [])
            if pos_list:
                e_log = edge_logits[pos_list]
                logits = torch.cat([e_log, logit_no[ui:ui+1]], dim=0)
                labels = [f"g{gi}" for gi in gs_list] + ["NO_TX"]
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

    def forward_policy(self, slot: Slot):
        return self.net(slot)

    def forward_value(self, slot: Slot) -> torch.Tensor:
        device = next(self.parameters()).device
        u_feat, g_feat, _, _, _, _ = pack_features(slot)
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
              device: torch.device = get_device(),
              seed=42) -> PPOPolicy:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    policy = PPOPolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    # IL su sintetico con teacher greedy semplice
    for step in range(il_steps):
        scn = gen_synth(T=1, U=curriculum[0], G=curriculum[1], seed=seed+step)
        slot = scn.slots[0]
        target = greedy_policy(slot)
        probs, raw = policy.forward_policy(slot)
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
            probs, _ = policy.forward_policy(slot)
            mapping = safety_project(slot, probs)
            logp = 0.0
            for uid, choices in probs.items():
                d = dict(choices)
                chosen = mapping.get(uid, "NO_TX")
                p = max(1e-8, d.get(chosen, d.get("NO_TX", 1e-8)))
                logp += math.log(p)
            r, _info = env.step(slot, mapping)
            v = policy.forward_value(slot)
            traj.append((slot, mapping, logp, r))
            values.append(v)

        # GAE
        with torch.no_grad():
            vals = torch.stack(values + [torch.tensor(0.0, device=device)])
        rewards = torch.tensor([tr[3] for tr in traj], dtype=torch.float32, device=device)
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
        for (slot, mapping, old_logp, _r), A, R in zip(traj, adv, returns):
            probs, _ = policy.forward_policy(slot)
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
# = CAPITOLO 7: Training REALE (LOCO) completo con PPO operativo     =
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
                        device: torch.device = get_device(),
                        seed=123) -> PPOPolicy:
    """
    PPO su dati REALI:
    - IL warm-start contro teacher greedy+isteresi (capacity-aware con memoria).
    - Rollout PPO scorrendo gli slot reali (manteniamo contiguità per l’handover).
    - Validazione LOCO: media reward su scenari di validazione; tiene best checkpoint.
    """
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    policy = PPOPolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    # ----------- IL su REALI con teacher “capacity-aware + isteresi” -----------
    all_train_slots: List[Tuple[Slot, int]] = []
    for si, sc in enumerate(train_scn):
        for sl in sc.slots:
            all_train_slots.append((sl, si))

    prev_choice_teacher: Dict[int, Dict[str, str]] = {i: {} for i in range(len(train_scn))}

    for step in range(il_steps):
        slot, scen_idx = random.choice(all_train_slots)
        # teacher con isteresi: usa memoria per scenario (per far contare gli handover)
        teacher_map = greedy_with_hysteresis(slot, prev_choice_teacher[scen_idx])
        # aggiorna memoria teacher per slot successivo
        prev_choice_teacher[scen_idx] = {u.id: teacher_map.get(u.id, "NO_TX") for u in slot.uav}

        probs, raw = policy.forward_policy(slot)
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
        # Scorriamo tutti gli scenari di training (mantenendo la sequenza di slot)
        random.shuffle(train_scn)  # randomizza l’ordine degli scenari a ogni epoca
        total_rew = 0.0
        total_cnt = 0

        for sc in train_scn:
            env.prev_choice = {}  # reset per scenario
            traj, values = [], []

            for slot in sc.slots:
                probs, _ = policy.forward_policy(slot)
                mapping = safety_project(slot, probs)  # safety prima di agire

                # logp aggregata (indipendenza per-UAV)
                logp = 0.0
                for uid, choices in probs.items():
                    d = dict(choices)
                    chosen = mapping.get(uid, "NO_TX")
                    p = max(1e-8, d.get(chosen, d.get("NO_TX", 1e-8)))
                    logp += math.log(p)

                r, _info = env.step(slot, mapping)
                v = policy.forward_value(slot)

                traj.append((slot, mapping, logp, r))
                values.append(v)
                total_rew += r
                total_cnt += 1

            # ---- Update PPO per scenario (mantiene continuità handover) ----
            with torch.no_grad():
                vals = torch.stack(values + [torch.tensor(0.0, device=device)])
            rewards = torch.tensor([tr[3] for tr in traj], dtype=torch.float32, device=device)
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
            for (slot, mapping, old_logp, _r), A, R in zip(traj, adv, returns):
                probs, _ = policy.forward_policy(slot)

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
                    probs, _ = policy.forward_policy(slot)
                    mapping = safety_project(slot, probs)
                    r, _info = env.step(slot, mapping)
                    vtot += r; vcnt += 1
            avg_val = vtot / max(1, vcnt)
            print(f"[PPO-real][VAL] ep {ep+1}/{episodes} avg_val_reward={avg_val:.3f}")
            if avg_val > best_val:
                best_val = avg_val
                best_state = {k: v.clone().detach().cpu() for k, v in policy.state_dict().items()}

    if best_state is not None:
        policy.load_state_dict(best_state)
        print(f"[PPO-real] ripristinato best checkpoint (val reward={best_val:.3f})")
    return policy


# =================================================
# = CAPITOLO 8: Inferenza e produzione submission =
# =================================================

def run_inference(policy: PPOPolicy, scenario: Scenario, device: torch.device) -> List[Dict[str,str]]:
    policy.to(device)
    env = DataComEnv()
    assignments: List[Dict[str, str]] = []
    env.prev_choice = {}
    for slot in scenario.slots:
        err = validate_slot(slot)
        if err:
            raise ValueError(err)
        probs, _ = policy.forward_policy(slot)
        mapping = safety_project(slot, probs)
        assignments.append(mapping)
        _r, _info = env.step(slot, mapping)
    return assignments


# ====================================
# = CAPITOLO 9: Entry-point /  CLI   =
# ====================================

def main():
    ap = argparse.ArgumentParser(description="Future Network – DataCom (esteso: LOCO, reali, hybrid)")
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
                                         device=device)
        else:
            # Fallback: training sintetico
            policy = ppo_train(env,
                               episodes=args.episodes,
                               il_steps=args.il_steps,
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
        assignments = run_inference(policy, scn, device)
        out = args.out or Path("submission.json")
        save_submission({"solver": "gnn_ppo_ctde_loco"}, assignments, out)
        print(f"Submission scritta -> {out}")
    if args.eval:
        scn = load_scenario(args.eval)
        policy = PPOPolicy().to(device)
        if args.ckpt.exists():
            policy.load_state_dict(torch.load(args.ckpt, map_location=device))
        else:
            print("ATTENZIONE: nessun checkpoint trovato, policy random.")
        _ = evaluate_scenario(policy, scn, device=device, per_slot=args.per_slot)


def evaluate_scenario(policy: PPOPolicy, scenario: Scenario, device: torch.device, per_slot: bool = False) -> Dict[str, float]:
    """
    Valuta la policy sullo scenario (senza salvare submission).
    Usa DataComEnv per ricostruire metriche e reward slot-by-slot.
    Ritorna un dizionario con le medie aggregate.
    """

    """
    Per ogni tempo t stampiamo le metriche che il simulatore calcola usando la mappa UAV→GS scelta dalla policy:
    - R = reward dello slot =
    - α·throughput − β·latency − γ·distance − δ·handover − penalty·violations
    - (con i pesi del codice: α=1.0, β=0.02, γ=0.0015, δ=0.3, penalty=50.0).
    - thr = throughput servito nello slot (somma dei min(backlog, bitrate) su tutti gli UAV).
    - lat = latenza proxy (latency base GS + backlog residuo ponderato), sommata sugli UAV.
    - dist = somma delle distanze dei link scelti (proxy di path-loss/costo).
    - ho = handover contati nello slot (quanti UAV hanno cambiato GS rispetto allo slot precedente).
    - viol = violazioni di capacità (quante assegnazioni avrebbero sforato il budget GS; dovrebbero essere 0 grazie al safety layer).
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
        probs, _ = policy.forward_policy(slot)
        mapping = safety_project(slot, probs)
        _r, info = env.step(slot, mapping)

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
