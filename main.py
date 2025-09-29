#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Future Network – Data Communications (Huawei Tech Arena 2025)
IMPLEMENTAZIONE END-TO-END IN UN UNICO FILE, SUDDIVISA IN "CAPITOLI"
Dipendenze: Python 3.9+, numpy, torch. Nessun altro pacchetto.

USO RAPIDO
----------
1) Addestramento su scenari sintetici (IL + PPO con curriculum), salva checkpoint:
   python datacom_full.py --train --episodes 120 --il_steps 200 --ckpt policy.pt

2) Inferenza su Scenario JSON (mappa i tuoi dati al formato generico e lancia):
   python datacom_full.py --infer scenario.json --ckpt policy.pt --out submission.json

NOTE
----
- GNN minimale senza librerie esterne: modello "edge-centric" (EdgeNet) con MLP sugli archi
  del grafo bipartito UAV–GS. Softmax per-UAV su (GS..., NO_TX).
- CTDE (Centralized Training, Decentralized Execution): il critic usa uno stato “globale”
  semplificato durante il training; l’actor resta eseguibile con le sole feature dello slot.
- Safety layer: proiezione greedy entro capacità dei GS prima dell’output finale.
- IL (imitation learning) come warm-start contro baseline greedy capacity-aware.
- PPO con GAE, clipping, entropy bonus e gradient clipping.
================================================================================
"""

# ==============================
# = CAPITOLO 0 · IMPORT & META =
# ==============================

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import argparse, json, math, random, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =======================================
# = CAPITOLO 1 · DATA MODEL & I/O (JSON) =
# =======================================

"""
Formato GENERICO per lavorare subito (etichette “neutre”, poi mappi i tuoi dati):
- Scenario JSON: { "meta": {...}, "slots": [ Slot, ... ] }
- Slot:  t (int), uav: [UAV...], gs: [GS...], links: [Link...]
  UAV:   { "id": "u3", "f1": float, "f2": float, "f3": float }        # es. backlog, deadline, speed
  GS:    { "id": "g1", "c1": float, "c2": float, "c3": float }        # es. capacity, base-latency, queue
  Link:  { "uav_id": "u3", "gs_id": "g1", "w1": float, "w2": float }  # es. distanza/pathloss, bitrate stimato

Output submission:
- { "meta": {...}, "assignments": [ { "t": int, "map": { "uav_id": "gs_id"|"NO_TX" } }, ... ] }
"""

@dataclass
class UAV:
    id: str
    f1: float  # backlog
    f2: float  # deadline
    f3: float  # speed

@dataclass
class GS:
    id: str
    c1: float  # capacity budget (unità astratte)
    c2: float  # base latency
    c3: float  # queue level

@dataclass
class Link:
    uav_id: str
    gs_id: str
    w1: float  # distanza / pathloss proxy
    w2: float  # bitrate stimato

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
    """Carica Scenario JSON in memoria come dataclass strutturati."""
    obj = json.loads(path.read_text())
    slots: List[Slot] = []
    for s in obj["slots"]:
        u = [UAV(**x) for x in s["uav"]]
        g = [GS(**x) for x in s["gs"]]
        l = [Link(**x) for x in s["links"]]
        slots.append(Slot(t=s["t"], uav=u, gs=g, links=l))
    return Scenario(meta=obj.get("meta", {}), slots=slots)


def save_submission(meta: Dict[str, str], assignments: List[Dict[str, str]], out: Path):
    """Scrive la submission nel formato atteso (lista per-slot di mappature UAV->GS/NO_TX)."""
    out.write_text(json.dumps({
        "meta": meta,
        "assignments": [{"t": t, "map": m} for t, m in enumerate(assignments)]
    }, ensure_ascii=False, indent=2))


# ================================================
# = CAPITOLO 2 · GENERATORE DI SCENARI SINTETICI =
# ================================================

def gen_synth(T=32, U=12, G=5, seed=7) -> Scenario:
    """
    Genera un piccolo mondo dinamico:
    - UAV in random walk 2D implicito (solo per calcolare distanza->bitrate).
    - GS fissi con capacità/latency/queue variabili.
    - ogni link ha (w1=dist, w2=bitrate) con rumore.
    """
    random.seed(seed)
    slots: List[Slot] = []
    u_pos = np.random.RandomState(seed).rand(U, 2) * 1000.0
    g_pos = np.random.RandomState(seed+1).rand(G, 2) * 1000.0
    for t in range(T):
        uav = []
        for u in range(U):
            u_pos[u] += np.random.randn(2) * 15.0             # passo casuale
            backlog = max(0.0, np.random.normal(150.0, 60.0)) # dati in coda
            deadline = max(1.0, np.random.uniform(1.0, 6.0))  # finestra temporale
            speed = float(np.linalg.norm(np.random.randn(2) * 5.0))
            uav.append(UAV(id=f"u{u}", f1=backlog, f2=deadline, f3=speed))
        gs = []
        for g in range(G):
            cap = max(30.0, np.random.normal(120.0, 30.0))    # capacità residua
            lat = max(5.0, np.random.normal(20.0, 8.0))       # latenza base
            q   = max(0.0, np.random.normal(20.0, 10.0))      # coda GS
            gs.append(GS(id=f"g{g}", c1=cap, c2=lat, c3=q))
        links = []
        for u in range(U):
            for g in range(G):
                dist = float(np.linalg.norm(u_pos[u] - g_pos[g]) + 1.0)
                rate = max(1.0, 320.0 - 0.25 * dist + np.random.randn()*3.0)
                links.append(Link(uav_id=f"u{u}", gs_id=f"g{g}", w1=dist, w2=rate))
        slots.append(Slot(t=t, uav=uav, gs=gs, links=links))
    return Scenario(meta={"name": "synth"}, slots=slots)


# ===========================================
# = CAPITOLO 3 · VALIDAZIONE DI SANITY CHECK =
# ===========================================

def validate_slot(slot: Slot) -> Optional[str]:
    """Verifica coerenza referenziale degli ID in un singolo slot."""
    uids = {u.id for u in slot.uav}
    gids = {g.id for g in slot.gs}
    for lk in slot.links:
        if lk.uav_id not in uids:
            return f"link con uav_id inesistente: {lk.uav_id} @t={slot.t}"
        if lk.gs_id not in gids:
            return f"link con gs_id inesistente: {lk.gs_id} @t={slot.t}"
    return None


# =======================================
# = CAPITOLO 4 · AMBIENTE/SIMULATORE RL =
# =======================================

class DataComEnv:
    """
    Simulatore a slot singolo con reward:
    reward = α·throughput − β·latency − γ·distance − δ·handover − penalty·violations
    - throughput: somma dei dati serviti (min(backlog, bitrate))
    - latency: proxy = base-latency GS + backlog residuo ponderato
    - distance: somma distanze link scelti (proxy di costo/path-loss)
    - handover: conteggio cambi GS per lo stesso UAV, memoria rapida
    - violations: assegnamenti che sforano la capacità GS
    """
    def __init__(self, alpha=1.0, beta=0.02, gamma=0.0015, delta=0.3, penalty=50.0):
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta
        self.penalty = penalty
        self.prev_choice: Dict[str, str] = {}  # memorizza GS scelto nello slot precedente per conteggiare handover

    def step(self, slot: Slot, mapping: Dict[str, str]) -> Tuple[float, Dict[str, float]]:
        # capacità residua (unità astratte)
        gs_capacity = {g.id: max(1.0, g.c1) for g in slot.gs}
        # indicizzazione rapida dei link reali
        link_map: Dict[Tuple[str, str], Link] = {(lk.uav_id, lk.gs_id): lk for lk in slot.links}

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

            # consumo di capacità proporzionale al bitrate usato (scala arbitraria)
            cons = lk.w2 / 40.0
            if gs_capacity[ch] - cons < -1e-6:
                violations += 1.0
                continue
            gs_capacity[ch] -= cons

            # throughput servito nello slot
            served = min(u.f1, lk.w2)
            throughput += served

            # latenza proxy: latenza base GS + backlog residuo ponderato
            g_obj = next(g for g in slot.gs if g.id == ch)
            latency += (max(1.0, g_obj.c2) + max(0.0, u.f1 - served) * 0.05)

            # distanza/costo
            dist_sum += lk.w1

            # handover (cambio GS rispetto allo slot precedente)
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


# ============================================
# = CAPITOLO 5 · BASELINE GREEDY (TEACHER IL) =
# ============================================

def greedy_policy(slot: Slot, cap_scale: float = 40.0) -> Dict[str, str]:
    """
    Baseline “capacity-aware”:
    - per ogni UAV ordina i GS per utilità ~ rate/dist (preferisce bitrate alto e distanza bassa),
    - assegna rispettando capacità residua (c1/cap_scale).
    Ritorna una mappatura ammissibile o NO_TX se impossibile.
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
        # ordina per (rate/dist)
        cands.sort(key=lambda e: (e.w2 / (e.w1+1.0)), reverse=True)
        chosen = "NO_TX"
        for e in cands:
            need = e.w2 / cap_scale
            if gs_left.get(e.gs_id, 0.0) >= need:
                gs_left[e.gs_id] -= need
                chosen = e.gs_id
                break
        decision[u.id] = chosen
    return decision


# =======================================
# = CAPITOLO 6 · SAFETY LAYER (PROIEZIONE)
# =======================================

def safety_project(slot: Slot,
                   probs: Dict[str, List[Tuple[str,float]]],
                   cap_scale: float = 40.0) -> Dict[str, str]:
    """
    Proiezione greedy delle scelte della policy entro i vincoli di capacità GS:
    - probs: per ogni UAV, lista di (gs_id|NO_TX, probabilità)
    - si prova in ordine di probabilità decrescente finché la capacità del GS lo consente
    """
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


# =======================================================
# = CAPITOLO 7 · POLICY NEURALE "GNN MINIMALE" (EdgeNet) =
# =======================================================

"""
Senza PyG, implementiamo un layer "edge-centric":
- pack_features() costruisce tensori:
  u_feat[U,3] = [f1,f2,f3], g_feat[G,3] = [c1,c2,c3], e_feat[E,2] = [w1,w2]
  edge_index[E] = (u_idx,g_idx)
- EdgeNet concatena per ogni edge: [featU || featG || featLink] -> MLP edge -> logit
- Per ciascun UAV si fa una softmax sui logits dei propri archi + 1 logit NO_TX (calcolato da featU)
Risultato: per UAV una distribuzione su {GS valiti, NO_TX}.
"""

def pack_features(slot: Slot) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                       Dict[str,int], Dict[str,int], List[Tuple[int,int]]]:
    u_ids = {u.id: i for i,u in enumerate(slot.uav)}
    g_ids = {g.id: i for i,g in enumerate(slot.gs)}
    u_feat = torch.tensor([[u.f1, u.f2, u.f3] for u in slot.uav], dtype=torch.float32)
    g_feat = torch.tensor([[g.c1, g.c2, g.c3] for g in slot.gs], dtype=torch.float32)
    e_feat = torch.tensor([[lk.w1, lk.w2] for lk in slot.links], dtype=torch.float32)
    edge_index: List[Tuple[int,int]] = [(u_ids[lk.uav_id], g_ids[lk.gs_id]) for lk in slot.links]
    return u_feat, g_feat, e_feat, u_ids, g_ids, edge_index


class EdgeNet(nn.Module):
    """MLP sugli archi del grafo bipartito + logit NO_TX per ogni UAV."""
    def __init__(self, dU=3, dG=3, dE=2, dh=128):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(dU+dG+dE, dh), nn.ReLU(),
            nn.Linear(dh, dh), nn.ReLU(),
            nn.Linear(dh, 1)
        )
        self.notx_mlp = nn.Sequential(
            nn.Linear(dU, dh//2), nn.ReLU(),
            nn.Linear(dh//2, 1)
        )

    def forward(self, slot: Slot) -> Tuple[Dict[str,List[Tuple[str,float]]], Dict[str,torch.Tensor]]:
        """
        Ritorna:
          - probs_dict: per UAV -> [(gs_id o NO_TX, p), ...]
          - raw_logits_store: per UAV -> tensor logits (GS..., NO_TX) PRIMA della softmax
        """
        u_feat, g_feat, e_feat, u_ids, g_ids, edge_index = pack_features(slot)

        # caso limite: nessun arco disponibile nello slot
        if len(edge_index) == 0:
            probs_dict = {uid: [("NO_TX", 1.0)] for uid in u_ids.keys()}
            return probs_dict, {}

        u_idx = torch.tensor([ui for ui,_ in edge_index], dtype=torch.long)
        g_idx = torch.tensor([gi for _,gi in edge_index], dtype=torch.long)

        # feature edge concatenata: [feat(U) || feat(G) || feat(Link)]
        x  = torch.cat([u_feat[u_idx], g_feat[g_idx], e_feat], dim=-1)  # [E, dU+dG+dE]
        edge_logits = self.edge_mlp(x).squeeze(-1)                      # [E]

        # raggruppa logits per UAV
        per_u_edges: Dict[int, List[Tuple[int, float]]] = {}
        for k,(ui,gi) in enumerate(edge_index):
            per_u_edges.setdefault(ui, []).append((gi, float(edge_logits[k].item())))

        # logit addizionale per NO_TX basato su feature del UAV
        logit_no = self.notx_mlp(u_feat).squeeze(-1)      # [U]

        probs_dict: Dict[str, List[Tuple[str,float]]] = {}
        raw_logits_store: Dict[str, torch.Tensor] = {}

        # costruiamo softmax per-UAV su (GS... + NO_TX)
        for uid, ui in u_ids.items():
            pairs = per_u_edges.get(ui, [])
            logits = torch.tensor([s for _, s in pairs] + [float(logit_no[ui].item())], dtype=torch.float32)
            p = torch.softmax(logits, dim=0).numpy().tolist()
            labels = [f"g{gi}" for gi,_ in pairs] + ["NO_TX"]  # nomi GS derivati dagli indici locali
            probs_dict[uid] = list(zip(labels, p))
            raw_logits_store[uid] = logits

        return probs_dict, raw_logits_store


class PPOPolicy(nn.Module):
    """
    Policy completa:
    - EdgeNet: produce distribuzioni per-UAV
    - Value head (critic): valuta lo stato con un embedding globale “leggero”
      (media delle feature di tutti i UAV e di tutti i GS).
    """
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
        u_feat, g_feat, _, _, _, _ = pack_features(slot)
        u_mean = u_feat.mean(dim=0)
        g_mean = g_feat.mean(dim=0)
        z = torch.cat([u_mean, g_mean], -1)
        return self.v_mlp(z).squeeze(-1)  # scalare


# ==============================================
# = CAPITOLO 8 · TRAINER PPO (+ IL & CURRICULUM) =
# ==============================================

def ppo_train(env,
              episodes=200,
              gamma=0.98,
              lam=0.95,
              clip=0.2,
              lr=3e-4,
              curriculum=(8,3,16,6),
              il_steps=200,
              device="cpu",
              seed=42) -> PPOPolicy:
    """
    Fasi:
    1) IL warm-start: la policy imita la baseline greedy su scenari semplici (U,G piccoli).
    2) PPO con curriculum: dimensione problema cresce da (U0,G0) a (U1,G1).
    - GAE semplificato
    - Clipping ratio PPO
    - Entropy bonus per esplorazione
    - Gradient clipping per stabilità
    """
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    policy = PPOPolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    # --- Fase IL: cross-entropy contro greedy (teacher) ---
    for step in range(il_steps):
        scn = gen_synth(T=1, U=curriculum[0], G=curriculum[1], seed=seed+step)
        slot = scn.slots[0]
        target = greedy_policy(slot)
        probs, raw = policy.forward_policy(slot)

        loss = 0.0
        for uid, choices in probs.items():
            labels = [lab for lab,_ in choices]
            y = target.get(uid, "NO_TX")
            if y not in labels:
                y = "NO_TX"
            yi = labels.index(y)
            logits = raw[uid]
            ce = F.cross_entropy(logits.unsqueeze(0), torch.tensor([yi]))
            loss = loss + ce

        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if (step+1) % 50 == 0:
            print(f"[IL] step {step+1}/{il_steps} loss={float(loss):.4f}")

    # --- Fase PPO: roll-out, GAE, update ---
    U0,G0, U1,G1 = curriculum
    for ep in range(episodes):
        frac = ep / max(1, episodes-1)
        U = int(U0 + (U1-U0)*frac)
        G = int(G0 + (G1-G0)*frac)

        scn = gen_synth(T=8, U=U, G=G, seed=seed+1000+ep)
        env.prev_choice = {}  # azzera memoria handover a inizio episodio

        traj, values = [], []

        # Roll-out episodico
        for slot in scn.slots:
            probs, _ = policy.forward_policy(slot)
            # Safety layer prima di agire
            mapping = safety_project(slot, probs)
            # log-prob totale (indipendenza per UAV)
            logp = 0.0
            for uid, choices in probs.items():
                d = dict(choices)
                chosen = mapping.get(uid, "NO_TX")
                p = max(1e-8, d.get(chosen, d.get("NO_TX", 1e-8)))
                logp += math.log(p)

            r, info = env.step(slot, mapping)
            v = policy.forward_value(slot)

            traj.append((slot, mapping, logp, r, info))
            values.append(v)

        # GAE (Generalized Advantage Estimation) semplice
        with torch.no_grad():
            vals = torch.stack(values + [torch.tensor(0.0)])
        rewards = torch.tensor([tr[3] for tr in traj], dtype=torch.float32)
        adv = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(traj))):
            delta = rewards[t] + gamma * float(vals[t+1]) - float(vals[t])
            gae = delta + gamma * lam * gae
            adv[t] = gae
        returns = adv + vals[:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Aggiornamento PPO (1 epoca compatta)
        pol_loss = 0.0; val_loss = 0.0; ent_bonus = 0.0
        for (slot, mapping, old_logp, r, _), A, R in zip(traj, adv, returns):
            probs, _ = policy.forward_policy(slot)

            # Nuovo logp e entropia (sommata su UAV)
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
        pol_loss /= n; val_loss /= n; ent_bonus /= n
        loss = pol_loss + 0.5*val_loss - 0.001*ent_bonus

        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if (ep+1) % 10 == 0:
            avg_R = float(torch.mean(rewards))
            print(f"[PPO] ep {ep+1}/{episodes}  U={U} G={G}  Ravg={avg_R:.2f}  loss={float(loss):.3f}")

    return policy


# =========================================
# = CAPITOLO 9 · INFERENZA E PRODUZIONE I/O =
# =========================================

def run_inference(policy: PPOPolicy, scenario: Scenario) -> List[Dict[str,str]]:
    """
    Per ogni slot:
    - forward policy -> probabilità per-UAV
    - safety projection -> mappa ammissibile UAV->GS/NO_TX
    - opzionale: step nell’ambiente per loggare metriche locali
    """
    env = DataComEnv()
    assignments: List[Dict[str,str]] = []
    env.prev_choice = {}
    for slot in scenario.slots:
        err = validate_slot(slot)
        if err:
            raise ValueError(err)
        probs, _ = policy.forward_policy(slot)
        mapping = safety_project(slot, probs)
        assignments.append(mapping)
        _r, _info = env.step(slot, mapping)  # puoi loggare info se ti serve
    return assignments


# ====================================
# = CAPITOLO 10 · CLI / ENTRY-POINT  =
# ====================================

def main():
    ap = argparse.ArgumentParser(description="Future Network – DataCom (GNN+PPO full minimal)")
    ap.add_argument("--train", action="store_true", help="train su scenari sintetici (IL+PPO) e salva checkpoint")
    ap.add_argument("--episodes", type=int, default=100, help="PPO episodes")
    ap.add_argument("--il_steps", type=int, default=150, help="imitation (warm-start) steps")
    ap.add_argument("--ckpt", type=Path, default=Path("policy.pt"), help="percorso checkpoint .pt")
    ap.add_argument("--infer", type=Path, help="inferenzia su input Scenario JSON")
    ap.add_argument("--out",   type=Path, help="salva submission JSON")
    args = ap.parse_args()

    if args.train:
        env = DataComEnv()
        policy = ppo_train(env,
                           episodes=args.episodes,
                           il_steps=args.il_steps,
                           curriculum=(8,3,16,6),  # (U0,G0,U1,G1)
                           lr=3e-4,
                           seed=123)
        torch.save(policy.state_dict(), args.ckpt)
        print(f"Checkpoint salvato -> {args.ckpt}")

    if args.infer:
        scn = load_scenario(args.infer)
        policy = PPOPolicy()
        if args.ckpt.exists():
            policy.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
        else:
            print("ATTENZIONE: nessun checkpoint trovato, policy random.")
        assignments = run_inference(policy, scn)
        out = args.out or Path("submission.json")
        save_submission({"solver":"gnn_ppo_ctde"}, assignments, out)
        print(f"Submission scritta -> {out}")


if __name__ == "__main__":
    main()
