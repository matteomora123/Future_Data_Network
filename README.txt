### README TECNICO — PIPELINE COMPLETA NWPU → DATACOM-FULL (LOCO TRAINING & INFERENZA)

---

#### 1. Origine dei dati

Dataset: **“Data set for UWB Cooperative Navigation and Positioning of UAV Cluster”**
Autori: Chen, Zhang, Wang, Zhao – *Northwest Polytechnical University (NWPU), Xi’an, Cina*
Fonte: ResearchGate, DOI: 10.6084/m9.figshare.28121846 (2024)
Linguaggio: Cinese/inglese tecnico — licenza accademica per uso non commerciale

L’esperimento misura la cooperazione di **7 UAV e 8 GS (Ground Station)** mediante moduli UWB DWM1000.
Ogni “channel” (`ch#`) è una sessione reale di 3–6 s di volo campionata a **5–10 Hz** (30 frame).
Le GS sono disposte in un’area rettangolare di ~1.8 × 1.8 km nel campus NWPU, con UAV tra 10 m e 60 m di quota.

Struttura originale:

* `Anchors.mat` → coordinate fisse GS (8 × 3)
* `Position_label_ch#.mat` → posizioni UAV nel tempo (7 × 3 × 30)
* `Dis_anchor_label_ch#.mat` → distanze UAV–GS (8 × 7 × 30)
* Script MATLAB (`Main_straight.m`, `Save_Distance_AL.m`) per validazione numerica

Errori medi UWB < 0.2 m RMS: dati reali e non simulati.

---

#### 2. Obiettivo del progetto

Costruire un **ambiente realistico per addestramento e validazione** di modelli RL/GNN per allocazione dinamica UAV–GS (Huawei Tech Arena 2025).
La pipeline trasforma i `.mat` NWPU in uno **scenario JSON strutturato per DataCom-Full**, mantenendo le distanze reali e aggiungendo feature sintetiche coerenti (capacità, backlog, bitrate, deadline).

---

#### 3. Fase di preparazione dati

Script principale: `prepare_env0_nwpu.py`
Funzioni:

* conversione `.mat` → `.json` multipli o aggregati (per più canali)
* modalità `per_ch`, `concat`, `average`
* generazione metadati e seed riproducibile

Dati mantenuti:

* posizioni 3D reali UAV
* distanze reali UAV–GS

Dati derivati:

* velocità UAV (`f3 = ‖ΔP‖`)
* bitrate empirico (`w2 ≈ 320 − 0.25 × w1 + N(0, 3)`)

Feature sintetiche:

* `f1` backlog ∼ N(150, 60), `f2` deadline ∼ U(1–6)
* `c1` capacità residua ∼ N(120, 30)
* `c2` latenza base ∼ N(20, 8)
* `c3` livello di coda ∼ N(20, 10)

Output:

* JSON slot-based, 30 slot × 7 UAV × 8 GS (≈ 1680 link)
* compatibile con il motore RL DataCom-Full

---

#### 4. Modello e training

Script principale: `datacom_full.py`
Architettura: **GNN EdgeNet + PPO (con Imitation Learning e Safety Layer)**

Pipeline:

1. **IL (warm-start)** imita una baseline greedy *capacity-aware* (allocazione euristica per rapporto bitrate/distanza).
2. **PPO** ottimizza la policy tramite reward combinato:
   R = α·throughput − β·latency − γ·distance − δ·handover − penalty·violations.
3. **Safety Layer** proietta le scelte entro la capacità GS (`cap_scale = 30`).
4. **LOCO** (*Leave-One-Channel-Out*):
   addestramento su canali 2–4 (scenario concatenato), validazione su ch5.

Output:

* `policy.pt` → pesi della rete addestrata
* `submission_*.json` → mappatura UAV→GS/NO_TX per slot

---

#### 5. Modalità di inferenza e valutazione

Inferenza: `--infer scenario_env0_ch5.json --ckpt policy.pt --out submission_ch5.json`
Genera un file che descrive, per ogni slot, la scelta della GS migliore o assenza di trasmissione.

Valutazione (`--eval`):
calcola e stampa per-slot e medi su 30 frame:

* **R** → reward complessivo
* **thr** → throughput servito (MB/s simulati)
* **lat** → latenza media (ms simulati)
* **dist** → distanza UAV–GS media
* **ho** → handover (cambi GS)
* **viol** → violazioni capacità

Esempio medio (cap_scale = 30):
reward ≈ 24 / slot, throughput ≈ 28, latenza ≈ 157 ms, distanza ≈ 87 m, handover ≈ 1.5, nessuna violazione.

---

#### 6. Risultato finale

* Addestramento stabile (loss IL ≈ 4.8, reward PPO > 100)
* Policy valida, con preferenza di GS coerenti con geometria (g5–g7)
* Nessuna violazione di capacità e handover contenuti → comportamento realistico

---

#### 7. Struttura dei file principali

```
Future_Data_Network/
│
├── dataset_addestramento/
│   ├── prepare_env0_nwpu.py      → conversione .mat → .json
│   ├── scan_dataset.py           → analisi struttura dataset originale
│   ├── scenario_env0_*.json      → scenari multi-channel reali
│   └── submission_*.json         → risultati inferenza
│
├── datacom_full.py               → training/inferenza PPO+GNN
├── policy.pt                     → modello addestrato
└── README.md                     → documento tecnico (questo)
```

---

#### 8. Sintesi finale

L’intera pipeline riproduce fedelmente il comportamento fisico del dataset NWPU, trasformandolo in un ambiente di simulazione RL completo e riproducibile.
La strategia **LOCO** (addestramento su subset di canali, validazione su uno escluso) garantisce robustezza e generalizzazione.
Il sistema risultante consente esperimenti su politiche di allocazione UAV–GS, studio degli effetti di capacità, latenza, distanza e handover in uno scenario realistico e misurato sul campo.
