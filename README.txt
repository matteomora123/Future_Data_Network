### README TECNICO — PIPELINE COMPLETA NWPU → DATACOM-FULL (LOCO TRAINING & INFERENZA)

---

#### 1. Origine dei dati

Dataset: **“Data set for UWB Cooperative Navigation and Positioning of UAV Cluster”**
Autori: Chen, Zhang, Wang, Zhao – *Northwest Polytechnical University (NWPU), Xi’an, Cina*
Fonte: ResearchGate, DOI: 10.6084/m9.figshare.28121846 (2024)
Linguaggio: cinese/inglese tecnico — licenza accademica per uso non commerciale

L’esperimento studia la cooperazione di **7 UAV e 8 GS (Ground Station)** dotati di moduli **UWB DWM1000 / LinkTrack P-B**.
Le prove “Environment 0” sono condotte **all’aperto**, su un **campo di 20 m × 18 m × 10 m**, con frequenza di acquisizione **50 Hz** e **30 frame campionati** per ciascun volo (≈ 3 s).
Le quote UAV variano tra **0 m e 10 m**, e gli anchor/GS sono fissati a terra ai vertici del campo.

Sono forniti quattro **canali UWB (ch2 – ch5)**: non rappresentano multipath, ma **configurazioni RF diverse** (due frequenze operative ≈ 3.99 GHz e 4.49 GHz × due codifiche di spreading), utili a confrontare l’accuratezza delle misure di distanza.

Struttura originale:

* `Anchors.mat` → coordinate fisse GS (8 × 3, in metri)
* `Position_label_ch#.mat` → posizioni UAV (7 × 3 × 30)
* `Dis_anchor_label_ch#.mat` → distanze UAV–GS (8 × 7 × 30, in centimetri → convertite in metri)
* Script MATLAB (`Main_straight.m`, `Save_Distance_AL.m`) per la validazione numerica

Errore medio UWB < 0.2 m RMS → **dati reali e non simulati**.

---

#### 2. Obiettivo del progetto

Costruire un **ambiente realistico per addestramento e validazione** di modelli RL/GNN dedicati all’allocazione dinamica UAV–GS per la **Huawei Tech Arena 2025**.
La pipeline trasforma i `.mat` NWPU in **file JSON slot-based** compatibili con **DataCom-Full**, mantenendo le misure reali e generando feature sintetiche coerenti (capacità, backlog, bitrate, deadline).

---

#### 3. Fase di preparazione dati

Script: `prepare_env0_nwpu.py`
Funzioni principali:

* conversione `.mat` → `.json` per singolo o multi-canale
* modalità `per_ch`, `concat`, `average`
* generazione metadati e seed riproducibile

**Dati reali conservati:**

* posizioni 3D UAV
* distanze UAV–GS (convertite cm → m)

**Dati derivati:**

* velocità UAV `f3 = ‖ΔP‖`
* bitrate empirico `w2 ≈ 320 − 0.25 × w1 + N(0, 3)`

**Feature sintetiche (float, distribuzioni plausibili):**

* `f1` backlog ∼ N(150, 60)
* `f2` deadline ∼ U(1, 6)
* `c1` capacità GS ∼ N(120, 30)
* `c2` latenza base ∼ N(20, 8)
* `c3` livello coda ∼ N(20, 10)

Output:

* JSON con 30 slot × 7 UAV × 8 GS (≈ 1680 link)
* compatibile con il motore RL DataCom-Full

---

#### 4. Modello e training

Script: `datacom_full.py`
Architettura: **EdgeNet (GNN) + PPO**, con fase iniziale di **Imitation Learning** e **Safety Layer**

Pipeline:

1. **IL (warm-start)**: apprende una baseline *greedy capacity-aware* (scelta locale massimizza bitrate/distanza).
2. **PPO**: ottimizza la policy con reward composito
   `R = α·throughput − β·latency − γ·distance − δ·handover − penalty·violations`.
3. **Safety Layer**: limita le scelte entro la capacità GS (`cap_scale = 30`).
4. **LOCO (Leave-One-Channel-Out)**: train su ch2–ch4 (concat), validazione su ch5.

Output:

* `policy.pt` → pesi rete addestrata
* `submission_*.json` → mappatura UAV → GS/NO_TX per slot

---

#### 5. Inferenza e valutazione

Comando inferenza:
`python datacom_full.py --infer scenario_env0_ch5.json --ckpt policy.pt --out submission_ch5.json`

Genera per ogni slot la scelta della GS più efficiente o nessuna trasmissione.

Valutazione (`--eval`):
riporta metriche per slot e medie su 30 frame:

* **R** reward complessiva
* **thr** throughput simulato (MB/s)
* **lat** latenza (ms)
* **dist** distanza media UAV–GS (m)
* **ho** handover (cambi GS)
* **viol** violazioni di capacità

Esempio medio (cap_scale = 30):
reward ≈ 24, throughput ≈ 28, latenza ≈ 157 ms, distanza ≈ 87 m, handover ≈ 1.5, violazioni = 0.

---

#### 6. Risultati

* Addestramento stabile (loss IL ≈ 4.8, reward PPO > 100)
* Policy valida, preferenze GS coerenti (g5–g7 più prossime agli UAV)
* Nessuna violazione di capacità, handover limitati → comportamento realistico

---

#### 7. Struttura file

```
Future_Data_Network/
│
├── dataset_addestramento/
│   ├── prepare_env0_nwpu.py
│   ├── scan_dataset.py
│   ├── scenario_env0_*.json
│   └── submission_*.json
│
├── datacom_full.py
├── policy.pt
└── README.md
```

---

#### 8. Sintesi finale

La pipeline **NWPU → DataCom-Full** ricrea in Python lo scenario fisico UWB reale (20×18 m, 7 UAV, 8 GS, 50 Hz) in forma discreta da 30 slot, con variabili sintetiche utili al training RL.
La strategia **LOCO** (train ch2–ch4, val ch5) assicura generalizzazione e robustezza.
Il sistema consente test realistici di politiche UAV–GS, analisi di capacità, latenza, distanza e handover in un ambiente basato su misure realmente acquisite.
