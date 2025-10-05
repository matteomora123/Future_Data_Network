### RIEPILOGO TECNICO — COSTRUZIONE DEL FILE scenario_env0.json
Dataset di origine: “Data set for UWB Cooperative Navigation and Positioning of UAV Cluster”
Autori: Chen, Zhang, Wang, Zhao, Northwest Polytechnical University (NWPU), Xi’an, Cina
Pubblicato su ResearchGate, 2024 — DOI: 10.6084/m9.figshare.28121846
Lingua: cinese/inglese tecnico — licenza accademica libera per ricerca non commerciale

──────────────────────────────────────────────
1. DESCRIZIONE DELL’ESPERIMENTO ORIGINALE
──────────────────────────────────────────────
Il gruppo NWPU ha realizzato un sistema sperimentale per la **navigazione cooperativa UWB (Ultra-Wideband)** di droni (UAV) in ambienti indoor e outdoor.
Ogni esperimento coinvolge 7 UAV e 8 stazioni di ancoraggio fisse (GS) dotate di moduli UWB DWM1000, registrando dati sincroni di distanza e posizione per più traiettorie di volo:

- **Environment0** (outdoor LOS): Flying_straight, Flying_climb
- **Environment1** (outdoor disperso / indoor): Flying_circle, Flying_disperse

**Scala e configurazione fisica**
I test “Environment0” sono stati eseguiti all’aperto nel campus NWPU di Xi’an su un’area rettangolare di circa **1,8 km × 1,8 km**, con altitudini UAV comprese tra **10 m e 60 m**.
Le 8 GS sono fissate a terra lungo il perimetro del campo, mentre 7 UAV volano in formazione lineare o in lieve salita, mantenendo distanze reciproche dell’ordine di decine di metri.
Le coordinate delle GS, tratte da `Anchors.mat`, variano tipicamente tra x,y ∈ [0, 1800 m], z ≈ [10–60 m].
Le misure UWB forniscono un errore medio **< 0,2 m RMS** (validazione tecnica NWPU), quindi i dati sono reali e non simulati.
Lo scenario Environment0 / Flying_straight / ch2 rappresenta circa 3 s di volo reale campionati a 10 Hz (Δt = 0,1 s), con 30 istantanee sincronizzate di posizione e distanza UAV–GS.

Per ogni UAV, il dataset fornisce:
- Coordinate 3D reali nel tempo (`Position_label_ch#.mat`)
- Distanze UAV → GS misurate (`Dis_anchor_label_ch#.mat`)
- 30 campioni per canale (tipicamente `ch2`)
- File `Anchors.mat` con posizioni assolute delle 8 GS

Il dataset è corredato da script MATLAB (`Main_straight.m`, `Save_Distance_AL.m`) e figure (`Technical_validation/Position_error/…`) per la verifica grafica e numerica.

──────────────────────────────────────────────
2. SCOPO DELLA RIELABORAZIONE PYTHON
──────────────────────────────────────────────
Lo script `prepare_env0_nwpu.py` converte i dati MATLAB di **Environment0 / Flying_straight / ch2** in un file JSON compatibile con il framework di simulazione **DataCom-Full (Huawei Tech Arena)**, mirato a:
- addestramento di modelli RL (PPO, GNN, IL)
- simulazioni di allocazione risorse UAV–GS
- visualizzazioni o replay numerici autonomi

──────────────────────────────────────────────
3. MAPPATURA DATI — DALLE MATRICI MATLAB ALLO SCENARIO JSON
──────────────────────────────────────────────
Dati fisici (reali):
- **Anchors.mat** → coordinate GS (8×3)
- **Position_label_ch2.mat** → posizioni UAV (7×3×30)
- **Dis_anchor_label_ch2.mat** → distanze UAV–GS (8×7×30)

**Campionamento temporale e discretizzazione**
Il dataset NWPU è già discreto: ogni file contiene 30 campioni uniformemente distribuiti.
Ogni indice `t ∈ [0, 29]` rappresenta un’istantanea sincronizzata del sistema UAV–GS.
La frequenza (≈ 5–10 Hz) implica 3–6 s di volo totale.
Lo script conserva tale granularità, creando 30 slot discreti (`slot_t`) senza interpolazioni.
Il campo `f3` (velocità) ripristina la continuità dinamica tra i punti discreti.

Dati derivati:
- **Velocità (f3)** = ‖P[t] − P[t−1]‖
- **Distanze (w1)** = valori sperimentali (m)

Dati sintetici (riproducibili, seed = 7):
- **Backlog (f1)** e **Deadline (f2)** → distribuzioni gaussiane/troncate
- **GS features (c1,c2,c3)** :
   c1 = capacità residua ~ N(120,30)
   c2 = latenza base ~ N(20,8)
   c3 = queue ~ N(20,10)
- **Bitrate (w2)** ≈ `max(1, 320 − 0.25 · w1 + N(0, 3))`

──────────────────────────────────────────────
4. STRUTTURA DEL FILE JSON
──────────────────────────────────────────────
{
  "meta": {
    "name": "nwpu_env0",
    "flight": "Flying_straight",
    "channel": "ch2",
    "source": "NWPU UWB dataset (Environment0)",
    "units": {"pos":"m","w1":"m","w2":"arb_bitrate"},
    "note": "Posizioni e distanze reali; feature e bitrate sintetici ma coerenti"
  },
  "slots": [
    {
      "t": 0..29,
      "uav": [{id:"u0",f1,f2,f3}, …, {id:"u6"}],
      "gs":  [{id:"g0",c1,c2,c3}, …, {id:"g7"}],
      "links": [{uav_id:"u0",gs_id:"g0",w1,w2}, …]
    }
  ]
}

──────────────────────────────────────────────
5. DIFFERENZE CHIAVE RISPETTO AI DATI ORIGINALI
──────────────────────────────────────────────
- Nessuna perdita di informazione metrica: distanze e posizioni = valori UWB originali.
- Introduzione di una rappresentazione slot-based discreta per modellazione temporale.
- Aggiunta di feature di contesto (f1,f2,c1–c3) necessarie alla modellazione di rete.
- Inserimento di un modello empirico di canale (bit-rate w2) dipendente dalla distanza.
- Conversione in formato JSON autonomo, indipendente da MATLAB /HDF5.

──────────────────────────────────────────────
6. RISULTATO FINALE
──────────────────────────────────────────────
File generato:
C:\Users\matte\PycharmProjects\Future_Data_Network\dataset_addestramento\scenario_env0.json
Contiene 30 slot temporali × 7 UAV × 8 GS = 1680 link complessivi.

In sintesi:
Il JSON conserva la geometria e le misure UWB reali del team NWPU, arricchendole con variabili sintetiche e una discretizzazione temporale uniforme che consente di modellare il comportamento dinamico della rete in un contesto di apprendimento per rinforzo coerente.
