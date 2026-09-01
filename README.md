# Verit — Network IDS Pipeline

An end-to-end NIDS: live tcpdump capture → packet-level cleaning → flow
feature extraction → data cleaning/scaling/encoding → dual-model detection
(Autoencoder for zero-day, XGBoost for known attacks) → drop/alert on
malicious flows.

## Project layout

```
Verit/
├── config/config.yaml           # every tunable in one place
├── requirements.txt
├── main.py                      # CLI entry point
├── src/
│   ├── capture/                 # Step 1: tcpdump multi-NIC capture
│   │   ├── nic_discovery.py
│   │   └── tcpdump_capture.py
│   ├── preprocessing/           # Step 2: packet-level pre-cleaning
│   │   └── packet_cleaning.py
│   ├── features/                # Step 3: PCAP -> flow feature table
│   │   └── flow_extractor.py
│   ├── processing/               # Step 4: NaN/Inf, identity drop, zero-var, dedup, scaling
│   │   ├── data_cleaning.py
│   │   └── scaling.py
│   ├── encoding/                 # Steps 5-6: categorical + label encoding
│   │   ├── categorical_encoder.py
│   │   └── label_encoder.py
│   ├── models/                   # Step 7: Autoencoder + XGBoost + ensemble
│   │   ├── autoencoder.py
│   │   ├── xgboost_model.py
│   │   └── ensemble_detector.py
│   ├── training/                 # Step 8: split, train, evaluate
│   │   ├── train_pipeline.py
│   │   └── evaluate.py
│   ├── pipeline/                 # Live orchestration (capture -> ... -> drop)
│   │   └── realtime_pipeline.py
│   └── utils/config_loader.py
└── data/
    ├── raw_pcap/                 # tcpdump output
    ├── cleaned_pcap/             # post pre-cleaning
    ├── features/                 # flow feature CSVs + live detections
    └── models/                   # trained artifacts + evaluation reports
```

## Setup on your machine (`/home/rama/Desktop/Verit`)

```bash
# copy this project to your machine, then:
cd /home/rama/Desktop/Verit
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# tcpdump needs raw socket capability without full root, if preferred:
sudo setcap cap_net_raw,cap_net_admin=eip $(which tcpdump)
```

`config/config.yaml` already points `training.dataset_path` at
`/home/rama/Desktop/Verit/Database/all_data_combined.csv` — update the
`label_column` name in the config if your CSV's ground-truth column isn't
literally called `label`.

## Usage

**1. Train the models** (run this first — capture/live depend on the saved artifacts):
```bash
python main.py train
```
This will: load the CSV, clean it, encode categoricals/labels, do a
stratified 70/15/15 train/val/test split, fit the scaler on train only,
train XGBoost (multiclass) and the Autoencoder (benign-only), then evaluate
XGBoost, the Autoencoder, and the combined ensemble on the held-out test
set — saving confusion matrices, ROC curves, and text reports to
`data/models/reports/`.

**2. Capture live traffic** (needs root or tcpdump capability):
```bash
sudo python main.py capture
```
Auto-discovers every live/UP NIC and starts a rotating tcpdump per
interface into `data/raw_pcap/`.

**3. Run live detection** (in a second terminal, after training + while capture is running):
```bash
python main.py live
```
Watches `data/raw_pcap/` for newly rotated files, runs each through
cleaning → feature extraction → detection, and logs/drops malicious flows.
Enforcement (actually blocking traffic) is stubbed in
`src/pipeline/realtime_pipeline.py::enforce_drop()` — wire it to
iptables/nftables once you're ready to move from alert-only to
block-on-detect.

**Manual/debug commands:**
```bash
python main.py clean data/raw_pcap/eth0_20260101_000000.pcap
python main.py extract data/cleaned_pcap/cleaned_eth0_20260101_000000.pcap
```

## Design notes

- **Pre-cleaning (Step 2)** operates on packets, before flow assembly, so
  corrupted/truncated/checksum-invalid/duplicate/retransmitted/L2-noise
  packets never pollute flow statistics.
- **Flow features (Step 3)** are direction-normalized 5-tuple flows with the
  4 requested feature groups: flow & volume, IAT, TCP flags, packet length
  stats.
- **Autoencoder** is trained *only on benign flows* — it flags anything
  whose reconstruction error is unusual relative to normal traffic,
  independent of whether that attack pattern was ever labeled in training
  data. That's what gives zero-day coverage.
- **XGBoost** is trained supervised on all labeled classes and is strong on
  attack types it has seen before.
- **Ensemble** defaults to OR logic (either model firing = malicious),
  which biases toward higher recall — appropriate for a defensive NIDS
  where missing an attack is costlier than an extra alert. Switch to
  `and`/`weighted` in config if you want to trade recall for precision.
- Scaler/encoders are fit on **train only** and reused (via joblib) on
  val/test and on live traffic, to avoid data leakage and keep the live
  pipeline consistent with how the models were trained.
