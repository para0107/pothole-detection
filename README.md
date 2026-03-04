# Cluj Road Intelligence System

> **Automated urban road damage detection, classification, and prioritization using computer vision and machine learning — built for Cluj-Napoca, Romania.**

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL+PostGIS-15-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgis.net)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

**Bachelor's Thesis — Babeș-Bolyai University, Faculty of Computer Science**
**Author: Paraschiv Tudor · 2026**

[GitHub Repository](https://github.com/para0107/Cluj-Road-Intelligence-System)

</div>

---

## Overview

Cluj Road Intelligence System (CRIS) is an end-to-end urban infrastructure monitoring platform that automatically detects, classifies, and prioritizes road damage from smartphone dashcam footage. The system processes raw video surveys of Cluj-Napoca streets through a 9-stage machine learning pipeline, enriches each detection with spatial, depth, severity, lighting, weather, and infrastructure context features, and exposes the results through a REST API backed by a PostGIS spatial database.

The goal is to replace expensive, infrequent, and subjective manual road inspections with a low-cost automated alternative — a dashcam, a GPS logger, and an overnight processing run.

---

## Motivation

Romania has one of the highest road accident rates in the European Union. Deteriorating urban road infrastructure is a significant contributing factor. Traditional road condition surveys in Cluj-Napoca rely on manual inspection — expensive, infrequent, and subjective. A full city-wide survey can take months.

This project proposes an automated alternative that any municipality can adopt with minimal hardware investment. Survey footage collected during normal vehicle operations can be processed automatically every night, producing a continuously updated georeferenced damage map with severity scores and ranked repair lists.

---

## System Architecture

The system has four distinct layers:

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1 — ML Training          (ml/)                           │
│  RT-DETR-L · SAM · EfficientNet-B3 · Monodepth2 · XGBoost      │
│  PSO hyperparameter search · dataset preparation                │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2 — Inference Pipeline   (pipeline/ · scripts/)          │
│  Frame extraction → Detection → Segmentation → Depth →         │
│  Severity → Enrichment → Deduplication → DB write               │
│  Triggered manually (run_survey.py) or nightly (daily_job.py)  │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3 — Backend API          (backend/)                      │
│  FastAPI · SQLAlchemy · PostGIS · Pydantic v2                   │
│  Routes: detections · stats · heatmap · priority                │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 4 — Frontend             (frontend/)                     │
│  Interactive map · Severity filters · Priority repair list      │
│  (currently placeholder — React dashboard planned)              │
└─────────────────────────────────────────────────────────────────┘
```

### Inference Pipeline — Stage by Stage

```
┌─────────────────────────────────────────────────────────────────┐
│  PRE-SURVEY: ACO Route Planning                                 │
│  Ant Colony Optimization over Cluj-Napoca OSM road network      │
│  Finds minimum-distance route covering all primary/secondary    │
│  roads before the survey drive                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STAGE 1 — Preprocessor              [preprocessor.py]          │
│  • Extract frames from .mp4 (1 per 0.5 seconds)                │
│  • Sync GPS coordinates from .gpx to each frame timestamp       │
│  • Compute sun angle per frame (pysolar)                        │
│  • Classify lighting: daylight / overcast / low_light           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STAGE 2 — Detector                  [detector.py]              │
│  • RT-DETR-L inference on each frame (640×640)                  │
│  • Test Time Augmentation: flip + rotate, averaged predictions  │
│  • Confidence threshold: discard detections < 0.5              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STAGE 3 — Segmentor                 [segmentor.py]             │
│  • RT-DETR bounding boxes → SAM box prompts                     │
│  • SAM outputs pixel-level mask per detection                   │
│  • Computes: surface_area, edge_sharpness,                      │
│    interior_contrast, mask_compactness                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STAGE 4 — Depth Estimator           [depth_estimator.py]       │
│  • EfficientNet-B3 regression → depth in cm                     │
│  • Monodepth2 dense depth map → depth at detection region       │
│  • Both estimates fused for final depth_estimate                │
│  • Fallback: mask geometry proxy when depth_confidence < 0.4   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STAGE 5 — Severity Classifier       [severity_classifier.py]   │
│  • XGBoost on WOA-selected feature subset → S1–S5               │
│  • Rule-based fallback (depth + area) until Cluj data ready     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STAGE 6 — Enricher                  [enricher.py]              │
│  • Nominatim API    → street_name                               │
│  • OSM Overpass API → road_importance, infra_proximity          │
│  • Open-Meteo API   → weather at detection timestamp            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STAGE 7 — Deduplicator              [deduplicator.py]          │
│  • DBSCAN spatial clustering (2m radius)                        │
│  • Merges duplicates from multiple survey passes                │
│  • PostGIS upsert: UPDATE existing or INSERT new                │
│  • Updates detection_count and deterioration_rate               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STAGE 8 — Database          [PostgreSQL 15 + PostGIS]          │
│  • detections table — all ML-derived attributes + geometry      │
│  • survey_log table — tracks each pipeline run                  │
│  • GIST spatial index on geom column                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STAGE 9 — Backend API + Dashboard   [FastAPI + frontend/]      │
│  • REST API reads from PostGIS                                   │
│  • Routes: detections · stats · heatmap · priority              │
│  • Daily pipeline trigger via APScheduler (02:00 Bucharest TZ)  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Machine Learning Stack

### Detection — RT-DETR-L

**RT-DETR-L** (Real-Time Detection Transformer, Large) is a transformer-based object detector that outperforms YOLO-series models on accuracy while maintaining real-time inference. It replaces the anchor-based detection head with a transformer decoder, eliminating NMS post-processing.

| Property | Value |
|---|---|
| Architecture | HGStem backbone + AIFI encoder + RepC3 neck + RTDETRDecoder |
| Parameters | 32.8M |
| Input resolution | 640 × 640 |
| Pretrained weights | COCO (80 classes) |
| Fine-tuned on | RDD2022 + Pothole600 (27,336 images) |
| Output classes | 5 road damage types |
| Training hardware | NVIDIA GeForce RTX 2050 (4 GB VRAM) |

**Training dataset:**

| Dataset | Images | Annotations | Countries |
|---|---|---|---|
| RDD2022 | 21,479 | 33,913 | India, Japan, Norway, China, USA |
| Pothole600 | 5,857 | 8,970 | Mixed |
| **Total** | **27,336** | **42,883** | 6 |

**Training strategy:**
- **Phase 1** (10 epochs): Frozen backbone — only decoder and detection head trained. LR = 1e-4.
- **Phase 2** (50 epochs): Full fine-tune — backbone unfrozen, LR = 1e-5. Early stopping patience = 20.
- **SWA**: Stochastic Weight Averaging over last 5 checkpoints → `swa.pt`.

Checkpoint priority at evaluation and inference: `swa.pt` > `best.pt` > `rtdetr_l_rdd2022.pt`.

**Training techniques:**
- Focal Loss (γ=2.0, built into RT-DETR) for class imbalance
- Mixup (α=0.15) and Mosaic augmentation
- Albumentations pipeline (HSV jitter, rotation, scale, shear, horizontal flip)
- Test Time Augmentation at inference (flip + rotate, averaged)
- fp16 AMP, gradient accumulation ×4 (effective batch = 16)
- `cudnn.benchmark = True`, `deterministic = False`

**Hyperparameter optimization — PSO:**
Particle Swarm Optimization searches a 7-dimensional space (`lr0`, `weight_decay`, `warmup_epochs`, `mosaic`, `mixup`, `box`, `cls`) with 15 particles over 10 iterations. Each particle is evaluated by training for 5 epochs and measuring validation mAP50-95. Output saved to `ml/optimization/pso_best.json` — `train.py` loads this automatically on the next run with no code changes needed.

### Segmentation — SAM

Used **zero-shot** — no fine-tuning required. RT-DETR bounding boxes serve as box prompts. Four geometry features computed from each mask:

| Feature | Description |
|---|---|
| `surface_area` | Damage extent in cm² |
| `edge_sharpness` | Sobel gradient magnitude along mask boundary |
| `interior_contrast` | Mean pixel intensity inside vs. outside mask |
| `mask_compactness` | 4π × area / perimeter² (circle=1.0, crack≈0.05) |

### Depth Estimation — EfficientNet-B3 + Monodepth2

- **EfficientNet-B3**: regression head on cropped detection region + sun angle → depth in cm. Trained on Cluj ground truth measurements + Blender synthetic renders. Tuned with Optuna (TPE sampler, 50 trials).
- **Monodepth2**: self-supervised dense depth map — depth at detection region extracted and fused with EfficientNet estimate.
- **Fallback**: proxy depth from mask geometry when `depth_confidence < 0.4` or `lighting_condition = low_light`.

### Severity Classification — XGBoost + WOA

**Whale Optimization Algorithm** performs binary feature selection across the 16-feature ML vector before XGBoost training.

| Level | Description | Typical depth | Action |
|---|---|---|---|
| S1 | Superficial | < 1 cm | Monitor |
| S2 | Minor | 1–3 cm | Schedule maintenance |
| S3 | Moderate | 3–6 cm | Priority repair |
| S4 | Severe | 6–10 cm | Urgent repair |
| S5 | Critical | > 10 cm | Emergency closure |

### Route Optimization — ACO

**Ant Colony Optimization** computes the optimal pre-survey driving route through the Cluj-Napoca OSM road network (loaded via `osmnx`), minimizing total distance while covering all primary and secondary roads.

---

## Detection Classes

| ID | Class | Description |
|---|---|---|
| 0 | `longitudinal_crack` | Cracks parallel to road direction |
| 1 | `transverse_crack` | Cracks perpendicular to road direction |
| 2 | `alligator_crack` | Interconnected crack networks (fatigue damage) |
| 3 | `pothole` | Bowl-shaped depressions with measurable depth |
| 4 | `patch_deterioration` | Degraded previously-repaired sections |

**Class distribution in training set:**

| Class | Instances | Share |
|---|---|---|
| longitudinal_crack | 18,201 | 42.4% |
| pothole | 8,770 | 20.5% |
| transverse_crack | 8,386 | 19.6% |
| alligator_crack | 7,526 | 17.5% |
| patch_deterioration | 0 | Reserved for Cluj data |

---

## Database

PostgreSQL 15 + PostGIS runs in Docker. Two tables:

**`detections`** — one row per unique georeferenced damage instance:
- Spatial geometry (`POINT`, SRID 4326) with GIST index
- All ML-derived attributes: `damage_type`, `confidence`, SAM geometry features, `depth_estimate`, `depth_confidence`, `severity`, `severity_confidence`
- Lighting, weather (JSONB), location context from OSM
- Temporal tracking: `first_detected`, `last_detected`, `detection_count`, `deterioration_rate`
- Derived: `surrounding_density`, `priority_score`

**`survey_log`** — one row per pipeline run, tracking input footage, timestamps, detection counts, and processing status.

**Priority score formula** (implemented in `Detection.compute_priority_score()`):
```
priority_score = severity_weight × road_weight × infra_weight × log(detection_count + 1)
```

**DB session access:**
- `get_db()` — FastAPI `Depends()` injection for route handlers
- `get_db_session()` — direct async context manager for pipeline scripts

---

## Project Structure

```
Cluj-Road-Intelligence-System/
│
├── ml/
│   ├── detection/
│   │   ├── train.py                ✅ RT-DETR-L two-phase training + SWA
│   │   ├── evaluate.py             ✅ Per-class AP, mAP50-95, checkpoint comparison
│   │   ├── monitor.py              ✅ Live training dashboard (reads results.csv)
│   │   └── data_prep/
│   │       ├── prep_rdd2022.py     ✅ RDD2022 format conversion
│   │       ├── prep_pothole600.py  ✅ Pothole600 format conversion
│   │       ├── merge_datasets.py   ✅ Merge into unified train/val/test split
│   │       └── coco_to_yolo.py     ✅ COCO JSON → YOLO .txt conversion
│   ├── optimization/
│   │   └── pso_hyperparams.py      ✅ PSO search (7-dim, 15 particles × 10 iters)
│   ├── segmentation/               ⬜ SAM inference module
│   ├── depth/                      ⬜ EfficientNet-B3 depth training
│   ├── severity/                   ⬜ XGBoost + WOA feature selection
│   └── weights/
│       ├── rtdetr_l_rdd2022.pt     🔄 Training in progress
│       ├── rtdetr_l_cluj.pt        ⬜ Future: fine-tuned on Cluj footage
│       ├── depth_effnet.pt         ⬜ Future: EfficientNet-B3 depth model
│       └── xgboost_severity.json   ⬜ Future: XGBoost classifier
│
├── pipeline/
│   ├── preprocessor.py             ⬜ Frame extraction + GPS sync + lighting
│   ├── detector.py                 ⬜ RT-DETR inference + TTA
│   ├── segmentor.py                ⬜ SAM masks + geometry features
│   ├── depth_estimator.py          ⬜ EfficientNet-B3 + Monodepth2
│   ├── severity_classifier.py      ⬜ XGBoost (rule-based fallback)
│   ├── enricher.py                 ⬜ OSM + Nominatim + Open-Meteo
│   ├── deduplicator.py             ⬜ DBSCAN + PostGIS upsert
│   └── orchestrator.py             ⬜ End-to-end coordinator
│
├── backend/
│   ├── main.py                     ✅ FastAPI app + CORS
│   ├── database.py                 ✅ SQLAlchemy engine, get_db(), get_db_session()
│   ├── models.py                   ✅ Detection + SurveyLog ORM models
│   ├── schemas.py                  ✅ Pydantic v2 schemas
│   └── routes/
│       ├── detections.py           ✅ GET /detections, /{id}, /nearby
│       ├── stats.py                ✅ GET /stats
│       ├── heatmap.py              ✅ GET /heatmap
│       └── priority.py             ✅ GET /priority-list
│
├── scheduler/
│   └── daily_job.py                ✅ APScheduler cron — fires pipeline nightly
│                                      (Europe/Bucharest TZ)
│
├── scripts/
│   ├── download_datasets.py        ✅ Download RDD2022 + Pothole600
│   ├── inspect_datasets.py         ✅ Distribution analysis and plots
│   ├── verify_merge.py             ✅ Verify merged dataset integrity
│   └── run_survey.py               ✅ Manual one-shot pipeline trigger
│
├── frontend/
│   └── (placeholder)               ⬜ React dashboard planned
│
├── data/
│   ├── raw/
│   │   ├── footage/                Input dashcam .mp4 files
│   │   └── gps_logs/               GPX telemetry files
│   ├── processed/
│   │   ├── frames/                 Extracted video frames
│   │   └── metadata/               Per-frame annotation metadata
│   ├── datasets/                   Prepared training/evaluation datasets
│   └── detection/
│       ├── dataset.yaml            ✅ YOLO-format dataset config
│       ├── train.json              ✅ 27,336 images
│       ├── val.json                ✅ 5,857 images
│       └── test.json               ✅ 5,857 images
│
├── docker-compose.yml              ✅ PostgreSQL 15 + PostGIS + pgAdmin
└── requirements.txt                ✅
```

**Legend:** ✅ Done · 🔄 In progress · ⬜ Planned

---

## Current Training Status

Phase 1 (10 frozen epochs) complete. Phase 2 (50 full fine-tune epochs) running.

| Epoch | GIoU ↓ | L1 ↓ | Recall ↑ | mAP50 ↑ | mAP50-95 ↑ |
|---|---|---|---|---|---|
| 1 | 1.418 | 1.033 | 0.201 | 0.00248 | 0.000631 |
| 3 | 1.157 | 0.721 | 0.335 | 0.0130 | 0.00398 |
| 5 | 1.028 | 0.609 | 0.447 | 0.0204 | 0.00690 |
| 7 | 0.977 | 0.569 | 0.500 | 0.0235 | 0.00845 |
| 10 | 0.942 | 0.546 | 0.533 | 0.0266 | 0.00992 |

GIoU and L1 losses decrease consistently. Recall climbs from 0.201 to 0.533 across the frozen phase. Precision reports `nan` during frozen training — expected, resolves in phase 2 as the full network adapts.

---

## Planned: Cluj Data Collection & Fine-tuning

**Equipment:** smartphone dashcam + GPS logger + measuring tape for ground truth depth.

**Survey route:** generated by ACO over the Cluj-Napoca OSM road network.

**Annotation:** Label Studio — bounding boxes for all 5 classes + manual depth measurements for EfficientNet ground truth.

**Fine-tuning pipeline:**
```
rtdetr_l_rdd2022.pt  →  Cluj frames        →  rtdetr_l_cluj.pt
EfficientNet-B3       →  depth measurements →  depth_effnet.pt
XGBoost               →  Cluj features      →  xgboost_severity.json
```

---

## Usage

### Dataset Preparation

```bash
# Download raw datasets
python scripts/download_datasets.py

# Convert and merge (run in order)
python ml/detection/data_prep/prep_rdd2022.py
python ml/detection/data_prep/prep_pothole600.py
python ml/detection/data_prep/merge_datasets.py
python ml/detection/data_prep/coco_to_yolo.py

# Verify
python scripts/inspect_datasets.py
python scripts/verify_merge.py
```

### Training

```bash
# (Optional) PSO hyperparameter search — ~9-12h on RTX 2050
python ml/optimization/pso_hyperparams.py

# Train RT-DETR-L (auto-loads pso_best.json if present)
python ml/detection/train.py
python ml/detection/train.py --smoke_test         # 2-epoch pipeline validation
python ml/detection/train.py --resume runs/detect/rtdetr_road/weights/last.pt

# Monitor in a second terminal
python ml/detection/monitor.py                    # auto-refreshes every 30s
python ml/detection/monitor.py --save             # save PNG
python ml/detection/monitor.py --interval 60
```

### Evaluation

```bash
python ml/detection/evaluate.py
python ml/detection/evaluate.py --full            # val + test + TTA + comparison
python ml/detection/evaluate.py --compare         # best.pt vs swa.pt vs last.pt
```

### Database

```bash
docker-compose up -d
python scripts/setup_db.py
```

### Backend

```bash
uvicorn backend.main:app --reload
# Swagger UI: http://localhost:8000/docs
```

### Pipeline

```bash
# Manual one-shot survey run
python scripts/run_survey.py
python scripts/run_survey.py --date 2024-06-15

# Nightly scheduler (blocks — fires at 02:00 Europe/Bucharest)
python scheduler/daily_job.py
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/detections` | All detections, paginated and filterable |
| `GET` | `/detections/{id}` | Single detection with all features |
| `GET` | `/detections/nearby` | Detections within radius of coordinates |
| `GET` | `/stats` | City-wide counts by type and severity |
| `GET` | `/heatmap` | Density grid for map overlay |
| `GET` | `/priority-list` | Ranked repair list by priority_score |
| `POST` | `/process` | Trigger processing of new survey footage |

---

## Roadmap

**Done:**
- [x] Dataset download, conversion, merge, and verification scripts
- [x] Data distribution analysis and plots
- [x] RT-DETR-L training pipeline (two-phase + SWA + PSO integration)
- [x] PSO hyperparameter optimization script
- [x] Evaluation script with per-class AP and checkpoint comparison
- [x] Live training monitor
- [x] Docker Compose — PostgreSQL 15 + PostGIS + pgAdmin
- [x] Database schema — `detections` + `survey_log` tables, enums, GIST index
- [x] SQLAlchemy ORM models + Pydantic v2 schemas
- [x] FastAPI backend — all routes (tested on Swagger)
- [x] APScheduler daily job (`scheduler/daily_job.py`)

**In progress:**
- [ ] RT-DETR-L Phase 2 training (50 epochs)
- [ ] PSO search + retraining with best hyperparameters

**Planned:**
- [ ] Inference pipeline — all 8 modules + orchestrator
- [ ] React dashboard — map, sidebar, detail panel
- [ ] ACO survey route generation
- [ ] Cluj-Napoca data collection drive
- [ ] Label Studio annotation
- [ ] EfficientNet-B3 depth model training
- [ ] XGBoost + WOA severity classifier
- [ ] RT-DETR fine-tuning on Cluj footage
- [ ] End-to-end integration test on real Cluj footage
- [ ] City Hall pilot demonstration

---

## Papers

### Object Detection & Transformers

| Paper | Year |
|---|---|
| DETRs Beat YOLOs on Real-Time Object Detection | 2023 |
| End-to-End Object Detection with Transformers (DETR) | 2020 |
| RT-DETRv2: Improved Baseline with Bag-of-Freebies | 2024 |
| RT-DETRv3: Real-Time End-to-End Object Detection with Hierarchical Dense Positive Supervision | 2024 |
| Feature Pyramid Networks for Object Detection | 2017 |
| Focal Loss for Dense Object Detection | 2017 |

### Road Damage Detection

| Paper | Year |
|---|---|
| Computer Vision for Road Imaging and Pothole Detection: A State-of-the-Art Review | — |
| Road Damage Detection and Classification Using Deep Neural Networks | — |
| Real-Time Road Damage Detection System on Deep Learning | — |
| RDD2022: A Multi-National Image Dataset for Automatic Road Damage Detection | 2024 |
| An Annotated Street View Image Dataset for Automated Road Damage Detection (SVRDD) | — |
| Robust Video-Based Pothole Detection and Area Estimation | — |
| Pothole Detection in Adverse Weather Leveraging Synthetic Images and Attention-Based Method | — |
| When Segment Anything Model Meets Inventorying of Roadway Assets | — |

### Depth Estimation

| Paper | Year |
|---|---|
| Digging into Self-Supervised Monocular Depth Estimation | 2019 |
| A Comparison of Low-Cost Monocular Vision Techniques for Pothole Distance Estimation | — |

### Segmentation

| Paper | Year |
|---|---|
| Segment Anything | 2023 |
| SAM 2: Segment Anything in Images and Videos | 2024 |

### ML / Boosting / Severity Scoring

| Paper | Year |
|---|---|
| XGBoost: A Scalable Tree Boosting System | 2016 |
| Modified XGBoost Hyper-Parameter Tuning Using Adaptive Particle Swarm Optimization | — |
| EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks | 2019 |

### Hyperparameter Optimization

| Paper | Year |
|---|---|
| Particle Swarm Optimization | 1995 |
| Particle Swarm Optimization for Hyper-Parameter Selection in Deep Neural Networks | 2015 |
| Particle Swarm Optimization-Based Automatic Parameter Tuning | — |
| A Review of Whale Optimization Algorithm for Feature Selection | — |
| Optuna: A Next-generation Hyperparameter Optimization Framework | 2019 |

### Training Techniques

| Paper | Year |
|---|---|
| Averaging Weights Leads to Wider Optima and Better Generalization (SWA) | 2018 |
| Albumentations: Fast and Flexible Image Augmentations | 2020 |

### Clustering / Spatial

| Paper | Year |
|---|---|
| A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise (DBSCAN) | 1996 |

### Routing / Optimization

| Paper | Year |
|---|---|
| Ant Colony Optimisation for Vehicle Routing Problem | 1996 |

> **Total: 31 papers.** Full citations with authors, venues, and DOIs are listed in the thesis bibliography.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Detection | RT-DETR-L (Ultralytics 8.2) |
| Segmentation | SAM — Segment Anything Model |
| Depth estimation | EfficientNet-B3 + Monodepth2 |
| Severity classification | XGBoost |
| Hyperparameter optimization | PSO (custom) · Optuna (TPE) |
| Feature selection | WOA — Whale Optimization Algorithm |
| Survey route planning | ACO · osmnx |
| Spatial clustering | DBSCAN (scikit-learn) |
| Augmentation | Albumentations · Mixup · Mosaic |
| Training | Focal Loss · SWA · TTA · fp16 AMP · gradient accumulation |
| Database | PostgreSQL 15 + PostGIS |
| ORM | SQLAlchemy 2.0 (async) |
| Backend | FastAPI + Pydantic v2 |
| Scheduler | APScheduler (Europe/Bucharest TZ) |
| Frontend | Planned: React 18 + Leaflet.js |
| Containerization | Docker Compose |
| Geocoding | Nominatim (OpenStreetMap) |
| Road network | OSM Overpass API |
| Weather | Open-Meteo API |
| Sun angle | pysolar |
| Annotation | Label Studio |
| Code style | Black |
| Language | Python 3.12 |

---

## License

Bachelor's thesis — Babeș-Bolyai University, Faculty of Computer Science, Cluj-Napoca.
**Author: Paraschiv Tudor, 2026.**

Dataset attributions: RDD2022 (Arya et al., 2024), Pothole600.
Model attributions: RT-DETR (Zhao et al., 2023), SAM (Kirillov et al., 2023), Monodepth2 (Godard et al., 2019).

---

<div align="center">
<i>Cluj-Napoca · Babeș-Bolyai University · Faculty of Computer Science · 2026</i>
</div>