# Cluj Road Intelligence System

> **Automated urban road damage detection, classification, and prioritization using computer vision and machine learning — built for Cluj-Napoca, Romania.**

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL+PostGIS-16-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)

*Bachelor's Thesis — Faculty of Mathematics and Computer Science, Babeș-Bolyai University Cluj-Napoca*

</div>

---

## Overview

Cluj Road Intelligence System (CRIS) is an end-to-end urban infrastructure monitoring platform that automatically detects, classifies, and prioritizes road damage from smartphone dashcam footage. The system processes raw video surveys of Cluj-Napoca streets, runs a 9-stage machine learning pipeline on each frame, and presents actionable findings through an interactive municipal dashboard — designed to assist Cluj-Napoca City Hall in data-driven road maintenance planning.

The pipeline handles everything from raw `.mp4` + `.gpx` input to a live map of georeferenced damage detections, each enriched with 26 features covering geometry, depth, severity, lighting, weather, and infrastructure context.

---

## Motivation

Romania has one of the highest road accident rates in the European Union, with deteriorating urban road infrastructure being a significant contributing factor. Traditional road condition surveys in Cluj-Napoca rely on manual inspection — expensive, infrequent, and subjective. A single survey of the entire city road network can take months and cost tens of thousands of euros.

This project proposes an automated alternative: a dashcam mounted on any vehicle, a GPS logger, and a laptop. The system processes the footage overnight and delivers a georeferenced damage map with severity scores and repair priority rankings — ready for municipal decision-making the next morning.

---

## System Architecture

The system is organized into 9 sequential pipeline stages, a machine learning training subsystem, a REST API backend, and a React dashboard.

```
┌─────────────────────────────────────────────────────────────────┐
│                        SURVEY PHASE                             │
│              (runs on each new footage collection)              │
└──────────────────────────────┬──────────────────────────────────┘
                               │
           ┌───────────────────▼───────────────────┐
           │  PRE-SURVEY: ACO Route Planning        │
           │  Ant Colony Optimization finds the     │
           │  optimal driving route through         │
           │  Cluj-Napoca road network (OSM)        │
           └───────────────────┬───────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  STAGE 1 — Preprocessing                  [preprocessor.py]     │
│  • Extract 1 frame per 0.5 seconds (OpenCV)                     │
│  • Sync GPS coordinates to each frame timestamp (.gpx)          │
│  • Compute sun angle per frame (pysolar)                        │
│  • Classify lighting: daylight / overcast / low_light           │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  STAGE 2 — Detection                         [detector.py]      │
│  • RT-DETR-L inference on each frame                            │
│  • Test Time Augmentation (flip + rotate, averaged predictions) │
│  • Confidence filter: discard detections < 0.5                  │
│  • 5 damage classes: longitudinal_crack, transverse_crack,      │
│    alligator_crack, pothole, patch_deterioration                │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  STAGE 3 — Segmentation                     [segmentor.py]      │
│  • RT-DETR bounding boxes → SAM prompts                         │
│  • SAM outputs pixel-level mask per detection                   │
│  • Computes: surface_area, edge_sharpness,                      │
│    interior_contrast, mask_compactness                          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  STAGE 4 — Depth Estimation           [depth_estimator.py]      │
│  PRIMARY:  EfficientNet-B3 regression → depth in cm             │
│            + Monodepth2 dense depth map → averaged              │
│  FALLBACK: mask geometry proxy (low_light / low confidence)     │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  STAGE 5 — Severity Classification   [severity_classifier.py]   │
│  • XGBoost on WOA-selected feature subset                       │
│  • Output: severity S1 (superficial) → S5 (critical)           │
│  • Fallback: rule-based (depth + area) until Cluj data ready    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  STAGE 6 — Feature Enrichment               [enricher.py]       │
│  • Nominatim API      → street_name                             │
│  • OSM Overpass API   → road_importance, infra_proximity        │
│  • Open-Meteo API     → weather at detection timestamp          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  STAGE 7 — Spatial Deduplication          [deduplicator.py]     │
│  • DBSCAN clustering (2m radius) — merges duplicate detections  │
│  • PostGIS upsert: UPDATE existing or INSERT new record         │
│  • Tracks deterioration_rate across survey runs                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  STAGE 8 — Database Storage       [PostgreSQL + PostGIS]        │
│  • 26-column detections table with GIST spatial index           │
│  • Full temporal tracking across multiple survey runs           │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│  STAGE 9 — Dashboard              [FastAPI + React + Leaflet]   │
│  • Interactive map with severity-coded pins                     │
│  • Heatmap overlay, filters, priority repair list               │
│  • Detail panel with all 26 features per detection              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Machine Learning Stack

### Detection — RT-DETR-L

The core detection model is **RT-DETR-L** (Real-Time Detection Transformer, Large variant), a transformer-based object detector that outperforms YOLO-series models on accuracy while maintaining real-time inference speed. RT-DETR replaces the traditional anchor-based detection head with a transformer decoder, eliminating the need for hand-crafted anchor configurations and NMS post-processing.

| Property | Value |
|---|---|
| Architecture | HGStem backbone + AIFI encoder + RepC3 neck + RTDETRDecoder |
| Parameters | 32.8M |
| Input resolution | 640 × 640 |
| Pretrained weights | COCO (80 classes) |
| Fine-tuned on | RDD2022 + Pothole600 (27,336 images) |
| Output classes | 5 road damage types |
| Training hardware | NVIDIA GeForce RTX 2050 (4 GB VRAM) |
| Training technique | Two-phase: frozen backbone → full fine-tune |

**Training dataset composition:**

| Dataset | Images | Annotations | Source |
|---|---|---|---|
| RDD2022 | 21,479 | 33,913 | India, Japan, Norway, China, USA |
| Pothole600 | 5,857 | 8,970 | Mixed international |
| **Total** | **27,336** | **42,883** | 6 countries |

**Training techniques applied:**
- Focal Loss (built into RT-DETR, γ=2.0) for class imbalance
- Stochastic Weight Averaging (SWA) over last 5 checkpoints
- Mixup augmentation (α=0.15)
- Mosaic augmentation
- Albumentations pipeline (HSV jitter, random rotation, scale, shear)
- Test Time Augmentation at inference (flip + rotate)
- fp16 AMP (mandatory for 4 GB VRAM)
- Gradient accumulation ×4 (effective batch = 16)
- Two-phase training: 10 frozen epochs + 50 fine-tune epochs

**Hyperparameter optimization — PSO:**

After baseline training, **Particle Swarm Optimization** searches for optimal hyperparameters across a 7-dimensional space (`lr0`, `weight_decay`, `warmup_epochs`, `mosaic`, `mixup`, `box`, `cls`) with 15 particles over 10 iterations. Each particle fitness is evaluated by training for 5 epochs and measuring validation mAP50-95. The best configuration is saved to `ml/optimization/pso_best.json` and automatically loaded on the next training run.

### Segmentation — SAM (Segment Anything Model)

Meta's SAM is used zero-shot — no training required. RT-DETR bounding boxes are passed as box prompts, and SAM produces pixel-level masks for each detection. Four geometry features are computed from each mask:

- `surface_area` — damage extent in cm² (pixel count × camera calibration factor)
- `edge_sharpness` — Sobel gradient magnitude along mask boundary
- `interior_contrast` — mean pixel intensity difference inside vs. outside mask
- `mask_compactness` — 4π × area / perimeter² (circle = 1.0, linear crack ≈ 0.05)

### Depth Estimation — EfficientNet-B3 + Monodepth2

Depth estimation operates in two complementary paths:

**Primary path (daylight / overcast):**
- EfficientNet-B3 regression head takes the cropped detection region + sun angle → outputs `depth_estimate` in cm
- Monodepth2 self-supervised monocular depth estimation produces a dense depth map of the full frame
- Both estimates are averaged for the final `depth_estimate`

**Fallback path (low_light / depth_confidence < 0.4):**
- Proxy depth inferred from mask geometry: `edge_sharpness × interior_contrast × log(surface_area)`

EfficientNet-B3 is trained on Cluj-specific ground truth data (real pothole depth measurements + Blender synthetic renders). Hyperparameters are optimized with **Optuna** (TPE sampler, 50 trials).

### Severity Classification — XGBoost + WOA

**Whale Optimization Algorithm** performs binary feature selection across the 16-feature vector, identifying the optimal subset for XGBoost classification. Typical WOA output retains 10-12 of the 16 features, discarding noisy or redundant ones.

XGBoost then classifies each detection into one of five severity levels:

| Level | Description | Typical depth | Action |
|---|---|---|---|
| S1 | Superficial — hairline cracks | < 1 cm | Monitor |
| S2 | Minor — visible surface damage | 1–3 cm | Schedule maintenance |
| S3 | Moderate — structural concern | 3–6 cm | Priority repair |
| S4 | Severe — immediate safety risk | 6–10 cm | Urgent repair |
| S5 | Critical — road impassable | > 10 cm | Emergency closure |

### Route Optimization — ACO

Before each survey drive, **Ant Colony Optimization** computes the optimal route through the Cluj-Napoca road network loaded from OpenStreetMap via `osmnx`. The colony finds the minimum-distance path that covers all primary and secondary roads, minimizing total survey driving time while ensuring complete coverage.

---

## Detection Classes

| ID | Class | Description |
|---|---|---|
| 0 | `longitudinal_crack` | Cracks running parallel to road direction |
| 1 | `transverse_crack` | Cracks perpendicular to road direction |
| 2 | `alligator_crack` | Interconnected crack networks (fatigue damage) |
| 3 | `pothole` | Bowl-shaped depressions with depth |
| 4 | `patch_deterioration` | Degraded previously-repaired sections |

---

## Database Schema

Every detection is stored in PostgreSQL + PostGIS with 26 columns across 8 feature groups:

```sql
CREATE TYPE damage_type_enum AS ENUM (
    'longitudinal_crack', 'transverse_crack', 'alligator_crack',
    'pothole', 'patch_deterioration'
);
CREATE TYPE severity_enum    AS ENUM ('S1', 'S2', 'S3', 'S4', 'S5');
CREATE TYPE lighting_enum    AS ENUM ('daylight', 'overcast', 'low_light');

CREATE TABLE detections (
    -- Identity
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    geom                  GEOMETRY(POINT, 4326),        -- PostGIS spatial point

    -- Group 1: Detection Core (RT-DETR)
    damage_type           damage_type_enum,
    confidence            FLOAT,

    -- Group 2: Geometry (SAM)
    surface_area          FLOAT,                        -- cm²
    edge_sharpness        FLOAT,
    interior_contrast     FLOAT,
    mask_compactness      FLOAT,

    -- Group 3: Depth + Severity (EfficientNet-B3 + Monodepth2 + XGBoost)
    depth_estimate        FLOAT,                        -- cm
    depth_confidence      FLOAT,
    severity              severity_enum,
    severity_confidence   FLOAT,

    -- Group 4: Lighting (preprocessor)
    lighting_condition    lighting_enum,
    shadow_geometry_score FLOAT,

    -- Group 5: Location Context (OSM + Nominatim)
    street_name           VARCHAR,
    road_importance       SMALLINT,                     -- 1=residential 2=secondary 3=primary
    infra_proximity       FLOAT,                        -- meters to nearest school/hospital/stop

    -- Group 6: Weather (Open-Meteo)
    weather               JSONB,                        -- {temp, precipitation, wind, condition}

    -- Group 7: Temporal Tracking
    first_detection_date  TIMESTAMP,
    last_detection_date   TIMESTAMP,
    detection_count       INT DEFAULT 1,
    deterioration_rate    FLOAT DEFAULT 0.0,            -- severity change per day

    -- Group 8: Derived
    surrounding_density   INT DEFAULT 0,                -- detections within 50m
    priority_score        FLOAT                         -- repair priority ranking
);

CREATE INDEX detections_geom_idx ON detections USING GIST(geom);
```

**Priority score formula:**
```
priority_score = (severity_weight × 3.0)
               + (road_importance × 2.0)
               + (1 / max(infra_proximity, 1) × 1.5)
               + (deterioration_rate × 2.5)
               + (log(detection_count + 1) × 1.0)
```

---

## Project Structure

```
Cluj-Road-Intelligence-System/
│
├── ml/
│   ├── detection/
│   │   ├── train.py                ✅ RT-DETR-L training (two-phase + SWA)
│   │   ├── evaluate.py             ✅ Per-class AP, mAP50-95, checkpoint comparison
│   │   ├── monitor.py              ✅ Live training dashboard (reads results.csv)
│   │   └── data_prep/
│   │       └── coco_to_yolo.py     ✅ COCO JSON → YOLO .txt label conversion
│   │
│   ├── depth/
│   │   └── train.py                ⬜ EfficientNet-B3 depth regression training
│   │
│   ├── severity/
│   │   ├── train_xgboost.py        ⬜ XGBoost severity classifier training
│   │   └── feature_selection_woa.py ⬜ WOA feature selection
│   │
│   ├── optimization/
│   │   ├── pso_hyperparams.py      ✅ PSO hyperparameter search (7-dim, 15 particles)
│   │   ├── optuna_search.py        ⬜ Optuna HPO for EfficientNet-B3 + XGBoost
│   │   ├── woa_features.py         ⬜ WOA feature selection wrapper
│   │   └── aco_route.py            ⬜ ACO optimal survey route (osmnx + Cluj OSM)
│   │
│   └── weights/
│       ├── rtdetr_l_rdd2022.pt     🔄 Training in progress (RDD2022 + Pothole600)
│       ├── rtdetr_l_cluj.pt        ⬜ Future: fine-tuned on Cluj footage
│       ├── depth_effnet.pt         ⬜ Future: EfficientNet-B3 depth model
│       └── xgboost_severity.json   ⬜ Future: XGBoost severity classifier
│
├── pipeline/
│   ├── preprocessor.py             ⬜ Frame extraction + GPS sync + lighting
│   ├── detector.py                 ⬜ RT-DETR inference + TTA
│   ├── segmentor.py                ⬜ SAM mask generation + geometry features
│   ├── depth_estimator.py          ⬜ EfficientNet-B3 + Monodepth2 fusion
│   ├── severity_classifier.py      ⬜ XGBoost inference (rule-based fallback)
│   ├── enricher.py                 ⬜ OSM + Nominatim + Open-Meteo APIs
│   ├── deduplicator.py             ⬜ DBSCAN + PostGIS spatial upsert
│   └── orchestrator.py             ⬜ End-to-end survey run coordinator
│
├── backend/
│   ├── core/
│   │   ├── config.py               ⬜ Settings + DB connection string
│   │   └── database.py             ⬜ SQLAlchemy + PostGIS async session
│   ├── models/
│   │   └── detection.py            ⬜ SQLAlchemy ORM model
│   ├── schemas/
│   │   └── detection.py            ⬜ Pydantic response schemas
│   ├── api/routes/
│   │   ├── detections.py           ⬜ GET /detections, /detections/{id}, /nearby
│   │   ├── stats.py                ⬜ GET /stats, /heatmap, /priority-list
│   │   └── processing.py           ⬜ POST /process (trigger survey run)
│   └── main.py                     ⬜ FastAPI app + CORS
│
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── Map/                ⬜ Leaflet map + severity pins + heatmap
│       │   ├── Sidebar/            ⬜ Stats panel + filters + priority list
│       │   └── DetailPanel/        ⬜ Per-detection feature display
│       └── App.jsx                 ⬜ Root component
│
├── data/
│   ├── datasets/
│   │   ├── rdd2022/                ✅ 21,479 images (India, Japan, Norway, China)
│   │   ├── pothole600/             ✅ 5,857 images
│   │   └── cluj/                   ⬜ Future: locally collected footage
│   └── detection/
│       ├── dataset.yaml            ✅ Ultralytics training config
│       ├── train_images.txt        ✅ 27,336 training image paths
│       ├── val_images.txt          ✅ 5,857 validation image paths
│       └── test_images.txt         ✅ 5,857 test image paths
│
├── scripts/
│   ├── setup_db.py                 ⬜ Create PostgreSQL schema + enums + GIST index
│   └── run_survey.py               ⬜ CLI entrypoint for full survey pipeline
│
├── docker-compose.yml              ⬜ PostgreSQL + PostGIS + pgAdmin
└── requirements.txt                ✅
```

**Legend:** ✅ Implemented · 🔄 In progress · ⬜ Planned

---

## Current Training Status

RT-DETR-L is currently training on the merged RDD2022 + Pothole600 dataset. Phase 1 (frozen backbone, 10 epochs) completed successfully.

**Phase 1 results (epochs 1–10):**

| Epoch | GIoU Loss ↓ | Cls Loss | L1 Loss ↓ | Recall ↑ | mAP50 ↑ | mAP50-95 ↑ |
|---|---|---|---|---|---|---|
| 1 | 1.418 | 0.827 | 1.033 | 0.201 | 0.00248 | 0.000631 |
| 3 | 1.157 | 0.768 | 0.721 | 0.335 | 0.0130 | 0.00398 |
| 5 | 1.028 | 0.852 | 0.609 | 0.447 | 0.0204 | 0.00690 |
| 7 | 0.977 | 0.892 | 0.569 | 0.500 | 0.0235 | 0.00845 |
| 10 | 0.942 | 0.920 | 0.546 | 0.533 | 0.0266 | 0.00992 |

Phase 2 (full fine-tune, 50 epochs, backbone unfrozen, LR ×0.1) is running. Significant mAP gains expected from epoch 15 onward as backbone features adapt to road damage domain.

---

## Planned: Cluj Data Collection & Fine-tuning

The current model is bootstrapped on international public datasets. The next phase involves collecting domain-specific data from Cluj-Napoca streets:

### Data Collection Protocol

1. **Equipment:**
   - Smartphone dashcam mounted on windshield
   - GPS logger recording `.gpx` track
   - Measuring tape for ground truth depth measurements

2. **Survey route:** Generated by ACO over the Cluj-Napoca OSM road network, covering all primary and secondary roads.

3. **Annotation:** Collected frames labeled in **Label Studio** with:
   - Bounding box annotations for all 5 damage classes
   - Manual depth measurements (ruler + photo) for pothole depth ground truth
   - Severity labels (S1–S5) based on engineering assessment

4. **Fine-tuning pipeline:**
   ```
   rtdetr_l_rdd2022.pt  →  fine-tune on Cluj frames  →  rtdetr_l_cluj.pt
   EfficientNet-B3       →  train on depth measurements + Blender renders
   XGBoost               →  train on Cluj feature vectors + severity labels
   ```

### Expected improvements from Cluj fine-tuning:
- Domain adaptation to Romanian asphalt types and road geometry
- Improved performance on local weather conditions (rain, snow, fog)
- Calibrated depth estimates using real measurement data
- Severity classifier trained on local engineering standards

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/detections` | All detections, paginated + filterable |
| `GET` | `/detections/{id}` | Single detection with all 26 features |
| `GET` | `/detections/nearby` | Detections within radius of coordinates |
| `GET` | `/stats` | City-wide counts by type and severity |
| `GET` | `/heatmap` | Density grid for map overlay |
| `GET` | `/priority-list` | Ranked repair list sorted by priority_score |
| `POST` | `/process` | Trigger processing of new survey footage |

---

## Dashboard Features

The React + Leaflet dashboard provides municipal decision-makers with:

**Map View (center)**
- Color-coded damage pins by severity: S1=yellow · S2=orange · S3=red · S4=darkred · S5=black
- Heatmap overlay showing damage density across the city
- Click any pin to open the detail panel

**Sidebar (left)**
- City-wide statistics (total detections, breakdown by type and severity)
- Filter controls: damage type / severity / date range / street / road importance
- Priority repair list ranked by `priority_score`

**Detail Panel (right, on pin click)**
- Cropped pothole photo from original dashcam frame
- All 26 features displayed with units
- Detection history (first seen, last seen, total count)
- Severity badge and priority score
- Deterioration rate trend

---

## Key Papers

| Concept | Paper | Year |
|---|---|---|
| RT-DETR | DETRs Beat YOLOs on Real-time Object Detection | 2023 |
| RT-DETRv2 | RT-DETRv2: Improved Baseline | 2024 |
| SAM | Segment Anything | 2023 |
| SAM 2 | SAM 2: Segment Anything in Images and Videos | 2024 |
| EfficientNet | EfficientNet: Rethinking Model Scaling | 2019 |
| Monodepth2 | Digging into Self-Supervised Monocular Depth Estimation | 2019 |
| XGBoost | XGBoost: A Scalable Tree Boosting System | 2016 |
| PSO | Particle Swarm Optimization | 1995 |
| WOA | The Whale Optimization Algorithm | 2016 |
| ACO | Ant Colony System | 1996 |
| Optuna | Optuna: A Next-generation Hyperparameter Optimization Framework | 2019 |
| Focal Loss | Focal Loss for Dense Object Detection | 2017 |
| FPN | Feature Pyramid Networks for Object Detection | 2017 |
| SWA | Averaging Weights Leads to Wider Optima | 2018 |
| Albumentations | Albumentations: Fast and Flexible Image Augmentations | 2020 |
| DBSCAN | A Density-Based Algorithm for Discovering Clusters | 1996 |
| RDD2022 | RDD2022: A Multi-National Image Dataset for Automatic Road Damage Detection | 2024 |

---

## Technology Stack

| Layer | Technology |
|---|---|
| Detection model | RT-DETR-L (Ultralytics) |
| Segmentation | SAM — Segment Anything Model (Meta AI) |
| Depth estimation | EfficientNet-B3 + Monodepth2 |
| Severity classification | XGBoost |
| Hyperparameter optimization | PSO (custom) + Optuna |
| Feature selection | WOA (Whale Optimization Algorithm) |
| Route planning | ACO (Ant Colony Optimization) + osmnx |
| Spatial clustering | DBSCAN (scikit-learn) |
| Training techniques | Focal Loss, SWA, Mixup, Albumentations, TTA, fp16 AMP |
| Database | PostgreSQL 16 + PostGIS 3.4 |
| ORM | SQLAlchemy 2.0 (async) |
| Backend | FastAPI + Pydantic v2 |
| Frontend | React 18 + Leaflet.js + OpenStreetMap |
| Containerization | Docker Compose |
| Geocoding | Nominatim API (OpenStreetMap) |
| Road network | OSM Overpass API |
| Weather | Open-Meteo API (historical + real-time) |
| Sun angle | pysolar |
| Annotation | Label Studio |
| Language | Python 3.12 |

---

## Roadmap

- [x] Dataset preparation (RDD2022 + Pothole600 merge, COCO → YOLO conversion)
- [x] RT-DETR-L training pipeline with two-phase strategy
- [x] PSO hyperparameter optimization script
- [x] Evaluation script with per-class AP and checkpoint comparison
- [x] Live training monitor dashboard
- [ ] Phase 2 training completion + PSO-optimized retraining
- [ ] Inference pipeline (all 8 stage modules)
- [ ] PostgreSQL + PostGIS Docker setup
- [ ] FastAPI backend (all endpoints)
- [ ] React dashboard (map + sidebar + detail panel)
- [ ] Cluj-Napoca data collection drive
- [ ] Label Studio annotation of Cluj frames
- [ ] EfficientNet-B3 depth model training on Cluj measurements
- [ ] XGBoost severity classifier training with WOA feature selection
- [ ] Fine-tuning RT-DETR on Cluj-specific footage
- [ ] ACO optimal survey route generation
- [ ] End-to-end integration test on real Cluj footage
- [ ] City Hall pilot demonstration

---

## Setup

### Prerequisites

```bash
Python 3.12
CUDA 11.8+ with compatible NVIDIA GPU
Docker Desktop (for PostgreSQL + PostGIS)
Node.js 18+ (for React frontend)
```

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/cluj-road-intelligence-system.git
cd cluj-road-intelligence-system

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download pretrained RT-DETR-L weights
# Place rtdetr-l.pt in project root

# Prepare dataset (run once after downloading RDD2022 + Pothole600)
python ml/detection/data_prep/coco_to_yolo.py

# Start training
python ml/detection/train.py

# Monitor training (in a second terminal)
python ml/detection/monitor.py --live
```

### Database Setup (when pipeline is ready)

```bash
# Start PostgreSQL + PostGIS
docker-compose up -d

# Initialize schema
python scripts/setup_db.py
```

### Running a Survey

```bash
# Process raw dashcam footage
python scripts/run_survey.py --video data/raw/footage/survey_01.mp4 \
                             --gps   data/raw/gps_logs/survey_01.gpx

# Start backend API
uvicorn backend.main:app --reload

# Start frontend
cd frontend && npm run dev
```

---

## License

This project is developed as a Bachelor's thesis at Babeș-Bolyai University Cluj-Napoca, Faculty of Mathematics and Computer Science. All rights reserved.

Dataset citations: RDD2022 (Arya et al., 2024), Pothole600.
Model citations: RT-DETR (Zhao et al., 2023), SAM (Kirillov et al., 2023), Monodepth2 (Godard et al., 2019).

---

<div align="center">
<i>Built for Cluj-Napoca · Babeș-Bolyai University · 2025</i>
</div>