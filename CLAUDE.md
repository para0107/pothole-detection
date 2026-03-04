# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Cluj Urban Monitor** — a Python-based road damage detection system for Cluj-Napoca. It processes dashcam footage and GPS logs to detect, classify, and prioritize road damage using RT-DETR-L (primary model), SAM segmentation, depth estimation, and XGBoost severity scoring. Results are stored in a PostGIS database and served through a FastAPI backend.

## Environment Setup

Python 3.12 with a virtual environment in `.venv/`:

```bash
source .venv/Scripts/activate   # Windows (bash)
pip install -r requirements.txt
# PyTorch is NOT in requirements.txt — install separately for your CUDA version:
# pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
# SAM also installed separately:
# pip install git+https://github.com/facebookresearch/segment-anything.git
```

Required `.env` file (see `docker-compose.yml` for DB vars):
```
DATABASE_URL=postgresql://user:password@localhost:5432/cluj_monitor
POSTGRES_USER=...
POSTGRES_PASSWORD=...
POSTGRES_DB=cluj_monitor
PIPELINE_RUN_HOUR=22   # optional, default 22
```

## Common Commands

```bash
# Start database (PostGIS in Docker)
docker-compose up -d

# Initialize DB schema (run once after docker-compose up)
python scripts/setup_db.py

# Run backend API
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
# Or: python backend/main.py
# API docs at http://localhost:8000/docs

# Run tests (tests/ is not yet populated)
pytest tests/
pytest tests/test_foo.py::test_bar   # run a single test

# Format code
black .
```

### ML Pipeline

```bash
# 0. Download raw datasets (RDD2022 + Pothole600)
python scripts/download_datasets.py

# 1. Prepare dataset — run in order:
python ml/detection/data_prep/prep_rdd2022.py      # convert RDD2022
python ml/detection/data_prep/prep_pothole600.py   # convert Pothole600
python ml/detection/data_prep/merge_datasets.py    # merge into one split
python ml/detection/data_prep/coco_to_yolo.py      # COCO → YOLO format
# Verify: python scripts/inspect_datasets.py / scripts/verify_merge.py

# 2. (Optional) Run PSO to find optimal hyperparameters — takes ~9-12h on RTX 2050
python ml/optimization/pso_hyperparams.py

# 3. Train RT-DETR-L (auto-loads pso_best.json if present)
python ml/detection/train.py
python ml/detection/train.py --smoke_test        # 2-epoch pipeline validation
python ml/detection/train.py --resume runs/detect/rtdetr_road/weights/last.pt

# 4. Evaluate
python ml/detection/evaluate.py
python ml/detection/evaluate.py --full           # val + test + TTA + comparison
python ml/detection/evaluate.py --compare        # compare best.pt vs swa.pt vs last.pt

# 5. Monitor training live (run in a separate terminal while train.py runs)
python ml/detection/monitor.py                   # auto-refreshes every 30s
python ml/detection/monitor.py --save            # one-shot, saves PNG
python ml/detection/monitor.py --interval 60     # custom refresh interval

# Manually trigger the full detection pipeline
python scripts/run_survey.py
python scripts/run_survey.py --date 2024-06-15

# Run the daily scheduler (blocks, fires pipeline at PIPELINE_RUN_HOUR each day)
python scheduler/daily_job.py
```

## Architecture

The system has four distinct layers:

### 1. ML Training (`ml/`)
- `ml/detection/` — RT-DETR-L training and evaluation against the merged RDD2022 + Pothole600 dataset
  - `data_prep/` — scripts to download, merge, and convert datasets to YOLO format
  - `train.py` — two-phase training: frozen backbone (phase 1) → full fine-tune (phase 2) + SWA
  - `evaluate.py` — val/test mAP evaluation with optional TTA
- `ml/optimization/pso_hyperparams.py` — PSO hyperparameter search; writes `pso_best.json` which `train.py` reads automatically
- `ml/weights/` — final production weights (`rtdetr_l_rdd2022.pt`)
- `ml/segmentation/`, `ml/depth/`, `ml/severity/` — SAM, monocular depth, XGBoost components (in progress)
- Training outputs go to `runs/detect/rtdetr_road/`; key checkpoints are `best.pt`, `swa.pt`, `last.pt`

### 2. Inference Pipeline (`pipeline/`, `scripts/`)
- `pipeline/orchestrator.py` — **not yet implemented**; planned to orchestrate: frame extraction → detection → segmentation → depth → severity → GPS clustering → DB write
- `scripts/run_survey.py` — manual one-shot trigger; calls `pipeline/orchestrator.py` when it exists
- `scheduler/daily_job.py` — APScheduler cron job that fires the orchestrator nightly (Europe/Bucharest timezone)

### 3. Backend API (`backend/`)
- FastAPI app at `backend/main.py`; reads from PostgreSQL/PostGIS via SQLAlchemy
- `backend/database.py` — engine, `get_db()` (FastAPI dependency), `get_db_session()` (pipeline use)
- `backend/models.py` — `Detection` and `SurveyLog` ORM models; `Detection.compute_priority_score()` implements the priority formula
- `backend/schemas.py` — Pydantic v2 schemas for all API responses
- Routes under `backend/routes/`: `detections`, `stats`, `heatmap`, `priority`
- Priority score formula: `severity × road_weight × infra_weight × log(detection_count + 1)`

### 4. Frontend (`frontend/`)
- Static JS (no build step) — currently a placeholder

## Data Layout

```
data/
  raw/
    footage/          # Input dashcam videos
    gps_logs/         # GPX GPS telemetry
  processed/
    frames/           # Extracted frames
    metadata/         # Per-frame annotation metadata
  datasets/           # Prepared training/evaluation datasets
  detection/
    dataset.yaml      # YOLO-format dataset config (generated by coco_to_yolo.py)
    train.json / val.json / test.json
```

## Database

PostGIS (PostgreSQL 15) runs in Docker. The `detections` table has spatial geometry (`POINT`, SRID 4326), all ML-derived attributes (severity, depth, SAM metrics), and temporal tracking (`first_detected`, `last_detected`, `detection_count`). `survey_log` tracks each pipeline run.

Use `get_db()` (FastAPI `Depends`) in routes and `get_db_session()` in pipeline scripts.

## Key Implementation Notes

- **GPU requirement**: training requires CUDA. The model trains on RTX 2050 (4 GB VRAM) using `amp=True` (fp16), `batch=4`, `grad_accumulate=4` (effective batch 16), `workers≤4` (Windows).
- **Weights priority**: evaluation auto-selects `swa.pt > best.pt > rtdetr_l_rdd2022.pt`.
- **PSO → train integration**: run PSO first, then `train.py` picks up `ml/optimization/pso_best.json` automatically — no code changes needed.
- **Dataset classes**: `longitudinal_crack`, `transverse_crack`, `alligator_crack`, `pothole`, `patch_deterioration` (last class has no training data in current dataset).
- **Code style**: Black formatter. Run `black .` before committing.
