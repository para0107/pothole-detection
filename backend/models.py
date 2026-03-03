"""
backend/models.py

SQLAlchemy ORM models — mirror the schema created by scripts/setup_db.py.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import (
    Column, String, Float, Integer, SmallInteger,
    Date, DateTime, Text, Boolean, JSON,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from geoalchemy2 import Geometry

from backend.database import Base


class Detection(Base):
    __tablename__ = "detections"

    # ── Identity ──────────────────────────────────────────────────────────────
    id                  = Column(UUID(as_uuid=True), primary_key=True,
                                 server_default=func.gen_random_uuid())
    created_at          = Column(DateTime(timezone=True), server_default=func.now())
    updated_at          = Column(DateTime(timezone=True), server_default=func.now(),
                                 onupdate=func.now())

    # ── Spatial ───────────────────────────────────────────────────────────────
    geom                = Column(Geometry("POINT", srid=4326), nullable=False)
    latitude            = Column(Float, nullable=False)
    longitude           = Column(Float, nullable=False)

    # ── RT-DETR outputs ───────────────────────────────────────────────────────
    damage_type         = Column(String(30), nullable=False)
    confidence          = Column(Float, nullable=False)
    frame_path          = Column(Text)
    crop_path           = Column(Text)

    # ── SAM segmentation ──────────────────────────────────────────────────────
    surface_area_cm2    = Column(Float)
    edge_sharpness      = Column(Float)
    interior_contrast   = Column(Float)
    mask_compactness    = Column(Float)

    # ── Depth estimation ──────────────────────────────────────────────────────
    depth_estimate_cm   = Column(Float)
    depth_confidence    = Column(Float)
    shadow_geometry_score = Column(Float)
    lighting_condition  = Column(String(15))

    # ── XGBoost severity ──────────────────────────────────────────────────────
    severity            = Column(SmallInteger)
    severity_confidence = Column(Float)

    # ── Location context ──────────────────────────────────────────────────────
    street_name         = Column(String(150))
    road_importance     = Column(SmallInteger)
    infra_proximity_m   = Column(Float)

    # ── Weather ───────────────────────────────────────────────────────────────
    weather             = Column(JSONB)

    # ── Spatial density ───────────────────────────────────────────────────────
    surrounding_density = Column(Integer, default=0)

    # ── Temporal ──────────────────────────────────────────────────────────────
    first_detected      = Column(Date, nullable=False)
    last_detected       = Column(Date, nullable=False)
    detection_count     = Column(Integer, default=1)
    deterioration_rate  = Column(Float, default=0.0)

    # ── Priority ──────────────────────────────────────────────────────────────
    priority_score      = Column(Float, default=0.0)

    # ── Survey metadata ───────────────────────────────────────────────────────
    survey_date         = Column(Date, nullable=False)
    survey_video_file   = Column(String(255))

    def __repr__(self):
        return (
            f"<Detection id={self.id} type={self.damage_type} "
            f"severity=S{self.severity} lat={self.latitude:.4f} lon={self.longitude:.4f}>"
        )

    def compute_priority_score(self):
        """
        priority = severity_score × road_weight × infra_weight × log(detection_count + 1)

        road_weight:  primary=3, secondary=2, residential=1
        infra_weight: within 50m of school/hospital/bus stop = 2, else = 1
        """
        import math

        severity_score  = self.severity or 1
        road_weight     = self.road_importance or 1
        infra_weight    = 2.0 if (self.infra_proximity_m or 9999) <= 50 else 1.0
        count_factor    = math.log((self.detection_count or 1) + 1)

        self.priority_score = round(
            severity_score * road_weight * infra_weight * count_factor, 4
        )
        return self.priority_score


class SurveyLog(Base):
    __tablename__ = "survey_log"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    survey_date         = Column(Date, nullable=False, unique=True)
    started_at          = Column(DateTime(timezone=True))
    finished_at         = Column(DateTime(timezone=True))
    status              = Column(String(20), default="pending")
    frames_processed    = Column(Integer, default=0)
    detections_found    = Column(Integer, default=0)
    new_detections      = Column(Integer, default=0)
    updated_detections  = Column(Integer, default=0)
    error_message       = Column(Text)
    video_files         = Column(JSONB)

    def __repr__(self):
        return (
            f"<SurveyLog date={self.survey_date} status={self.status} "
            f"detections={self.detections_found}>"
        )