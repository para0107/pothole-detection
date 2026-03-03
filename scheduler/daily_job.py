"""
scheduler/daily_job.py

Triggers the full pipeline once per day at the configured time.
Run this as a long-running background process:
    python scheduler/daily_job.py

It will block and fire the pipeline automatically every 24 hours.
You can also trigger a manual run at any time by running:
    python scripts/run_survey.py
"""

import os
import sys
import subprocess
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

RUN_HOUR   = int(os.getenv("PIPELINE_RUN_HOUR",   22))
RUN_MINUTE = int(os.getenv("PIPELINE_RUN_MINUTE",  0))

PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORCHESTRATOR    = os.path.join(PROJECT_ROOT, "pipeline", "orchestrator.py")
LOG_DIR         = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def run_pipeline():
    """Called by APScheduler at the configured time each day."""
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[{run_time}] Daily pipeline starting...")

    result = subprocess.run(
        [sys.executable, ORCHESTRATOR],
        cwd=PROJECT_ROOT,
        capture_output=False,   # let pipeline log to its own loguru handlers
    )

    if result.returncode == 0:
        logger.success(f"Daily pipeline completed successfully at {run_time}.")
    else:
        logger.error(
            f"Daily pipeline FAILED at {run_time} "
            f"(exit code {result.returncode}). "
            f"Check logs/pipeline.log for details."
        )


def main():
    scheduler = BlockingScheduler(timezone="Europe/Bucharest")

    scheduler.add_job(
        run_pipeline,
        trigger=CronTrigger(hour=RUN_HOUR, minute=RUN_MINUTE),
        id="daily_pipeline",
        name="Cluj Urban Monitor — Daily Pipeline",
        replace_existing=True,
    )

    logger.info(
        f"Scheduler started. Pipeline will run every day at "
        f"{RUN_HOUR:02d}:{RUN_MINUTE:02d} (Europe/Bucharest)."
    )
    logger.info("Press Ctrl+C to stop.")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user.")


if __name__ == "__main__":
    main()