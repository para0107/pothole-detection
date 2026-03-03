"""
scripts/run_survey.py

Manual trigger for the full pipeline.
Use this to test outside of the scheduler, or to re-process a specific day.

Usage:
    python scripts/run_survey.py
    python scripts/run_survey.py --date 2024-06-15
"""

import os
import sys
import argparse
import subprocess
from datetime import date

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORCHESTRATOR = os.path.join(PROJECT_ROOT, "pipeline", "orchestrator.py")


def main():
    parser = argparse.ArgumentParser(
        description="Manually trigger the Cluj Urban Monitor pipeline."
    )
    parser.add_argument(
        "--date",
        type=str,
        default=str(date.today()),
        help="Survey date in YYYY-MM-DD format (default: today)",
    )
    args = parser.parse_args()

    logger.info(f"Manually triggering pipeline for date: {args.date}")

    result = subprocess.run(
        [sys.executable, ORCHESTRATOR, "--date", args.date],
        cwd=PROJECT_ROOT,
    )

    if result.returncode == 0:
        logger.success("Pipeline completed successfully.")
    else:
        logger.error(f"Pipeline failed with exit code {result.returncode}.")
        sys.exit(1)


if __name__ == "__main__":
    main()