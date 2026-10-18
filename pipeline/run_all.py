"""Run all GIS pipeline milestones (Steps 1 through 4 and Step 6 validation) in sequence."""
from __future__ import annotations

import argparse

from pipeline.step1_dem import run_step1
from pipeline.step2_hydrology import run_step2
from pipeline.step3_hand import run_step3
from pipeline.step4_classify import run_step4
from pipeline.step6_validate import evaluate_points_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete Lagos Flood Susceptibility GIS pipeline.")
    parser.add_argument("--force", action="store_true", help="Recompute all steps even if cached outputs exist.")
    args = parser.parse_args()

    run_step1(force=args.force)
    run_step2(force=args.force)
    run_step3(force=args.force)
    run_step4(force=args.force)
    evaluate_points_csv()


if __name__ == "__main__":
    main()
