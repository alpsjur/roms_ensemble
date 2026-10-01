"""Command-line interface for generating Norkyst-ROMS ensemble initial condition files."""

import argparse
import logging
import sys
from pathlib import Path
from .generator import generate_ensemble


def main():
    parser = argparse.ArgumentParser(
        description="Generate perturbed initial condition NetCDF files for Norkyst-ROMS ensembles."
    )
    parser.add_argument(
        "--base-ini",
        required=True,
        type=Path,
        help="Path to unperturbed base initial/restart NetCDF file.",
    )
    parser.add_argument(
        "--historical-snapshots",
        nargs="+",
        required=True,
        type=Path,
        help="Paths to historical restart NetCDF files used to compute anomaly library.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where perturbed initial condition files will be written.",
    )
    parser.add_argument(
        "--members",
        type=int,
        default=10,
        help="Number of ensemble members to generate (default: 10).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="Anomaly scaling factor (default: 0.10, i.e., 10%% of historical eddy variance).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible weights (default: 42).",
    )
    parser.add_argument(
        "--grid-file",
        type=Path,
        default=None,
        help="Optional external ROMS grid file (if bathymetry/masks are not in base-ini).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging.",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        generate_ensemble(
            base_file=args.base_ini,
            historical_snapshots=args.historical_snapshots,
            output_dir=args.output_dir,
            num_members=args.members,
            alpha=args.alpha,
            seed=args.seed,
            grid_file=args.grid_file,
        )
        print(f"Successfully generated {args.members} ensemble members in {args.output_dir}")
        sys.exit(0)
    except Exception as e:
        logging.exception(f"Ensemble generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
