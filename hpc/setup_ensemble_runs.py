"""Setup script to generate HPC simulation directories and ocean.in configurations for ensemble experiments.

Supports:
- A reference ensemble run (using a reference grid file, e.g. without wind farms).
- One or more wind farm parametrization variations (using a wind farm grid file containing
  farm locations/properties), with arbitrary user-specified parameter overrides in ocean.in.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def set_roms_parameter(content: str, param_name: str, param_value: Any) -> str:
    """Update or add a ROMS parameter in ocean.in content (e.g. PARAM == VALUE)."""
    pattern = rf"(?m)^(\s*{re.escape(param_name)}\s*==\s*)[^\n!#]+"
    val_str = str(param_value)

    if re.search(pattern, content):
        content = re.sub(pattern, rf"\g<1>{val_str}", content)
    else:
        # If parameter not present in file, append it cleanly at the end
        if not content.endswith("\n"):
            content += "\n"
        content += f"{param_name} == {val_str}\n"

    return content


def customize_ocean_in(
    template_path: Path,
    output_path: Path,
    ini_path: Path,
    rst_path: Path,
    his_path: Path,
    grid_path: Path,
    title: str,
    param_overrides: Optional[Dict[str, Any]] = None,
) -> None:
    """Update standard ROMS parameters (ININAME, RSTNAME, HISNAME, GRDNAME, TITLE) plus custom overrides."""
    content = template_path.read_text()

    # Core file names and title
    content = set_roms_parameter(content, "ININAME", str(ini_path.resolve()))
    content = set_roms_parameter(content, "RSTNAME", str(rst_path.resolve()))
    content = set_roms_parameter(content, "HISNAME", str(his_path.resolve()))
    content = set_roms_parameter(content, "GRDNAME", str(grid_path.resolve()))
    content = set_roms_parameter(content, "TITLE", title)

    # Custom parameter overrides for this specific variation
    if param_overrides:
        for param, value in param_overrides.items():
            content = set_roms_parameter(content, param, value)

    output_path.write_text(content)


def setup_ensemble_experiments(
    work_dir: Path,
    template_ocean_in: Path,
    roms_executable: Path,
    ini_dir: Path,
    ref_grid: Path,
    farm_grid: Path,
    variations: Optional[Dict[str, Dict[str, Any]]] = None,
    num_members: int = 10,
    include_reference: bool = True,
) -> List[str]:
    """Set up experiment directory structure for reference and one or more wind farm variations.

    Directory layout:
      <work_dir>/
        ├── reference/
        │   ├── mem01/ (ocean.in using ref_grid, symlink to romsM, etc.)
        │   └── ...
        ├── farm_var1/
        │   ├── mem01/ (ocean.in using farm_grid + var1 parameter overrides)
        │   └── ...
        └── farm_var2/
            ├── mem01/ (ocean.in using farm_grid + var2 parameter overrides)
            └── ...

    Parameters
    ----------
    work_dir : Path
        Base experiment directory.
    template_ocean_in : Path
        Template ocean.in.
    roms_executable : Path
        Path to compiled romsM executable.
    ini_dir : Path
        Directory containing perturbed initial files (norkyst800_ini_memXX.nc).
    ref_grid : Path
        Grid file for reference run (no wind farms).
    farm_grid : Path
        Grid file for wind farm runs (contains wind farm locations).
    variations : dict of {var_name: {param_name: param_value}}
        Variations of wind farm parametrizations with their parameter overrides.
    num_members : int
        Number of ensemble members (default 10).
    include_reference : bool
        Whether to generate the reference experiment (default True).

    Returns
    -------
    List of experiment names configured.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    experiments: Dict[str, Dict[str, Any]] = {}

    if include_reference:
        experiments["reference"] = {
            "grid": ref_grid,
            "params": {},
        }

    # Default to one farm variation if none specified
    if not variations:
        variations = {"farm_default": {}}

    for var_name, params in variations.items():
        experiments[var_name] = {
            "grid": farm_grid,
            "params": params,
        }

    exp_names = list(experiments.keys())

    for exp_name, config in experiments.items():
        exp_dir = work_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        grid_file = config["grid"]
        overrides = config["params"]

        for k in range(1, num_members + 1):
            mem_name = f"mem{k:02d}"
            mem_dir = exp_dir / mem_name
            mem_dir.mkdir(parents=True, exist_ok=True)

            # Symlink ROMS executable
            roms_link = mem_dir / "romsM"
            if roms_link.exists() or roms_link.is_symlink():
                roms_link.unlink()
            roms_link.symlink_to(roms_executable.resolve())

            # Initial condition file for this member
            ini_file = ini_dir / f"norkyst800_ini_mem{k:02d}.nc"
            rst_file = mem_dir / f"norkyst800_rst_{exp_name}_{mem_name}.nc"
            his_file = mem_dir / f"norkyst800_his_{exp_name}_{mem_name}.nc"
            title = f"Norkyst-800 {exp_name} Ensemble {mem_name}"

            # Create customized ocean.in
            target_ocean_in = mem_dir / "ocean.in"
            customize_ocean_in(
                template_path=template_ocean_in,
                output_path=target_ocean_in,
                ini_path=ini_file,
                rst_path=rst_file,
                his_path=his_file,
                grid_path=grid_file,
                title=title,
                param_overrides=overrides,
            )

    print(f"Setup complete: {len(exp_names)} experiment(s) with {num_members} members each created in {work_dir}")
    print(f"Configured experiments: {', '.join(exp_names)}")
    return exp_names


def main():
    parser = argparse.ArgumentParser(
        description="Setup Norkyst-ROMS ensemble runs with reference and wind farm parametrization variations."
    )
    parser.add_argument("--work-dir", type=Path, required=True, help="Base simulation directory.")
    parser.add_argument("--template-ocean-in", type=Path, required=True, help="Template ocean.in file.")
    parser.add_argument("--roms-executable", type=Path, required=True, help="Path to compiled romsM.")
    parser.add_argument("--ini-dir", type=Path, required=True, help="Directory with perturbed initial files.")
    parser.add_argument("--ref-grid", type=Path, required=True, help="Grid file for reference (no wind farms).")
    parser.add_argument("--farm-grid", type=Path, required=True, help="Grid file for wind farm experiments.")
    parser.add_argument("--members", type=int, default=10, help="Number of ensemble members (default: 10).")
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Skip reference experiment generation and generate only farm variations.",
    )
    parser.add_argument(
        "--variations-json",
        type=Path,
        default=None,
        help="JSON file defining parametrization variations and parameter overrides. "
             'Example format: {"farm_drag_low": {"Cd_turb": 0.05}, "farm_drag_high": {"Cd_turb": 0.15}}',
    )
    parser.add_argument(
        "--variation",
        action="append",
        nargs="+",
        metavar=("VAR_NAME", "PARAM=VALUE"),
        help="Define a variation directly via CLI. Example: --variation farm_v1 Cd_turb=0.08 Mix_scale=1.5",
    )

    args = parser.parse_args()

    # Parse variations
    variations: Dict[str, Dict[str, Any]] = {}
    if args.variations_json and args.variations_json.exists():
        with open(args.variations_json, "r") as f:
            variations.update(json.load(f))

    if args.variation:
        for item in args.variation:
            var_name = item[0]
            if var_name not in variations:
                variations[var_name] = {}
            for kv in item[1:]:
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    # Try casting to float/int if possible
                    try:
                        v = int(v) if v.isdigit() else float(v)
                    except ValueError:
                        pass
                    variations[var_name][k.strip()] = v

    setup_ensemble_experiments(
        work_dir=args.work_dir,
        template_ocean_in=args.template_ocean_in,
        roms_executable=args.roms_executable,
        ini_dir=args.ini_dir,
        ref_grid=args.ref_grid,
        farm_grid=args.farm_grid,
        variations=variations if variations else None,
        num_members=args.members,
        include_reference=not args.no_reference,
    )


if __name__ == "__main__":
    main()
