"""Tests for the HPC experiment setup with reference and wind farm parameter variations."""

import tempfile
from pathlib import Path
import pytest

from hpc.setup_ensemble_runs import customize_ocean_in, setup_ensemble_experiments, set_roms_parameter


def test_set_roms_parameter():
    content = "TITLE == Base Title\nVAR1 == 10\n# comment\nVAR2 == 20\n"
    # Update existing
    c1 = set_roms_parameter(content, "VAR1", 99)
    assert "VAR1 == 99" in c1
    assert "VAR2 == 20" in c1

    # Add new parameter
    c2 = set_roms_parameter(c1, "NEW_PARAM", "0.05")
    assert "NEW_PARAM == 0.05" in c2


def test_setup_ensemble_experiments_with_variations():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        work_dir = tmp / "sim_work"
        template_in = tmp / "ocean.in"
        template_in.write_text(
            "TITLE == Test Run\n"
            "ININAME == /path/to/ini.nc\n"
            "RSTNAME == /path/to/rst.nc\n"
            "HISNAME == /path/to/his.nc\n"
            "GRDNAME == /path/to/grid.nc\n"
            "Cd_turb == 0.0\n"
        )

        roms_exe = tmp / "romsM"
        roms_exe.write_text("#!/bin/sh\nexit 0\n")
        roms_exe.chmod(0o755)

        ini_dir = tmp / "ini_files"
        ini_dir.mkdir(parents=True, exist_ok=True)
        for k in range(1, 4):
            (ini_dir / f"norkyst800_ini_mem{k:02d}.nc").touch()

        ref_grid = tmp / "grid_ref.nc"
        ref_grid.touch()
        farm_grid = tmp / "grid_with_farms.nc"
        farm_grid.touch()

        variations = {
            "farm_drag_low": {"Cd_turb": 0.05, "Mix_scale": 1.0},
            "farm_drag_high": {"Cd_turb": 0.15, "Mix_scale": 2.5},
        }

        exp_names = setup_ensemble_experiments(
            work_dir=work_dir,
            template_ocean_in=template_in,
            roms_executable=roms_exe,
            ini_dir=ini_dir,
            ref_grid=ref_grid,
            farm_grid=farm_grid,
            variations=variations,
            num_members=3,
            include_reference=True,
        )

        assert "reference" in exp_names
        assert "farm_drag_low" in exp_names
        assert "farm_drag_high" in exp_names

        # Check reference member 1
        ref_mem1_in = (work_dir / "reference" / "mem01" / "ocean.in").read_text()
        assert f"GRDNAME == {ref_grid.resolve()}" in ref_mem1_in
        assert "Cd_turb == 0.0" in ref_mem1_in

        # Check farm_drag_low member 2
        farm_low_in = (work_dir / "farm_drag_low" / "mem02" / "ocean.in").read_text()
        assert f"GRDNAME == {farm_grid.resolve()}" in farm_low_in
        assert "Cd_turb == 0.05" in farm_low_in
        assert "Mix_scale == 1.0" in farm_low_in

        # Check farm_drag_high member 3
        farm_high_in = (work_dir / "farm_drag_high" / "mem03" / "ocean.in").read_text()
        assert f"GRDNAME == {farm_grid.resolve()}" in farm_high_in
        assert "Cd_turb == 0.15" in farm_high_in
        assert "Mix_scale == 2.5" in farm_high_in
