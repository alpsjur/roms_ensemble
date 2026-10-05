#!/bin/bash -f

#$ -N create_ensemble
#$ -cwd
#$ -b y
#$ -l h_rt=03:00:00
#$ -l h_rss=16G,mem_free=16G,h_data=16G
#$ -S /bin/bash
#$ -q bigmem-r8.q
#$ -M ansju8054@met.no
#$ -m ae
#$ -j y
#$ -o /lustre/storeB/users/ansju8054/roms-ting/roms_ensemble/logs/

source /modules/rhel9/x86_64/mamba-mf3/etc/profile.d/ppimam.sh
mamba activate 2025-01-production

export PYTHONPATH=/lustre/storeB/users/ansju8054/roms-ting/roms_ensemble:$PYTHONPATH

python3 -m roms_ensemble.cli \
--base-ini /lustre/storeB/users/ansju8054/roms-ting/roms_ensemble/files/base/norkyst_ini.nc-20220202-REF \
--historical-snapshots /lustre/storeB/users/ansju8054/roms-ting/roms_ensemble/files/archive/*.nc \
--output-dir /lustre/storeB/users/ansju8054/roms-ting/roms_ensemble/files/ensemble \
--members 4 \
--alpha 0.10 \
--seed 42 \
--verbose