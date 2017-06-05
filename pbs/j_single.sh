#!/bin/bash
#
#PBS -l select=1:ncpus=2:mem=2GB

python $HOME/HMRI/scripts/run_single_experiment.py ${EXP_FILE}
