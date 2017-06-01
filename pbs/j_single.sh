#!/bin/bash
#
#PBS -l select=1:ncpus=1:mem=500MB

python $HOME/HMRI/scripts/run_single_experiment.py ${EXP_FILE}
