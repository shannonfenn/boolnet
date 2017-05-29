#!/bin/bash
#
#PBS -l select=1:ncpus=1:mem=500MB
#PBS -l walltime=4:00:00

python $HOME/HMRI/scripts/run_single_experiment.py ${SKF_EXP_DIR} ${SKF_EXP_INDEX}
