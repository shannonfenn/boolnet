#!/bin/bash
#
#PBS -l select=1:ncpus=1:mem=500MB

python $HOME/HMRI/scripts/run_single_experiment.py ${SKF_EXP_DIR} ${SKF_EXP_INDEX}
