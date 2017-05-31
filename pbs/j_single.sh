#!/bin/bash
#
#PBS -l select=1:ncpus=1

python $HOME/HMRI/scripts/run_single_experiment.py ${SKF_EXP_DIR} ${SKF_EXP_INDEX}
