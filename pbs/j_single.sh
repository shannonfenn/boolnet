#!/bin/bash
#
#PBS -l select=1:ncpus=2:mem=2GB

python $HOME/HMRI/scripts/run_experiments.py ${EXP_FILE}
