#!/bin/bash
#
#PBS -l select=1:ncpus=1:mem=32GB

run_experiments.py ${EXP_FILE}
