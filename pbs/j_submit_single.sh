#!/bin/bash

######################################################################################################
# Your parameters
EXP_FILE=$1
JOB_NAME=$2
SOUT=$3
SERR=$4
QUEUE=$5
RESOURCES=$6

######################################################################################################
#1. Running the job
jnum=$(qsub -q ${QUEUE} -l ${RESOURCES} -N ${JOB_NAME} -o ${SOUT} -e ${SERR} -v EXP_FILE=${EXP_FILE} ${HOME}/HMRI/code/boolnet/pbs/j_single.sh)
echo ${jnum}
