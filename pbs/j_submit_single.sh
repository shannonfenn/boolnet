#!/bin/bash

######################################################################################################
# Your parameters
EXP_FILE=$1
SOUT=$2
SERR=$3
QUEUE=$4
RESOURCES=$5
SCRIPT=$6

######################################################################################################
#1. Running the job
jnum=$(qsub -q ${QUEUE} -l ${RESOURCES} -o ${SOUT} -e ${SERR} -v EXP_FILE=${EXP_FILE} ${SCRIPT})
echo ${jnum}
