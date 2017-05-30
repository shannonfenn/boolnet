#!/bin/bash

######################################################################################################
# Your parameters
SKF_EXP_DIR=$1
SKF_EXP_INDEX=$2
SKF_EXP_NAME_BASE=$3
# QUEUE=$4

######################################################################################################
#1. Prepping args
JOB_NAME=${SKF_EXP_NAME_BASE}_${SKF_EXP_INDEX}
SOUT=${SKF_EXP_DIR}/working/${SKF_EXP_INDEX}.stdout
SERR=${SKF_EXP_DIR}/working/${SKF_EXP_INDEX}.stderr

#2. Running the job
# jnum=$(qsub -q ${QUEUE} -N ${JOB_NAME} -o ${SOUT} -e ${SERR} -v SKF_EXP_DIR=${SKF_EXP_DIR},SKF_EXP_INDEX=${SKF_EXP_INDEX} j_single.sh)
jnum=$(qsub -N ${JOB_NAME} -o ${SOUT} -e ${SERR} -v SKF_EXP_DIR=${SKF_EXP_DIR},SKF_EXP_INDEX=${SKF_EXP_INDEX} j_single.sh)
echo ${jnum}
