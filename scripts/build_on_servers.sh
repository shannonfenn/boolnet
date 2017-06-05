#!/bin/bash
GRN='\033[0;32m'
MAG='\033[0;35m'
NC='\033[0m' # No Color

CIBM_SERVER=fisher
UNI_SERVER=darwin15

printf "building ${GRN}locally${NC}\n"
python setup.py clean > local_build.log 2>&1
python setup.py build_ext --inplace >> local_build.log 2>&1
pip install -e . 
tail -n 1 local_build.log

printf "building on ${GRN}$CIBM_SERVER${NC}\n"
{
  ssh -T $CIBM_SERVER << EOF
  source scoop_setup.sh
  cd /home/cibm01/shannon/sf/HMRI/code/boolnet
  git pull
  python setup.py clean
  python setup.py build_ext --inplace
  pip install -e .
EOF
} > cibm_server_build.log 2>&1
tail -n 1 cibm_server_build.log

printf "building on ${GRN}$UNI_SERVER${NC}\n"
{
  ssh -T $UNI_SERVER << EOF
  source scoop_setup.sh
  cd /home/cibm01/shannon/dar/HMRI/code/boolnet
  git pull
  python setup.py clean
  scl enable devtoolset-3 "python setup.py build_ext --inplace"
  pip install -e .
EOF
} > uni_server_build.log 2>&1
tail -n 1 uni_server_build.log
