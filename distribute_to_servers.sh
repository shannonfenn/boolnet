#!/bin/bash
GRN='\033[0;32m'
MAG='\033[0;35m'
NC='\033[0m' # No Color

CIBM_SERVER=fisher
UNI_SERVER=darwin15

printf "distributing to ${GRN}$CIBM_SERVER${NC}\n"
{
  ssh -T $CIBM_SERVER << EOF
  source scoop_setup.sh
  cd /home/cibm01/shannon/sf/HMRI/code/boolnet
  git pull
EOF
} > cibm_server_build.log 2>&1
tail -n 1 cibm_server_build.log

printf "distributing to ${GRN}$UNI_SERVER${NC}\n"
{
  ssh -T $UNI_SERVER << EOF
  source scoop_setup.sh
  cd /home/cibm01/shannon/dar/HMRI/code/boolnet
  git pull
EOF
} > uni_server_build.log 2>&1
tail -n 1 uni_server_build.log

printf "distributing ${MAG}runexp.py${NC}\n"
scp runexp.py fisher:HMRI/scripts/
cp -f runexp.py $HOME/HMRI/scripts/
