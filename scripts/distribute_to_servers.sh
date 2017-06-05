#!/bin/bash
GRN='\033[0;32m'
MAG='\033[0;35m'
NC='\033[0m' # No Color

CIBM_SERVER=fisher
UNI_SERVER=darwin15

printf "distributing to ${GRN}$CIBM_SERVER${NC}\n"
ssh -T $CIBM_SERVER << EOF
source scoop_setup.sh
cd /home/cibm01/shannon/sf/HMRI/code/boolnet
git pull
EOF

printf "distributing to ${GRN}$UNI_SERVER${NC}\n"
ssh -T $UNI_SERVER << EOF
source scoop_setup.sh
cd /home/cibm01/shannon/dar/HMRI/code/boolnet
git pull
EOF
