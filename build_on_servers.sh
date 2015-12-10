#!/bin/sh
GRN='\033[0;32m'
MAG='\033[0;35m'
NC='\033[0m' # No Color

CIBM_SERVER=fisher
UNI_SERVER=darwin15

echo -e "building ${GRN}locally${NC}"
rm -f boolnet/*/*.so
rm -f boolnet/*/*.c
rm -f boolnet/*/*.cpp
python setup.py build_ext --inplace > local_build.log 2>&1
tail -n 1 local_build.log

echo -e "building on ${GRN}$CIBM_SERVER${NC}"
{
	ssh -T $CIBM_SERVER << EOF
  source scoop_setup.sh
  cd /home/cibm01/shannon/HMRI/code_sf/boolnet
  git pull
  git clean -xdf
  python setup.py build_ext --inplace
EOF
} > cibm_server_build.log 2>&1
tail -n 1 cibm_server_build.log

echo -e "building on ${GRN}$UNI_SERVER${NC}"
{
ssh -T $UNI_SERVER << EOF
  source scoop_setup.sh
  cd /home/cibm01/shannon/HMRI/code_dar/boolnet
  git pull
  git clean -xdf
  scl enable devtoolset-3 "python setup.py build_ext --inplace"
EOF
} > uni_server_build.log 2>&1
tail -n 1 uni_server_build.log

echo "distributing ${MAG}runexp.py${NC}"
scp runexp.py fisher:HMRI/scripts/
cp -f runexp.py ~/HMRI/scripts/
