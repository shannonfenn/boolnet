#!/bin/sh
GRN='\033[0;32m'
MAG='\033[0;35m'
NC='\033[0m' # No Color

echo -e "building ${GRN}locally${NC}"
rm -f boolnet/*/*.so
rm -f boolnet/*/*.c
rm -f boolnet/*/*.cpp
python setup.py build_ext --inplace > local_build.log 2>&1
tail -n 1 local_build.log

echo -e "building on ${GRN}shannon${NC}"
{
	ssh -T shannon << EOF
  source scoop_setup.sh
  cd /home/cibm01/shannon/HMRI/code_sf/boolnet
  git clean -xdf
  python setup.py build_ext --inplace
EOF
} > shannon_build.log 2>&1
tail -n 1 shannon_build.log

echo -e "building on ${GRN}darwin15${NC}"
{
ssh -T darwin15 << EOF
  source scoop_setup.sh
  cd /home/cibm01/shannon/HMRI/code_dar/boolnet
  git clean -xdf
  scl enable devtoolset-3 "python setup.py build_ext --inplace"
EOF
} > darwin15_build.log 2>&1
tail -n 1 darwin15_build.log

echo "distributing ${MAG}runexp.py${NC}"
scp runexp.py shannon:HMRI/scripts/
scp runexp.py darwin15:HMRI/scripts/
cp -f runexp.py ~/HMRI/scripts/
