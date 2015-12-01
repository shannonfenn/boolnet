#!/bin/sh
{
	ssh -T shannon << EOF
  source scoop_setup.sh
  cd /home/cibm01/shannon/HMRI/code_sf/boolnet
  git clean -xdf
  python setup.py build_ext --inplace
EOF
} > shannon_build.log 2>&1

{
ssh -T darwin15 << EOF
  source scoop_setup.sh
  cd /home/cibm01/shannon/HMRI/code_dar/boolnet
  git clean -xdf
  scl enable devtoolset-3 'python setup.py build_ext --inplace'
EOF
} > darwin15_build.log 2>&1

scp runexp.py shannon:HMRI/scripts/
scp runexp.py darwin15:HMRI/scripts/
cp -f runexp.py ~/HMRI/scripts/

tail -n 1 shannon_build.log
tail -n 1 darwin15_build.log
