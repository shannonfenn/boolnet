#!/bin/sh
{
	ssh -T shannon << EOF
  cd HMRI/code/boolnet
  git clean -xdf
  ~/hyrule/bin/python setup.py build_ext --inplace
EOF
} > shannon_build.log 2>&1

{
ssh -T darwin01 << EOF
  cd HMRI/code/boolnet
  git clean -xdf
  scl enable devtoolset-3 '~/hyrule/bin/python setup.py build_ext --inplace'
EOF
} > darwin01_build.log 2>&1

tail -n 1 shannon_build.log
tail -n 1 darwin01_build.log