#!/bin/bash

printf "[" > results.json
for f in `find working/ -type f -name '*.json' | sort -V`;
do
    cat "${f}"
    printf "\n,"
done >> results.json
sed -i '$s/.*/]/' results.json
