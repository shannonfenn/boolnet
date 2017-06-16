#!/bin/bash

printf "[" > results.json
for f in `ls -v working/*.json`;
do
    cat "${f}" >> results.json
    printf "\n," >> results.json
done
sed -i '$s/.*/]/' results.json
