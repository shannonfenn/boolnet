#!/bin/bash

printf "[" > results.json
for f in working/*.json;
do
    cat "${f}" >> results.json
    printf "\n," >> results.json
done
sed -i '$s/.*/]/' results.json