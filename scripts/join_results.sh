#!/bin/bash

for f in `find working/ -type f -name '*.json' | sort -V`;
do
    # replace first "[" with "," and remove final "]" 
    sed '1 s/\[/\,/' "${f}" | sed -e '$s/]$//'
done > results.json
# remove blank lines
sed -i '/^$/d' results.json
# replace first "," with "["
sed -i '1 s/\,/\[/' results.json
# add final "]"
printf "]\n" >> results.json
