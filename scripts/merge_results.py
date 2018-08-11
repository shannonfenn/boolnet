#! /usr/bin/env python

import argparse
import re


def main():
    parser = argparse.ArgumentParser(description='combine results')
    parser.add_argument('inputs', type=str, nargs='+',
                        help='[later will overwrite keys of earlier]')
    args = parser.parse_args()

    # build a mapping with 'id' keys 
    # this allows later records with the 
    # same id to override previous records
    id_map = {}
    for fname in args.inputs:
        with open(fname) as f:
            # print([len(line) for line in f])
            for line in f:
                # print(line)
                match = re.search(pattern='id":\s?(\d+)(,|\})',
                                  string=line)
                key = match.group(1)
                id_map[key] = line

    for line in id_map.values():
        print(line, end='') # lines already contain newlines


if __name__ == '__main__':
    main()
