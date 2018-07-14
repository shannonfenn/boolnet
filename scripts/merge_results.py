#! /usr/bin/env python

import rapidjson as json
import argparse


def main():
    parser = argparse.ArgumentParser(description='combine results')
    parser.add_argument('inputs', type=str, nargs='+',
                        help='[later will overwrite keys of earlier]')
    args = parser.parse_args()

    # build a mapping with 'id' keys 
    id_map = {}
    for fname in args.inputs:
        with open(fname) as f:
            for lineno, line in enumerate(f):
                if not line.startswith(']'):
                    record = json.loads(line[1:]) # trim leading delimiter
                    id_map[record['id']] = line[1:]

    for i, line in enumerate(id_map.values()):
        print(',' if i else '[', end='')
        print(line, end='')
    print(']')


if __name__ == '__main__':
    main()
