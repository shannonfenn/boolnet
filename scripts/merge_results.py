#! /usr/bin/env python

import rapidjson as json
import argparse


def load(fname):
    with open(fname) as f:
        records = json.loads(f.read())
    mapping = {record['id']: record for record in records}
    if len(mapping) != len(records):
        raise ValueError(f'Duplicates record ids exist in {fname}')
    return mapping


def main():
    parser = argparse.ArgumentParser(description='combine results')
    parser.add_argument('inputs', type=str, nargs='+',
                        help='[later will overwrite keys of earlier]')
    args = parser.parse_args()

    in_maps = [load(f) for f in args.inputs]

    output = dict(item
                  for m in in_maps
                  for item in m.items())

    output = [output[k] for k in sorted(output)]
    output = '\n,'.join(json.dumps(record) for record in output)
    print(f'[{output}\n]')


if __name__ == '__main__':
    main()
