#! /usr/bin/env python

import rapidjson as json
import argparse


def load(fname):
    with open(fname) as f:
        records = json.loads(f.read())
    mapping = {record['id']: record for record in records}
    if len(mapping) != len(records):
        raise ValueError('Duplicates record ids exist in {}'.format(fname))
    return mapping


def main():
    parser = argparse.ArgumentParser(description='combine results')
    parser.add_argument('inputs', type=str, nargs='+',
                        help='[later will overwrite keys of earlier]')
    args = parser.parse_args()

    output = {}
    for f in args.inputs:
        output.update(load(f))

    output = '\n,'.join(json.dumps(output[k]) for k in sorted(output))
    print(f'[{output}\n]')


if __name__ == '__main__':
    main()
