#! /usr/bin/env python

import rapidjson as json
import argparse
import lzma
import os.path


def directory_type(directory):
    # Handle tilde
    directory = os.path.abspath(os.path.expanduser(directory))
    if os.path.isdir(directory):
        return directory
    else:
        raise Exception('{0} is not a valid path'.format(directory))


def memorised(record):
    if record['learner'] == 'ecc_member':
        # this learner uses bagging so gets non zero trg error even when mem'd
        return sum(record['best_err']) == 0
    else:
        return record['trg_err'] == 0


def summary(infile):
    num_failed = num_succeeded = 0
    for line in infile:
        if memorised(json.loads(line)):
            num_succeeded += 1
        else:
            num_failed += 1
    print(f'Total: {num_succeeded + num_failed} failed: {num_failed}')


def main():
    parser = argparse.ArgumentParser(
        description='Check experiment (.json) files')

    parser.add_argument('inputfile', type=str)

    args = parser.parse_args()

    if args.inputfile.endswith('.xz'):
        with lzma.open(args.inputfile, 'rt') as f:
            summary(f)
    else:
        with open(args.inputfile) as f:
            summary(f)


if __name__ == '__main__':
    main()
