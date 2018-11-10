#! /usr/bin/env python

import sys
import glob
import argparse
import pickle
import gzip
import rapidjson as json
from os.path import abspath, expanduser, isdir, join, basename, splitext
from natsort import natsorted


def directory_type(directory):
    # Handle tilde
    directory = abspath(expanduser(directory))
    if isdir(directory):
        return directory
    else:
        raise Exception('{0} is not a valid path'.format(directory))


def get_all_experiments(directory):
    bundles = glob.glob(join(directory, '*.explist'))
    all_exps = []
    for explist in bundles:
        with open(explist) as f:
            bunch = [line
                     for line in f.read().splitlines()
                     if line]
            all_exps.extend(bunch)
    return all_exps


def get_remaining_experiments(args, fast=True):
    directory = args.dir
    all_json = glob.glob(join(directory, '*.json'))

    finished_ids = []
    for jsonfile in all_json:
        with open(jsonfile) as f:
            records = [json.loads(line) for line in f if line.strip()]
        finished_ids.extend(record['id'] for record in records)

    explist = get_all_experiments(directory)
    expmap = dict()

    for exp in explist:
        if fast:
            eid = int(splitext(basename(exp))[0])
        else:
            with gzip.open(exp, 'rb') as f:
                eid = pickle.load(f)['id']
        expmap[eid] = exp
    if len(expmap) != len(explist):
        raise ValueError('.exp files with duplicate ids!')

    remaining = natsorted(exp_filename
                          for i, exp_filename in expmap.items()
                          if i not in finished_ids)
    return '\n'.join(remaining)


def get_non_memorised_experiments(args, fast=True):
    directory = args.dir
    all_json = glob.glob(join(directory, '*.json'))

    failed_ids = []
    for jsonfile in all_json:
        with open(jsonfile, 'r') as f:
            records = [json.loads(line) for line in f if line.strip()]
        failed_ids.extend(record['id']
                          for record in records
                          if record['trg_err'] != 0)

    explist = get_all_experiments(directory)
    expmap = dict()

    for exp in explist:
        if fast:
            eid = int(splitext(basename(exp))[0])
        else:
            with gzip.open(exp, 'rb') as f:
                eid = pickle.load(f)['id']
        expmap[eid] = exp
    if len(expmap) != len(explist):
        raise ValueError('.exp files with duplicate ids!')

    failed_ids = natsorted(failed_ids)
    failed_paths = (expmap[i] for i in failed_ids)
    return '\n'.join(failed_paths)


def summary(args):
    directory = args.dir
    swallow_errors = args.s
    all_json = glob.glob(join(directory, '*.json'))

    num_exp = len(get_all_experiments(directory))
    num_failed = 0
    num_succeeded = 0
    for fname in all_json:
            try:
                with open(fname, 'r') as f:
                    records = [json.loads(line) for line in f if line.strip()]
            except (ValueError, TypeError) as e:
                if not swallow_errors:
                    print(f'Warning: could not read {fname}\n{e}',
                          file=sys.stderr)
            else:
                for record in records:
                    if record['trg_err'] == 0:
                        num_succeeded += 1
                    else:
                        num_failed += 1
    return 'remaining: {} memorised: {} not-memorised: {}'.format(
        num_exp - num_succeeded - num_failed, num_succeeded, num_failed)


def main():
    parser = argparse.ArgumentParser(
        description='Tools for filtering experiment (.exp/.json) files')

    subparsers = parser.add_subparsers(help='commands', dest='command')

    parser_remaining = subparsers.add_parser('rem')
    parser_remaining.add_argument('dir', type=directory_type)
    parser_remaining.set_defaults(func=get_remaining_experiments)

    parser_failed = subparsers.add_parser('not')
    parser_failed.add_argument('dir', type=directory_type)
    parser_failed.set_defaults(func=get_non_memorised_experiments)

    parser_summary = subparsers.add_parser('sum')
    parser_summary.add_argument('dir', type=directory_type)
    parser_summary.add_argument('-s', action='store_true',
                                help='silence warnings.')
    parser_summary.set_defaults(func=summary)

    args = parser.parse_args()

    print(args.func(args))


if __name__ == '__main__':
    main()
