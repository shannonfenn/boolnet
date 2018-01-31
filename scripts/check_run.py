#! /usr/bin/env python

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


def read_json(contents):
    ''' Attempts to read string as json list. 
        If exception thrown, reattempt after appending "]".
        If that fails, reattempt after removing last line.
        Else the original exception is raised.'''
    contents = contents.strip()
    if not contents:
        return []
    try:
        return json.loads(contents)
    except:
        try:
            return json.loads(contents + ']')
        except:
            return json.loads(contents.rsplit('\n', 1)[0] + ']')




def get_remaining_experiments(directory, fast=True):
    all_json = glob.glob(join(directory, '*.json'))

    finished_ids = []
    for jsonfile in all_json:
        with open(jsonfile) as f:
            records = read_json(f.read())
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
    print('\n'.join(remaining))


def get_non_memorised_experiments(directory, fast=True):
    all_json = glob.glob(join(directory, '*.json'))

    failed_ids = []
    for jsonfile in all_json:
        with open(jsonfile, 'r') as f:
            records = read_json(f.read())
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
    print('\n'.join(failed_paths))


def summary(directory):
    all_json = glob.glob(join(directory, '*.json'))

    num_exp = len(get_all_experiments(directory))
    num_failed = 0
    num_succeeded = 0
    for fname in all_json:
        with open(fname, 'r') as f:
            try:
                records = read_json(f.read())
            except:
                print('Warning: could not read {}'.format(fname))
            else:
                for record in records:
                    if record['trg_err'] == 0:
                        num_succeeded += 1
                    else:
                        num_failed += 1
    print('remaining: {} memorised: {} not-memorised: {}'.format(
        num_exp - num_succeeded - num_failed, num_succeeded, num_failed))


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
    parser_summary.set_defaults(func=summary)

    args = parser.parse_args()

    args.func(args.dir)

if __name__ == '__main__':
    main()
