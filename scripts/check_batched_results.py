import glob
import argparse
import rapidjson as json
from os.path import abspath, expanduser, isdir, splitext, join, basename
from natsort import natsorted


def directory_type(directory):
    # Handle tilde
    directory = abspath(expanduser(directory))
    if isdir(directory):
        return directory
    else:
        raise Exception('{0} is not a valid path'.format(directory))


def get_all_experiments(directory):
    try:
        with open(join(directory, 'working', 'all.exp')) as f:
            return f.read.splitlines()
    except:
        return glob.glob('{}/working/*.exp'.format(directory))


def read_json(contents):
    ''' Attempts to read string as json list. If exception thrown, reattempts
        after appending "]". If that fails the exception is not caught.'''
    if not contents.strip():
        return []

    try:
        return json.loads(contents)
    except:
        return json.loads(contents + ']')


def get_remaining_experiments(directory):
    all_exp = get_all_experiments(directory)
    # strip directory and extension to be left with list of ids
    all_ids = [splitext(basename(s))[0] for s in all_exp]

    all_json = glob.glob('{}/working/*.json'.format(directory))

    finished_ids = []
    for jsonfile in all_json:
        with open(jsonfile) as f:
            records = read_json(f.read())
            finished_ids.extend(str(record['id']) for record in records)

    remaining = natsorted(i for i in all_ids if i not in finished_ids)
    remaining = ('{}/working/{}.json'.format(directory, i) for i in remaining)

    print('\n'.join(remaining))


def get_non_memorised_experiments(directory):
    all_json = glob.glob('{}/working/*.json'.format(directory))
    failed_ids = []
    for fname in all_json:
        with open(fname, 'r') as f:
            records = read_json(f.read())
            failed_ids.extend(str(record['id'])
                              for record in records
                              if record['trg_err'] != 0)
    failed_ids = natsorted(failed_ids)
    failed_paths = ('{}/working/{}.json'.format(directory, i)
                    for i in failed_ids)
    print('\n'.join(failed_paths))


def summary(directory):
    all_json = glob.glob('{}/working/*.json'.format(directory))

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
