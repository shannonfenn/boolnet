import argparse
import glob
import re
import os.path
import subprocess as sp


def walltime_arg_type(s):
    if re.fullmatch('[0-9]?[0-9]:[0-9][0-9]:[0-9][0-9]|[0-9]+', s):
        return s


def parse_args():
    # queues = ['computeq', 'xeon3q', 'xeon4q']
    queues = ['xeon3q']

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('file', type=argparse.FileType(),
                        help='File containing list of .exp[list] files')
    parser.add_argument('walltime', type=walltime_arg_type)
    parser.add_argument('--queue', '-q', type=str, metavar='queue',
                        default='xeon3q', choices=queues)
    parser.add_argument('--out', '-o', type=argparse.FileType('w'),
                        help='optional file to dump job ids.')

    args = parser.parse_args()

    return args


def submit(bundles, queue, walltime, joblistfile):
    ids = []
    script = os.path.expanduser('~/HMRI/code/boolnet/pbs/j_submit_single.sh')

    if not os.path.isfile(script):
        print('Error: script does not exist. Aborting.')
        print('Bad script path: ' + script)
        return
    # pbs job limit
    if len(bundles) > 7500:
        print('Error: cannot submit {} jobs. Aborting.'.format(
            len(bundles)))
        return

    try:
        resources = 'walltime={}'.format(walltime)
        for i, expfile in enumerate(bundles):
            sout = '{}.sout'.format(expfile)
            serr = '{}.serr'.format(expfile)
            cmd = [script, expfile, sout, serr, queue, resources]
            status = sp.run(cmd, stdout=sp.PIPE, universal_newlines=True)
            ids.append(status.stdout + '\n')
    finally:
        print('{} jobs submitted.'.format(len(ids)))
        if joblistfile:
            joblistfile.writelines(ids)


def main():
    args = parse_args()

    bundles = args.file.read().splitlines()
    args.file.close()

    submit(bundles, args.queue, args.walltime, args.out)


if __name__ == '__main__':
    main()
