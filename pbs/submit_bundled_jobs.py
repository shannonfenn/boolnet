import argparse
import glob
import re
import subprocess as sp
from os.path import isfile, basename, expanduser, splitext


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
    parser.add_argument('--jobname', '-j', type=str)

    args = parser.parse_args()

    if not args.jobname:
        args.jobname = basename(splitext(args.file.name)[1])
        args.jobname = args.jobname[:min(10, len(args.jobname))]
    elif len(args.jobname) > 10:
        parser.error('jobname ({}) must be 10 char or less.'.format(
            args.jobname))

    return args


def get_remaining_experiments(directory):
    all_exp = set(splitext(f)[0] for f in glob.iglob(
        '{}/working/*.exp'.format(directory)))
    all_json = set(splitext(f)[0] for f in glob.iglob(
        '{}/working/*.json'.format(directory)))
    remaining = all_exp - all_json
    return [f + '.exp' for f in remaining]


def submit(bundles, base_jobname, queue, walltime):
    ids = []
    script = expanduser('~/HMRI/code/boolnet/pbs/j_submit_single.sh')

    if not isfile(script):
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
            sout = 'b{}.sout'.format(splitext(expfile)[0])
            serr = 'b{}.serr'.format(splitext(expfile)[0])
            jobname = '{}_j{}'.format(base_jobname, i)
            cmd = [script, expfile, jobname, sout, serr, queue, resources]
            status = sp.run(cmd, stdout=sp.PIPE, universal_newlines=True)
            ids.append(status.stdout)
    finally:
        print(''.join(str(s) for s in ids))


def main():
    args = parse_args()

    bundles = args.file.read().splitlines()
    args.file.close()

    submit(bundles, args.jobname, args.queue, args.walltime)


if __name__ == '__main__':
    main()
