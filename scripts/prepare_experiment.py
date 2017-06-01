from datetime import datetime       # for date for result dir
import os                           # for mkdir
import os.path                      # for path manipulation
import yaml                         # for loading experiment files
try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader
import shutil                       # file copying
import argparse                     # CLI
import itertools                    # imap and count
import pickle

from boolnet.utils import BetterETABar
import boolnet.exptools.config_tools as cfg


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('experiment',
                        type=argparse.FileType('r'),
                        help='experiment config filename.')
    parser.add_argument('-d', '--directory', type=str, metavar='dir',
                        default='~/HMRI/experiments/results',
                        help='working directory to dump configurations in.')
    parser.add_argument('-b', '--batch-mode', action='store_true',
                        help='suppress progress bars.')
    return parser.parse_args()


def create_result_dir(base_dir, exp_name):
    ''' Generates a new timestamped directory for the results and copies
        the experiment script, experiment file and git hash into it.'''
    # make directory for the results
    base_dir = os.path.expanduser(base_dir)

    working_dir = os.path.join(base_dir, '{}_{}_'.format(
        exp_name, datetime.now().strftime('%y-%m-%d')))

    # find the first number i such that 'Results_datetime_i' is not
    # already a directory and make that new directory
    working_dir += next(str(i) for i in itertools.count()
                        if not os.path.isdir(working_dir + str(i)))
    os.makedirs(working_dir)

    # return the directory name for the results to go in
    return working_dir


def initialise(args):
    ''' This is responsible for setting up the output directory.'''
    # load experiment file
    settings = yaml.load(args.experiment, Loader=Loader)

    # create result directory
    working_dir = create_result_dir(args.directory, settings['name'])

    # copy experiment config into results directory
    shutil.copy(args.experiment.name, working_dir)

    return settings, working_dir


def dump_tasks(tasks, working_dir, batch_mode):
    if not batch_mode:
        bar = BetterETABar('Dumping tasks', max=len(tasks))
        bar.update()
    try:
        for i, task in enumerate(tasks):
            fname = '{}/working/{}.exp'.format(working_dir, i)
            with open(fname, 'wb') as f:
                pickle.dump(task, f)
            if not batch_mode:
                bar.next()
    finally:
        # clean up progress bar before printing anything else
        if not batch_mode:
            bar.finish()


# ############################## MAIN ####################################### #
def main():
    args = parse_arguments()

    settings, working_dir = initialise(args)

    print('Directories initialised.')
    print('Prepping experiment files in: ' + working_dir + '\n')

    # generate learning tasks
    try:
        configurations = cfg.generate_configurations(settings,
                                                     args.batch_mode)
        print('{} configurations generated.'.format(len(configurations)))
        tasks = cfg.generate_tasks(configurations, args.batch_mode)
        print('{} tasks generated.\n'.format(len(tasks)))

        os.makedirs(os.path.join(working_dir, 'working'))

        dump_tasks(tasks, working_dir, args.batch_mode)
    except cfg.ValidationError as err:
        print(err)
        print('\nFailed to prep experiment.')
        print('Failed! Directory will still exist: {}'.format(working_dir))
        return

    print('Success! Experiment prepped in: {}.'.format(working_dir))

if __name__ == '__main__':
    main()
