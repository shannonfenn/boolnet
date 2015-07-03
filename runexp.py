from datetime import datetime       # for date for result dir
from multiprocessing import Pool    # non-distributed parallellism      ## REPLACE WITH SCOOP LATER
from time import time               # timing
from progress.bar import Bar        # progress indicators
import os                           # for mkdir
import os.path                      # for path manipulation
import yaml                         # for loading experiment files
import sys                          # for path, exit
import shutil                       # file copying
import subprocess                   # for git and cython compile
import logging                      # for logging, duh
import argparse                     # CLI
import itertools                    # imap and count
import scoop                        # for distributed parallellism

from boolnet.learning.learn_boolnet import learn_bool_net
import experiment.config_tools as config_tools


def check_git():
    ''' Checks the git directory for uncommitted source modifications.
        Since the experiment is tagged with a git hash it should not be
        run while the repo is dirty.'''
    cur_file_dir = os.path.dirname(os.path.realpath(__file__))
    changed_files = subprocess.check_output(
        ['git', '-C', cur_file_dir, 'diff', '--name-only'], universal_newlines=True).splitlines()
    if changed_files:
        print(('Warning, the following files in git repo '
               'have changes:\n\t{}').format('\n\t'.join(changed_files)))


def create_result_dir(args, exp_name):
    ''' Generates a new timestamped directory for the results and copies
        the experiment script, experiment file and git hash into it.'''
    # make directory for the results
    result_dir = os.path.join(args.results_dir, '{}_{}_'.format(
        exp_name, datetime.now().strftime('%y-%m-%d')))

    # find the first number i such that 'Results_datetime_i' is not
    # already a directory and make that new directory
    result_dir += next(str(i) for i in itertools.count()
                       if not os.path.isdir(result_dir + str(i)))
    os.mkdir(result_dir)

    # copy experiment config and git hash into results directory
    shutil.copy(args.experiment.name, result_dir)
    with open(os.path.join(result_dir, 'git_hash'), 'w') as hashfile:
        hashfile.write(subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], universal_newlines=True))

    # create a temp subdir
    os.mkdir(os.path.join(result_dir, 'temp'))

    # return the directory name for the results to go in
    return result_dir


def initialise_logging(settings, result_dir):
    ''' sets up logging with user defined level. '''
    level_dict = {'warning': logging.WARNING,
                  'info': logging.INFO,
                  'debug': logging.DEBUG,
                  'none': logging.ERROR}
    try:
        log_level = level_dict[settings['logging_level'].lower()]
    except:
        log_level = logging.WARNING
    logging.basicConfig(filename=os.path.join(result_dir, 'log'), level=log_level)


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('experiment',
                        type=argparse.FileType('r'),
                        default='experiment.yaml',
                        help='experiment config filename (relative to --exp-dir).')
    parser.add_argument('--evaluator', default='cy', choices=['py', 'cy', 'gpu'],
                        help='python, cython or cuda based evaluator.')
    parser.add_argument('--numprocs', '-n', metavar='N', type=int,
                        default=8, choices=range(0, 17),
                        help='how many parallel processes to use (give 0 for scoop).')
    parser.add_argument('--keep-temp', action='store_true',
                        help='provide to retain temporary FABCPP files.')
    parser.add_argument('--data-dir', default='datasets', type=str,
                        help='dataset directory.')
    parser.add_argument('--results-dir', default='results', type=str,
                        help='directory to store results in (in own subdir).')

    args = parser.parse_args()

    if args.evaluator == 'gpu':
        raise NotImplementedError()
        # from BoolNet.NetworkEvaluatorGPU import NetworkEvaluatorGPU
        # args.evaluator = NetworkEvaluatorGPU
    elif args.evaluator == 'cy':
        import pyximport
        pyximport.install()
        from BoolNet.networkstate import NetworkState
        args.evaluator = NetworkState
    else:
        raise NotImplementedError()
        # from BoolNet.NetworkEvaluator import NetworkEvaluator
        # args.evaluator = NetworkEvaluator

    return args


def initialise(args):
    ''' This is responsible for setting up the output directory,
        checking the git repository for cleanliness, setting up
        logging and compiling the cython code for boolean nets.'''
    # Check version
    if sys.version_info.major != 3:
        sys.exit("Requires python 3.")

    check_git()

    # load experiment file
    settings = yaml.load(args.experiment, Loader=yaml.CSafeLoader)

    # MUST FIX THIS SINCE BASE_DIR will be code, not above
    settings['data']['dir'] = os.path.abspath(args.data_dir)

    # create result directory
    result_dir = create_result_dir(args, settings['name'])

    # initialise logging
    initialise_logging(settings, result_dir)

    return settings, result_dir


def run_tasks(tasks, num_processes, out_stream):
    out_stream.write('[')
    if num_processes == 1:
        run_sequential(tasks, out_stream)
    elif num_processes < 1:
        run_scooped(tasks, out_stream)
    else:
        # Run the actual learning as a parallel process
        run_parallel(tasks, num_processes, out_stream)
    out_stream.write(']')


def run_parallel(tasks, num_processes, out_stream):
    ''' runs the given configurations '''
    with Pool(processes=num_processes) as pool:
        bar = Bar('Parallelised', max=len(tasks))
        bar.update()
        # uses unordered imap to ensure results are dumped as soon as available
        for i, result in enumerate(pool.imap_unordered(learn_bool_net, tasks)):
            config_tools.dump_results_partial(result, out_stream, i == 0)
            bar.next()
        bar.finish()


def run_scooped(tasks, out_stream):
    ''' runs the given configurations '''
    bar = Bar('Scooped', max=len(tasks))
    bar.update()
    # uses unordered imap to ensure results are dumped as soon as available
    for i, result in enumerate(scoop.futures.map_as_completed(learn_bool_net, tasks)):
        config_tools.dump_results_partial(result, out_stream, i == 0)
        bar.next()
    bar.finish()


def run_sequential(tasks, out_stream):
    ''' runs the given configurations '''
    bar = Bar('Running sequentially', max=len(tasks))
    bar.update()
    for i, result in enumerate(itertools.imap(learn_bool_net, tasks)):
        config_tools.dump_results_partial(result, out_stream, i == 0)
        bar.next()
    bar.finish()


def notify(pb_handle, exp_name, result_dirname, time):
    result_dirname = str(result_dirname)
    print('Experiment completed in {} seconds. Results in \"{}\"'.format(time, result_dirname))
    if pb_handle:
        pb_handle.push_note(
            'Experiment complete.', 'name: {} time: {} results: {}'.format(
                exp_name, time, result_dirname))


# ############################## MAIN ####################################### #
def main():
    start_time = time()
    try:
        from pushbullet import PushBullet, PushbulletError
        pb = PushBullet('on6qP2blHZbxs5h0xhDRcnfxHLoIc9Jo')
    except ImportError:
        print('Failed to import PushBullet - notifications will not be sent.')
        pb = None
    except PushbulletError:
        print('Failed to generate PushBullet interface - notifications will not be sent.')
        pb = None

    args = parse_arguments()

    # run_experiment(repo_dir, build_subdir, experiment_filename)

    settings, result_dir = initialise(args)

    settings['inter_file_base'] = os.path.join(result_dir, 'temp', 'inter_')

    with open(os.path.join(result_dir, 'results.json'), 'w') as results_stream:
        # generate learning tasks
        configurations = config_tools.generate_configurations(settings, args.evaluator)
        print('{} runs generated.'.format(len(configurations)))
        # Run the actual learning as a parallel process
        run_tasks(configurations, args.numprocs, results_stream)

    if not args.keep_temp:
        print('Deleting temp directory')
        shutil.rmtree(os.path.join(result_dir, 'temp'))

    total_time = time() - start_time

    try:
        notify(pb, settings['name'], result_dir, total_time)
    except PushbulletError as err:
        print('Failed to send PushBullet notification: {}.'.format(err))
