from datetime import datetime       # for date for result dir
from multiprocessing import Pool    # non-distributed parallellism
from time import time               # timing
from progress.bar import Bar        # progress indicators
import os                           # for mkdir
import os.path                      # for path manipulation
import yaml                         # for loading experiment files
import sys                          # for path, exit
import shutil                       # file copying
import logging                      # for logging, duh
import argparse                     # CLI
import itertools                    # imap and count
import scoop                        # for distributed parallellism

from boolnet.learning.learn_boolnet import learn_bool_net
import boolnet.exptools.config_tools as config_tools


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
    logging.basicConfig(filename=os.path.join(result_dir, 'log'),
                        level=log_level)


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('experiment',
                        type=argparse.FileType('r'),
                        default='experiment.yaml',
                        help='experiment config filename.')
    parser.add_argument('--numprocs', '-n', metavar='N', type=int,
                        default=8, choices=range(0, 17),
                        help='how many parallel processes to use (give 0 for '
                             'scoop).')
    parser.add_argument('--no-notify', action='store_true',
                        help='disable PushBullet notifications.')
    parser.add_argument('-d', '--data-dir', type=str, metavar='dir',
                        default='experiments/datasets/functions',
                        help='base directory for datasets.')
    parser.add_argument('-s', '--sample-dir', type=str, metavar='dir',
                        default='experiments/datasets/samples',
                        help='base directory for sampling files.')
    parser.add_argument('-r', '--result-dir', type=str, metavar='dir',
                        default='experiments/results',
                        help='directory to store results in (in own subdir).')

    return parser.parse_args()


def create_result_dir(base_dir, exp_name):
    ''' Generates a new timestamped directory for the results and copies
        the experiment script, experiment file and git hash into it.'''
    # make directory for the results
    result_dir = os.path.join(base_dir, '{}_{}_'.format(
        exp_name, datetime.now().strftime('%y-%m-%d')))

    # find the first number i such that 'Results_datetime_i' is not
    # already a directory and make that new directory
    result_dir += next(str(i) for i in itertools.count()
                       if not os.path.isdir(result_dir + str(i)))
    os.makedirs(result_dir)

    # return the directory name for the results to go in
    return result_dir


def initialise(args):
    ''' This is responsible for setting up the output directory,
        checking the git repo isn't dirty and setting up logging.'''
    # Check version
    if sys.version_info.major != 3:
        sys.exit("Requires python 3.")

    # load experiment file
    settings = yaml.load(args.experiment, Loader=yaml.CSafeLoader)

    settings['data']['dir'] = os.path.abspath(args.data_dir)

    settings['sampling']['dir'] = os.path.abspath(args.sample_dir)

    # create result directory
    result_dir = create_result_dir(args.result_dir, settings['name'])

    # copy experiment config into results directory
    shutil.copy(args.experiment.name, result_dir)

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
        bar = Bar('Parallelised ({})'.format(num_processes),
                  max=len(tasks), suffix='%(index)d/%(max)d : %(elapsed)ds')
        bar.update()
        # uses unordered map to ensure results are dumped as soon as available
        for i, result in enumerate(pool.imap_unordered(learn_bool_net, tasks)):
            config_tools.dump_results_partial(result, out_stream, i == 0)
            bar.next()
        bar.finish()


def scoop_worker_wrapper(*args, **kwargs):
    import traceback
    try:
        # call to actual worker code
        return learn_bool_net(*args, **kwargs)
    except:
        type, value, tb = sys.exc_info()
        lines = traceback.format_exception(type, value, tb)
        print(''.join(lines))
        raise


def run_scooped(tasks, out_stream):
    ''' runs the given configurations '''
    suffix_fmt = 'completed: %(index)d/%(max)d | elapsed: %(elapsed)ds'
    bar = Bar('Scooped', max=len(tasks), suffix=suffix_fmt)
    bar.update()
    # uses unordered map to ensure results are dumped as soon as available
    for i, result in enumerate(scoop.futures.map_as_completed(
            scoop_worker_wrapper, tasks)):
        config_tools.dump_results_partial(result, out_stream, i == 0)
        bar.next()
    bar.finish()


def run_sequential(tasks, out_stream):
    ''' runs the given configurations '''
    bar = Bar('Sequential', max=len(tasks),
              suffix='%(index)d/%(max)d : %(elapsed)ds')
    bar.update()
    # map gives an iterator so results are dumped as soon as available
    for i, result in enumerate(map(learn_bool_net, tasks)):
        config_tools.dump_results_partial(result, out_stream, i == 0)
        bar.next()
    bar.finish()


def notify(pb_handle, exp_name, result_dirname, time):
    result_dirname = str(result_dirname)
    print('Experiment completed in {} seconds. Results in \"{}\"'.
          format(time, result_dirname))
    if pb_handle:
        pb_handle.push_note(
            'Experiment complete.', 'name: {} time: {} results: {}'.
            format(exp_name, time, result_dirname))


# ############################## MAIN ####################################### #
def main():
    start_time = time()

    args = parse_arguments()

    if not args.no_notify:
        try:
            from pushbullet import PushBullet, PushbulletError
            pb = PushBullet('on6qP2blHZbxs5h0xhDRcnfxHLoIc9Jo')
        except ImportError:
            print('Failed to import PushBullet.')
            pb = None
        except PushbulletError:
            print('Failed to generate PushBullet interface.')
            pb = None

    settings, result_dir = initialise(args)

    print('Directories initialised. Results in: ' + result_dir)

    # generate learning tasks
    configurations = config_tools.generate_configurations(settings)
    print('Done: {} configurations generated.'.format(len(configurations)))
    tasks = config_tools.generate_tasks(configurations)
    print('Done: {} tasks generated.'.format(len(tasks)))

    with open(os.path.join(result_dir, 'results.json'), 'w') as results_stream:
        # Run the actual learning as a parallel process
        run_tasks(tasks, args.numprocs, results_stream)

    print('Runs completed.')

    total_time = time() - start_time

    if not args.no_notify:
        try:
            notify(pb, settings['name'], result_dir, total_time)
        except PushbulletError as err:
            print('Failed to send PushBullet notification: {}.'.format(err))

if __name__ == '__main__':
    main()
