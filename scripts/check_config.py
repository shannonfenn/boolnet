#! /usr/bin/env python

import yaml                         # for loading experiment files
try:
    from yaml import CSafeLoader as Loader
except ImportError:
    from yaml import SafeLoader as Loader
import argparse                     # CLI

import boolnet.exptools.config_tools as config_tools


def main():
    parser = argparse.ArgumentParser(description='Config file tester - to help'
                                                 ' debug config file problems '
                                                 'w/o running experiments.')

    parser.add_argument('experiment',
                        type=argparse.FileType('r'),
                        help='experiment config filename.')
    parser.add_argument('-r', '--result-dir', type=str, metavar='dir',
                        default='experiments/results',
                        help='directory to store results in (in own subdir).')

    args = parser.parse_args()

    # load experiment file
    settings = yaml.load(args.experiment, Loader=Loader)

    # test generation of tasks
    try:
        tasks = config_tools.generate_tasks(settings, False)
        print('{} tasks.'.format(len(tasks)))
    except config_tools.ValidationError as err:
        print()
        print(err)
        print('\nExperiment config is NOT valid.')
        return

    print('\nExperiment config is valid.')


if __name__ == '__main__':
    main()
