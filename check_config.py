import os                           # for mkdir
import os.path                      # for path manipulation
import yaml                         # for loading experiment files
import argparse                     # CLI

import boolnet.exptools.config_tools as config_tools


def main():
    parser = argparse.ArgumentParser(description='Config file tester - to help'
                                                 ' debug config file problems '
                                                 'w/o running experiments.')

    parser.add_argument('file',
                        type=argparse.FileType('r'),
                        help='experiment config filename.')
    parser.add_argument('--data-dir', default='experiments/datasets',
                        type=str, help='test data directory.')
    parser.add_argument('--results-dir', default='experiments/results',
                        type=str, help='test result directory.')

    args = parser.parse_args()

    # load experiment file
    settings = yaml.load(args.file, Loader=yaml.CSafeLoader)

    # MUST FIX THIS SINCE BASE_DIR will be code, not above
    settings['data']['dir'] = os.path.abspath(args.data_dir)

    settings['learner']['inter_file_base'] = os.path.join(args.results_dir,
                                                          'temp', 'inter_')

    # test generation of tasks
    config_tools.generate_configurations(settings)


if __name__ == '__main__':
    main()
