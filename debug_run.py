from boolnet.exptools.config_tools import generate_configurations
from boolnet.learning.learn_boolnet import (build_initial_network,
                                            build_training_state)
import boolnet.learning.learners as learners
import boolnet.learning.optimisers as optimisers
import numpy as np
import argparse
import yaml

# params = {
#     'data': {
#         'type': 'generated',
#         'dir': '/home/shannon/HMRI/experiments/datasets/samples/',
#         'bits': 5,
#         'operator': 'add'
#     },
#     'network': {
#         'Ng': 84,
#         'node_funcs': 'NAND'
#     },
#     'learner': {
#         'optimiser': {'guiding_function': 'e3L'},
#         'auto_target': False,
#         'kfs': True,
#         'inter_file_base': '/home/shannon/HMRI/shan_test/'
#     },
#     'configurations': [
#         {
#             'sampling': {
#                 'method': 'given',
#                 'Ns': 250,
#                 'Ne': 144,
#                 'indices': [161]
#             }
#         }
#     ]
# }


def main(config, config_num, trg_set_num):
    all_tasks = generate_configurations(config)

    tasks = [t for t in all_tasks
             if t['configuration_number'] == config_num and
             t['training_set_number'] == trg_set_num]

    if len(tasks) != 1:
        ValueError('There is {} matching tasks but should be only 1.'.
                   format(len(tasks)))

    params = tasks[0]

    training_data = params['training_mapping']

    gates = build_initial_network(params, training_data)
    state = build_training_state(gates, training_data)

    learner = learners.StratifiedLearner()
    optimiser = optimisers.LAHC()

    learner._setup(params['learner'], state, optimiser)

    print('gate boundaries:', learner.gate_boundaries)

    learner.learned_targets = [0, 1, 2, 3]

    learner.feature_sets[4, 4] = np.array([2, 4, 9, 50, 84, 89])

    optimisation_required = learner._apply_mask(state, 4)

    print('optimisation required:', optimisation_required)

    print(state)

    print(learner.feature_sets[4, 4])

    print('After randomisation')

    state.randomise()
    print(state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('file',
                        type=argparse.FileType('r'),
                        help='yaml config filename.')
    parser.add_argument('config_num', type=int)
    parser.add_argument('trg_set_num', type=int)

    args = parser.parse_args()

    config = yaml.load(args.file, Loader=yaml.CSafeLoader)

    config['learner']['inter_file_base'] = '/home/shannon/HMRI/shan_test/'
    config['data']['dir'] = '/home/shannon/HMRI/experiments/datasets/'

    main(config, args.config_num, args.trg_set_num)
