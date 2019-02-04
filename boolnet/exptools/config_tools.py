from copy import deepcopy
from itertools import product
from collections import MutableMapping
from good import Invalid
import numpy as np
import os
import random

import boolnet.exptools.config_schemata as sch
import boolnet.bintools.operator_iterator as opit
import boolnet.utils as utils


class SeedHandler:
    def __init__(self, primary_seed):
        self.registry = {}
        if primary_seed is None:
            primary_seed = random.randint(0, 2**32-1)
        self.primary_seed = primary_seed
        random.seed(primary_seed)

    def _randint(self):
        return random.randint(1, 2**32-1)

    def _lookup(self, group, key):
        ''' Keeps a registry of seeds for each key, if given a new
            key get_seed() generates a new seed for that key, but if
            given an existing key it returns that seed. Allows any number
            of named seeds.'''
        if group not in self.registry:
            # non-existant group, initialise
            self.registry[group] = {}
        if key not in self.registry[group]:
            # non-existant key, generate a seed
            self.registry[group][key] = self._randint()
        return self.registry[group][key]

    def generate_seed(self, group, seed):
        if seed is None:
            return self._randint()
        elif isinstance(seed, str):
            # s is actually a name
            return self._lookup(group, seed)
        else:
            return seed


def build_filename(params, extension, key='filename'):
    ''' filename helper with optional directory'''
    filename = os.path.expanduser(params[key])
    location = params.get('dir', None)
    # if 'filename' is absolute then ignore 'dir'
    if location and not os.path.isabs(filename):
        filename = os.path.join(location, filename)
    # add extension if missing
    if os.path.splitext(filename)[-1] == '':
        filename += extension
    return filename


def load_samples(seed_handler, params, N, Ni, Nt=None):
    if params['type'] == 'given':
        training_indices = np.array(params['indices'], dtype=np.uintp)
        if 'test' in params:
            test_indices = np.array(params['test'], dtype=np.uintp)
        else:
            test_indices = [None]*training_indices.shape[0]

    if params['type'] == 'file':
        filename = build_filename(params, '.npy')
        training_indices = np.load(filename)
        if 'test' in params:
            filename = build_filename(params, '.npy', 'test')
            test_indices = np.load(filename)
        else:
            test_indices = [None]*training_indices.shape[0]

    elif params['type'] == 'generated':
        # this provided seed allows us to generate the same set
        # of training indices across multiple configurations
        Ns = params['Ns']
        Ne = params['Ne']
        seed = seed_handler.generate_seed('sampling_seeds', params['seed'])
        params['seed'] = seed
        before_sampling = random.getstate()
        random.seed(seed)
        if 'test' in params:
            Ne_test = params['test']
            if Nt is None:
                # combined training and test data
                all_indices = np.array([
                    random.sample(range(N), Ne+Ne_test) for i in range(Ns)])
                training_indices, test_indices = np.hsplit(all_indices, [Ne])
            else:
                # separate training and test data
                training_indices = np.array([
                    random.sample(range(N), Ne) for i in range(Ns)])
                test_indices = np.array([
                    random.sample(range(N), Ne_test) for i in range(Ns)])
        else:
            training_indices = np.array([
                random.sample(range(N), Ne) for i in range(Ns)])
            test_indices = [None]*Ns
        random.setstate(before_sampling)
    return training_indices, test_indices


def load_dataset(settings, seed_handler):
    data_settings = settings['data']

    dtype = data_settings['type']

    if dtype == 'file':
        instance, N, Ni, No = file_instance(data_settings)
    elif dtype == 'split':
        instance, N, Ni, No, Nt = split_instance(data_settings)
    elif dtype == 'generated':
        instance, N, Ni, No = generated_instance(data_settings)

    if 'targets' in data_settings:
        targets = data_settings['targets']
        if targets == 'random':
            targets = list(range(No))
            random.shuffle(targets)
            data_settings['targets'] = targets
    else:
        targets = None
    instance['targets'] = targets


    # check for problematic case
    problematic = (dtype == 'split' and
                   settings['sampling']['type'] == 'blank' and
                   'test' not in settings['sampling'])
    if problematic:
        raise ValueError('Cannot use implicit test sampling with split data.')

    # Only handle sampling if necessary
    if dtype == 'split':
        if settings['sampling']['type'] == 'blank':
            return [instance]
        else:
            training_indices, test_indices = load_samples(
                seed_handler, settings['sampling'], N, Ni, Nt)
    else:
        training_indices, test_indices = load_samples(
            seed_handler, settings['sampling'], N, Ni)

    contexts = []
    for trg, test in zip(training_indices, test_indices):
        context = instance.copy()
        context.update({'training_indices': trg, 'test_indices': test})
        contexts.append(context)
    return contexts


def file_instance(params):
    filename = build_filename(params, '.npz')
    with np.load(filename) as dataset:
        # Ne in the dataset is the full number of examples which we are
        # referring to herein as 'N' to differentiate from the sample size
        N = dataset['Ne']
        Ni = dataset['Ni']
        No = dataset['matrix'].shape[0] - Ni

    instance = {'type': 'file_unsplit',
                'file': filename}

    return instance, N, Ni, No


def split_instance(params):
    trg_filename = build_filename(params, '.npz', key='training_filename')
    test_filename = build_filename(params, '.npz', key='test_filename')
    with np.load(trg_filename) as container:
        Ne_trg = container['Ne']
        Ni = container['Ni']
        No = container['matrix'].shape[0] - Ni
    with np.load(test_filename) as container:
        Ne_test = container['Ne']
        assert Ni == container['Ni']
        assert Ni + No == container['matrix'].shape[0]

    instance = {'type': 'file_split',
                'trg_file': trg_filename,
                'test_file': test_filename}

    return instance, Ne_trg, Ni, No, Ne_test


def generated_instance(params):
    Nb = params['bits']
    operator = params['operator']

    No = params.get('out_width', Nb)  # defaults to operand width

    instance = {
        'type': 'operator',
        'operator': operator,
        'Nb': Nb,
        'No': No
    }
    Ni = opit.num_operands[operator] * Nb

    return instance, 2**Ni, Ni, No


# def handle_initial_network(settings):
#     net_settings = settings['learner']['network']
#     net_method = net_settings['method']
#     if net_method == 'given':
#         data_dir = settings['data']['dir']
#         filename = net_settings['file']
#         index = net_settings['index']
#         with open(join(data_dir, filename)) as f:
#             gates = np.array(json.load(f)[index], dtype=np.uint32)
#             net_settings['initial_gates'] = gates


# to signal schema validation failure
# (with custom message formatting)
class ValidationError(Exception):
    pass


def validate_schema(config, schema, config_num, msg):
    try:
        schema(config)
    except Invalid as err:
        msg = ('Experiment instance {} invalid: {}'
               '\nConfig generation aborted.').format(config_num + 1, err)
        raise ValidationError(msg)


def update_nested(d, u):
    ''' this updates a dict with another where the two may contain nested
        dicts themselves (or more generally nested mutable mappings). '''
    for k, v in u.items():
        if isinstance(v, MutableMapping):
            r = update_nested(d.get(k, {}), v)
            d[k] = r
        else:
            # preference to second mapping if k exists in d
            d[k] = u[k]
    return d


def split_variables_from_base(settings):
    # configuration sub-dicts are popped
    try:
        variable_sets = settings['list']
    except KeyError:
        try:
            # build merged mappings for each combination
            lists = settings['product']
            variable_sets = []
            for combination in product(*lists):
                merged = dict()
                for var_set in combination:
                    update_nested(merged, var_set)
                variable_sets.append(merged)
        except KeyError:
            print('Warning: no variable configuration found.\n')
            variable_sets = [{}]

    return variable_sets, settings['base_config']


def insert_default_log_keys(settings):
    defaults = [
        ['learner', True, ['learner', 'name']],
        ['config_num', True, ['configuration_number']],
        ['trg_set_num', True, ['training_set_number']],
        ['tfs', True, ['learner', 'network_params', 'node_funcs']],
        ['guiding_function', True, ['optimiser', 'guiding_function']],
        ['given_tgt_order', True, ['learner', 'target_order']],
        ['targets', False, ['data', 'targets']],
        ['sample_seed', False, ['sampling', 'seed']],
        ['opt_{}', False, ['optimiser', '.*']],
    ]
    # give preference to user log_keys
    defaults.extend(settings['base_config'].get('log_keys', []))
    settings['base_config']['log_keys'] = defaults
    return settings


def _generate_configurations(variable_sets, base_settings,
                             seed_handler, batch_mode):
    # Build up the configuration list
    configurations = []

    if not batch_mode:
        bar = utils.BetterETABar('Generating configurations',
                                 max=len(variable_sets))
        bar.update()
    try:
        for conf_num, variables in enumerate(variable_sets):
            # keep contexts isolated
            context = deepcopy(base_settings)
            # update the settings dict with the values for this configuration
            update_nested(context, variables)
            # check the given config is a valid experiment
            validate_schema(context, sch.instance_schema, conf_num, variables)
            # record the config number for debugging
            context['configuration_number'] = conf_num
            # load the data for this configuration
            instances = load_dataset(context, seed_handler)

            configurations.append((context, instances))
            if not batch_mode:
                bar.next()
    finally:
        # clean up progress bar before printing anything else
        if not batch_mode:
            bar.finish()
    return configurations


def _generate_tasks(configurations, seed_handler, batch_mode):
    # Build up the task list
    tasks = []

    if not batch_mode:
        bar = utils.BetterETABar('Generating training tasks',
                                 max=len(configurations))
        bar.update()
    try:
        for context, instances in configurations:
            # for each sample
            for i, instance in enumerate(instances):
                task = deepcopy(context)
                task['learner']['seed'] = seed_handler.generate_seed(
                    'learner_seeds', task['learner'].get('seed', None))
                task['mapping'] = instance
                task['training_set_number'] = i
                tasks.append(task)
            if not batch_mode:
                bar.next()
    finally:
        # clean up progress bar before printing anything else
        if not batch_mode:
            bar.finish()
    return tasks


def generate_tasks(settings, batch_mode):
    # validate the given schema
    try:
        sch.experiment_schema(settings)
    except Invalid as err:
        raise ValidationError(
            'Top-level config invalid: {}'.format(err))

    seed_handler = SeedHandler(settings['seed'])
    # insert default log_keys values into base config
    insert_default_log_keys(settings)
    # the configurations approach involves having a multiple config dicts and
    # updating them with each element of the configurations list or product
    variable_sets, base_settings = split_variables_from_base(settings)

    configurations = _generate_configurations(variable_sets, base_settings,
                                              seed_handler, batch_mode)
    tasks = _generate_tasks(configurations, seed_handler, batch_mode)
    return tasks, seed_handler.primary_seed
