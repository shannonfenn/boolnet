from copy import deepcopy
from collections import MutableMapping, namedtuple
from os import fsync
from os.path import join, splitext
import numpy as np
import json

from boolnet.exptools.boolmapping import BoolMapping
from boolnet.exptools.config_schemata import config_schema


Instance = namedtuple('Instance', [
    'training_mapping', 'test_mapping', 'training_indices'])


class ExperimentJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, BoolMapping):
            return obj.toDict()
        return json.JSONEncoder.default(self, obj)


def dump_configurations(configurations, stream):
    first = True
    for conf in configurations:
        stream.write('[' if first else ', ')
        json.dump(conf[0], stream, cls=ExperimentJSONEncoder)
    stream.write(']')


def dump_results_partial(results, stream, first):
    if not first:
        stream.write(',')
    json.dump(results, stream, cls=ExperimentJSONEncoder)
    stream.write('\n')
    # ensure data is written to disk immediately
    stream.flush()
    fsync(stream.fileno())


def update_nested(d, u):
    ''' this updates a dict with another where the two may contain nested
        dicts themselves (or more generally nested mutable mappings). '''
    for k, v in u.items():
        if isinstance(v, MutableMapping):
            r = update_nested(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


def pack_examples(inputs, targets, training_indices):
    ''' Parititions the given function into training and test sets,
        based on the given training indices.'''
    # Parameters
    N = inputs.shape[0]
    Ns, Ne = training_indices.shape
    # Generate the test indices array, each row should contain all
    # indices not in the equivalent row of training_indices
    test_indices = np.zeros(shape=(Ns, N - Ne), dtype=int)
    for s in range(Ns):
        test_indices[s] = np.setdiff1d(np.arange(N), training_indices[s])
    # Using numpy's advanced indexing we can get the sets
    training_inps = inputs[training_indices]
    training_tgts = targets[training_indices]
    test_inps = inputs[test_indices]
    test_tgts = targets[test_indices]

    # build list of train/test set instances
    instances = [Instance(
        training_mapping=BoolMapping(training_inps[i], training_tgts[i], Ne),
        test_mapping=BoolMapping(test_inps[i], test_tgts[i], N - Ne),
        training_indices=training_indices[i],
        ) for i in range(Ns)]

    return instances


def load_dataset(settings):
    data_settings = settings['data']
    sampling_settings = settings['sampling']

    if data_settings['type'] == 'file':
        return file_instance(data_settings, sampling_settings)
    elif data_settings['type'] == 'generated':
        return generated_instance(data_settings, sampling_settings)
    else:
        raise ValueError('Invalid dataset type {}'.format(data_settings['type']))


def load_samples(params, data_dir, inputs):
    if params['method'] == 'given':
        # load samples from file
        # prepare filename
        N, Ni = inputs.shape
        Ns = params['Ns']
        Ne = params['Ne']
        if 'file_suffix' in params:
            base_name = '{}_{}_{}{}.npy'.format(Ni, Ns, Ne, params['file_suffix'])
        else:
            base_name = '{}_{}_{}.npy'.format(Ni, Ns, Ne)

        # load sample indices
        sample_filename = join(data_dir, 'samples', base_name)
        training_indices = np.load(sample_filename)
    elif params['method'] == 'generated':
        # generate samples
        Ns = params['Ns']
        Ne = params['Ne']
        # generate
        training_indices = np.random.randint(N, size=(Ns, Ne))
    else:
        raise ValueError('Invalid sampling method {}'.format(
                         params['method']))
    return training_indices


def file_instance(data_settings, sampling_settings):
    data_dir = data_settings['dir']
    # load data set from file
    dataset_filename = join(data_dir, 'functions', data_settings['filename'])
    if splitext(dataset_filename)[-1] == '':
        dataset_filename += '.npz'
    with np.load(dataset_filename) as dataset:
        inputs = dataset['input_matrix']
        targets = dataset['target_matrix']

    training_indices = load_samples(sampling_settings, data_dir, inputs)
    # partition the sets based on loaded indices
    return pack_examples(inputs, targets, training_indices)


def generated_instance(data_settings):
    # do the things
    raise NotImplementedError


def handle_initial_network(settings):
    net_method = settings['network']['method']
    if net_method == 'given':
        data_dir = settings['data_dir']
        filename = settings['network']['file']
        index = settings['network']['index']
        with open(join(data_dir, filename)) as f:
            gates = np.array(json.load(f)[index], dtype=np.uint32)
            settings['initial_gates'] = gates


def get_config_indices(instances, config_settings):
    # samples may be optionally sub-indexed
    if 'indices' in config_settings['sampling']:
        config_indices = config_settings['sampling']['indices']
        if any(i >= len(instances) for i in config_indices):
            raise ValueError('\"sampling\" -> indices has elements larger than Ns')
    else:
        config_indices = range(len(instances))
    return config_indices


def generate_configurations(settings, evaluator_class):
    # CAUTION: Will modify the settings parameter!
    # the configurations approach involves essentially having
    # a new settings dict for each configuration and updating
    # it with values in each dict in the configurations list

    variable_sets = settings['configurations']
    # no need to keep this sub-dict around
    settings.pop('configurations')
    # Build up the task list
    tasks = []
    for config_no, variables in enumerate(variable_sets):
        # keep each configuration isolated
        config_settings = deepcopy(settings)
        # update the settings dict with the values for this configuration
        update_nested(config_settings, variables)
        # record the config number for debuggin
        config_settings['configuration_number'] = config_no
        # load initial network from file if required
        handle_initial_network(config_settings)
        # load the data for this configuration
        instances = load_dataset(config_settings)
        # samples may be optionally sub-indexed
        config_indices = get_config_indices(instances, config_settings)
        # for each training set
        for i in config_indices:
            instance = instances[i]

            iteration_settings = deepcopy(config_settings)
            iteration_settings['training_indices'] = instance.training_indices
            iteration_settings['training_set'] = instance.training_mapping
            iteration_settings['test_set'] = instance.test_mapping
            iteration_settings['training_set_number'] = i
            iteration_settings['inter_file_base'] += '{}_{}_'.format(config_no, i)

            config_schema(iteration_settings)

            # dump the iteration settings out
            tasks.append((iteration_settings, evaluator_class))
    return tasks