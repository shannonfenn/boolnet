from good import (
    Schema, message, All, Any, Range, IsDir, Allow, Default, Match, Msg,
    In, Optional, Required, Exclusive, Length, Invalid, Entire, truth)
import boolnet.bintools.functions as fn
import re


def conditionally_required(trigger_key, trigger_val, required_key):
    ''' if trigger_key = trigger_val then required_key must be present.'''
    def validator(d):
        required = (trigger_key in d and d[trigger_key] == trigger_val)
        if required and required_key not in d:
            raise Invalid('\'{}\' = \'{}\' requires \'{}\'.'.format(
                trigger_key, trigger_val, required_key))
        # Return the dictionary
        return d
    return validator


def conditionally_forbidden(trigger_key, trigger_val, required_key):
    ''' if trigger_key = trigger_val then required_key must NOT be present.'''
    def validator(d):
        required = (trigger_key in d and d[trigger_key] == trigger_val)
        if required and required_key in d:
            raise Invalid('\'{}\' = \'{}\' forbids \'{}\'.'.format(
                trigger_key, trigger_val, required_key))
        # Return the dictionary
        return d
    return validator


@message('Must be a permutation.')
def permutation(l):
    assert (min(l) == 0 and max(l) == len(l) - 1 and len(set(l)) == len(l))
    return l


@truth('log keys must be [<key>, <T/F>, [<path>]')
def valid_log_key(v):
    return (len(v) == 3 and isinstance(v[0], str) and 
            isinstance(v[1], bool) and isinstance(v[2], list))


guiding_functions = fn.scalar_function_names()


seed_schema = Any(None, str, All(int, Range(min=0)))

data_schema = Any(
    # generated from operator
    Schema({
        'type':                     'generated',
        'operator':                 str,
        'bits':                     All(int, Range(min=1)),
        Optional('out_width'):      All(int, Range(min=1)),
        Optional('window_size'):    All(int, Range(min=1)),
        Optional('add_noise'):      Range(min=0.0),
        # Optional('targets'):        [All(int, Range(min=0))],
        }),
    # read from file
    Schema({
        'type':                 'file',
        'filename':             str,
        Optional('dir'):        IsDir(),
        Optional('add_noise'):  Range(min=0.0),
        Optional('targets'):    [All(int, Range(min=0))],
        }),
    # pre-split, read from file
    Schema({
        'type':                 'split',
        'training_filename':    str,
        'test_filename':        str,
        Optional('dir'):        IsDir(),
        Optional('add_noise'):  Range(min=0.0),
        Optional('targets'):    [All(int, Range(min=0))],
        })
    )

sampling_schema = Any(
    # randomly generated
    Schema({
        'type':           'generated',
        'Ns':             All(int, Range(min=1)),
        'Ne':             All(int, Range(min=1)),
        'seed':           seed_schema,
        Optional('test'): All(int, Range(min=0))
        }),
    # read from file
    Schema({
        'type':             'file',
        'filename':         str,
        Optional('test'):   str,
        Optional('dir'):    IsDir(),
        # allow for now, but don't force
        Optional('seed'):   seed_schema,
        }),
    # given in config file
    Schema({
        'type':             'given',
        'indices':          [[All(int, Range(min=0))]],
        Optional('test'):   [[All(int, Range(min=0))]],
        # allow for now, but don't force
        Optional('seed'):   seed_schema,
        }),
    # blank - data is already split
    Schema({'type': 'blank'})
    )


network_schema = All(
    Schema({
        'method':       'generated',
        'Ng':           Any(All(int, Range(min=1)), Match('[1-9][0-9]*n')),
        'node_funcs':   [All(int, Range(min=0, max=15))]
        }),
    # Schema({
    #     'method':   'given',
    #     'gates':    [All([All(int, Range(min=0))], Length(min=3, max=3))],
    #     })
    )

stopping_condition_schema = Any(['guiding', float],
                                [In(guiding_functions), float])

optimiser_schema = Any(
    # Simulated Annealing
    Schema({
        'name':                                  'SA',
        'num_temps':                             All(int, Range(min=1)),
        'init_temp':                             Range(min=0.0),
        'temp_rate':                             Range(min=0.0, max=1.0),
        'steps_per_temp':                        All(int, Range(min=1)),
        'guiding_function':                      In(guiding_functions),
        Optional('guiding_function_parameters'): Schema({}, extra_keys=Allow),
        Optional('return'):                      In(['best', 'last']),
        Optional('stopping_condition'):          stopping_condition_schema,
        Optional('max_restarts'):                All(int, Range(min=0))
        }),
    # Hill Climbing
    Schema({
        'name':                                  'HC',
        'max_iterations':                        All(int, Range(min=1)),
        'guiding_function':                      In(guiding_functions),
        Optional('guiding_function_parameters'): Schema({}, extra_keys=Allow),
        Optional('return'):                      In(['best', 'last']),
        Optional('stopping_condition'):          stopping_condition_schema,
        Optional('max_restarts'):                All(int, Range(min=0))
        }),
    # Late-Acceptance Hill Climbing
    Schema({
        'name':                                  'LAHC',
        'cost_list_length':                      All(int, Range(min=1)),
        'max_iterations':                        All(int, Range(min=1)),
        'guiding_function':                      In(guiding_functions),
        Optional('guiding_function_parameters'): Schema({}, extra_keys=Allow),
        Optional('return'):                      In(['best', 'last']),
        Optional('stopping_condition'):          stopping_condition_schema,
        Optional('max_restarts'):                All(int, Range(min=0))
        }),
    # LAHC with periodic percolation
    Schema({
        'name':                                  'LAHC_perc',
        'cost_list_length':                      All(int, Range(min=1)),
        'max_iterations':                        All(int, Range(min=1)),
        'percolation_period':                    All(int, Range(min=1)),
        'guiding_function':                      In(guiding_functions),
        Optional('guiding_function_parameters'): Schema({}, extra_keys=Allow),
        Optional('return'):                      In(['best', 'last']),
        Optional('stopping_condition'):          stopping_condition_schema,
        Optional('max_restarts'):                All(int, Range(min=0))
        })
    )


fs_selection_metric_schema = Any(
    'cardinality>first', 'cardinality>random', 'cardinality>entropy',
    'cardinality>feature_diversity', 'cardinality>pattern_diversity')

minfs_params_schema = Any(
    # CPLEX
    Schema({
        Optional('time_limit'):       Range(min=0.0),
        }),
    # Meta-RaPS
    Schema({
        Optional('iterations'):             All(int, Range(min=1)),
        Optional('improvement_iterations'): All(int, Range(min=1)),
        Optional('search_magnitude'):       All(float, Range(min=0, max=1)),
        Optional('priority'):               All(float, Range(min=0, max=1)),
        Optional('restriction'):            All(float, Range(min=0, max=1)),
        Optional('improvement'):            All(float, Range(min=0, max=1))
        }),
    )

target_order_schema = Any('auto', 'msb', 'lsb',
                          'random', All(list, permutation))

learner_schema = Schema(
    All(
        Schema({
            'name':         Any('monolithic', 'stratified', 'split'),
            'network':      network_schema,
            'optimiser':    optimiser_schema,
            'target_order': target_order_schema,
            'seed':         Any(None, All(int, Range(min=0)), Default(None)),
            Optional('minfs_masking'):          bool,
            Optional('minfs_solver'):           Any('cplex', 'greedy', 'raps'),
            Optional('minfs_solver_params'):    minfs_params_schema,
            Optional('minfs_selection_metric'): fs_selection_metric_schema,
            }),
        conditionally_required(
            'minfs_masking', True, 'minfs_selection_metric'),
        conditionally_required(
            'target_order', 'auto', 'minfs_selection_metric'),
        # if name monolithic then minfs_masking is not allowed
        conditionally_forbidden('name', 'monolithic', 'minfs_masking'),
        )
    )


log_keys_schema = Schema(
    [All([str, bool, list], Length(min=3, max=3), valid_log_key)]
    )


instance_schema = Schema({
    'data':     data_schema,
    'learner':  learner_schema,
    'sampling': sampling_schema,
    'log_keys': log_keys_schema,
    Optional('notes'):  Schema({}, extra_keys=Allow),
    Optional('verbose_errors'):             bool,
    Optional('verbose_timing'):             bool,
    Optional('record_final_net'):           bool,
    Optional('record_intermediate_nets'):   bool,
    })


# ########## Schemata for base configs ########## #
list_msg = '\'list\' must be a sequence of mappings.'
prod_msg = '\'product\' must be a length 2 sequence of sequences of mappings.'
experiment_schema = Schema({
    'name':                     str,
    # Must be any dict
    'base_config':              Schema({}, extra_keys=Allow),
    # only one of 'list_config' or 'product_config' are allowed
    # Must be a list of dicts
    Optional('list'):    Msg(All(Schema([Schema({}, extra_keys=Allow)]),
                                 Length(min=1)), list_msg),
    # Must be a length 2 list of lists of dicts
    Optional('product'): Msg(All([All(Schema([Schema({}, extra_keys=Allow)]),
                                      Length(min=1))],
                                 Length(min=2, max=2)), prod_msg),
    # optional level of debug logging
    Optional('debug_level'):        In(['none', 'warning', 'info', 'debug']),
    Entire: Exclusive('list', 'product')

    })
