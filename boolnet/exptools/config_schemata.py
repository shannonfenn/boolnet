from good import (
    Schema, message, All, Any, Range, Allow, Reject, Default, Match, Msg,
    In, Optional, Exclusive, Length, Invalid, Entire, truth)
import os.path
import boolnet.bintools.functions as fn


def conditionally_required(trigger_key, trigger_vals, required_key):
    ''' if trigger_key = trigger_val then required_key must be present.'''
    def validator(d):
        required = (trigger_key in d and d[trigger_key] in trigger_vals)
        if required and required_key not in d:
            raise Invalid('\'{}\' in \'{}\' requires \'{}\'.'.format(
                trigger_key, trigger_vals, required_key))
        # Return the dictionary
        return d
    return validator


def conditionally_forbidden(trigger_key, trigger_vals, required_key):
    ''' if trigger_key = trigger_val then required_key must NOT be present.'''
    def validator(d):
        required = (trigger_key in d and d[trigger_key] in trigger_vals)
        if required and required_key in d:
            raise Invalid('\'{}\' = \'{}\' forbids \'{}\'.'.format(
                trigger_key, trigger_vals, required_key))
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


@message('Must be a valid directory.')
def IsDir(d):
    d = os.path.expanduser(d)
    assert os.path.isdir(d)
    return d


guiding_functions = fn.scalar_function_names()

seed_schema = Any(Default(None), str, All(int, Range(min=0)))

target_subset_schema = Any('random', [All(int, Range(min=0))])

data_schema = Any(
    # generated from operator
    Schema({
        'type':                     'generated',
        'operator':                 str,
        'bits':                     All(int, Range(min=1)),
        Optional('out_width'):      All(int, Range(min=1)),
        Optional('add_noise'):      Range(min=0.0),
        Optional('targets'):        target_subset_schema,
        }),
    # read from file
    Schema({
        'type':                 'file',
        'filename':             str,
        Optional('dir'):        IsDir,
        Optional('add_noise'):  Range(min=0.0),
        Optional('targets'):    target_subset_schema,
        }),
    # pre-split, read from file
    Schema({
        'type':                 'split',
        'training_filename':    str,
        'test_filename':        str,
        Optional('dir'):        IsDir,
        Optional('add_noise'):  Range(min=0.0),
        Optional('targets'):    target_subset_schema,
        })
    )

sampling_schema = Any(
    # randomly generated
    Schema({
        'type':           'generated',
        'Ns':             All(int, Range(min=1)),
        'Ne':             All(int, Range(min=1)),
        Optional('seed'): seed_schema,
        Optional('test'): All(int, Range(min=0))
        }),
    # read from file
    Schema({
        'type':             'file',
        'filename':         str,
        Optional('test'):   str,
        Optional('dir'):    IsDir,
        }),
    # given in config file
    Schema({
        'type':             'given',
        'indices':          [[All(int, Range(min=0))]],
        Optional('test'):   [[All(int, Range(min=0))]],
        }),
    # blank - data is already split
    Schema({'type': 'blank'})
    )


network_schema = Schema({
    'Ng':           Any(All(int, Range(min=1)), Match('^[1-9][0-9]*n2?$')),
    'node_funcs':   [All(int, Range(min=0, max=15))]
    })

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
        Optional('return_option'):               In(['best', 'last']),
        Optional('stopping_condition'):          stopping_condition_schema,
        Optional('max_restarts'):                All(int, Range(min=0))
        }),
    # Hill Climbing
    Schema({
        'name':                                  'HC',
        'max_iterations':                        Any(All(int, Range(min=1)),
                                                     Match('^[1-9][0-9]*n$')),
        'guiding_function':                      In(guiding_functions),
        Optional('guiding_function_parameters'): Schema({}, extra_keys=Allow),
        Optional('return_option'):               In(['best', 'last']),
        Optional('stopping_condition'):          stopping_condition_schema,
        Optional('max_restarts'):                All(int, Range(min=0))
        }),
    # Late-Acceptance Hill Climbing
    Schema({
        'name':                                  'LAHC',
        'cost_list_length':                      All(int, Range(min=1)),
        'max_iterations':                        Any(All(int, Range(min=1)),
                                                     Match('^[1-9][0-9]*n$')),
        'guiding_function':                      In(guiding_functions),
        Optional('guiding_function_parameters'): Schema({}, extra_keys=Allow),
        Optional('return_option'):               In(['best', 'last']),
        Optional('stopping_condition'):          stopping_condition_schema,
        Optional('max_restarts'):                All(int, Range(min=0))
        }),
    )

solver_params_schema = Any(
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

minfs_schema = Schema({
    'solver':                       Any('cplex', 'greedy', 'raps'),
    'metric':                       Any('cardinality>first',
                                        'cardinality>random',
                                        'cardinality>entropy',
                                        'cardinality>feature_diversity',
                                        'cardinality>pattern_diversity'),
    Optional('solver_params'):      solver_params_schema,
    Optional('provide_prior_soln'): bool
    })

target_order_schema = Any('auto', 'msb', 'lsb', 'random',
                          All(list, permutation))
mask_schema = Any(bool, 'prefilter')
learner_schema = Schema(All(
    Any(
        Schema({
            'name':                         'monolithic',
            'network_params':               network_schema,
            'target_order':                 target_order_schema,
            Optional('seed'):               seed_schema,
            Optional('minfs_params'):       minfs_schema,
            }),
        Schema({
            'name':                         Any('stratified', 'stratmultipar'),
            'network_params':               network_schema,
            'target_order':                 target_order_schema,
            Optional('seed'):               seed_schema,
            Optional('minfs_params'):       minfs_schema,
            Optional('prefilter'):          Any(Default(''),
                                                'prev-strata+input',
                                                'prev-strata+prev-fs',
                                                'prev-strata+prev-fs+input'),
            Optional('apply_mask'):         mask_schema,
            Optional('shrink_subnets'):     bool,
            Optional('early_terminate'):    bool,
            }),
        Schema({
            'name':                         'classifierchain',
            'network_params':               network_schema,
            'target_order':                 target_order_schema,
            Optional('seed'):               seed_schema,
            Optional('minfs_params'):       minfs_schema,
            Optional('early_terminate'):    bool,
            }),
        Schema({
            'name':                         'ecc_member',
            'network_params':               network_schema,
            Optional('target_order'):       target_order_schema,
            Optional('seed'):               seed_schema,
            }),
        Schema({
            'name':                         Any('split',
                                                'classifierchain_plus'),
            'network_params':               network_schema,
            'target_order':                 target_order_schema,
            Optional('seed'):               seed_schema,
            Optional('minfs_params'):       minfs_schema,
            Optional('apply_mask'):         mask_schema,
            Optional('early_terminate'):    bool,
            }),
        # a bit hacky for now - the learner will have to catch config errors
        Schema({
            'name':                         'wrapper',
            'sublearner':                   str,
            'curricula_method':             Any('minfs', 'CEbCC',
                                                'label_effects'),
            Reject('target_order'):         None,
            Optional('options'):            Schema({}, extra_keys=Allow)
            }, extra_keys=Allow),
        ),
    conditionally_required('apply_mask', [True], 'minfs_params'),
    # conditionally_required('target_order', ['auto'], 'minfs_params'),
    ))


log_keys_schema = Schema(
    [All([str, bool, list], Length(min=3, max=3), valid_log_key)]
    )


instance_schema = Schema({
    'data':         data_schema,
    'learner':      learner_schema,
    'optimiser':    optimiser_schema,
    'sampling':     sampling_schema,
    'log_keys':     log_keys_schema,
    Optional(Match(r'^notes.*')):  str,
    Optional('verbose_errors'):             bool,
    Optional('verbose_timing'):             bool,
    Optional('record_final_net'):           bool,
    Optional('record_intermediate_nets'):   bool,
    })


# ########## Schemata for base configs ########## #
list_msg = '\'list\' must be a sequence of mappings.'
prod_msg = '\'product\' must be a sequence of sequences of mappings.'
experiment_schema = Schema({
    'name':         str,
    'seed':         seed_schema,
    # Must be any dict
    'base_config':  Schema({}, extra_keys=Allow),
    # only one of 'list_config' or 'product_config' are allowed
    # Must be a list of dicts
    Optional('list'):           Msg(All(Schema([Schema({},
                                                extra_keys=Allow)]),
                                        Length(min=1)), list_msg),
    # Must be a length >= 2 list of lists of dicts
    Optional('product'):        Msg(All([All(Schema([Schema({},
                                                     extra_keys=Allow)]),
                                             Length(min=1))],
                                        Length(min=2)), prod_msg),
    # optional level of debug logging
    Optional('debug_level'):    In(['none', 'warning', 'info', 'debug']),

    Entire: Exclusive('list', 'product')
    })
