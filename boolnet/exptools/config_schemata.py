from good import (
    Schema, In, All, Any, Range, Type, IsDir, Msg,
    message, Optional, Exclusive, Length, Entire)
import boolnet.bintools.functions as fn


def conditionally_required(trigger_key, trigger_val, required_key):
    ''' if trigger_key has trigger_val then required_key must be present.'''
    def validator(d):
        if trigger_key in d and d[trigger_key] == trigger_val:
            assert required_key in d
        # Return the dictionary
        return d
    return validator


@message('Must be a permutation.')
def permutation(l):
    assert (min(l) == 0 and max(l) == len(l) - 1 and len(set(l)) == len(l))
    return l


guiding_functions = fn.scalar_function_names()


data_schema_generated = Schema({
    'type':                     'generated',
    'dir':                      IsDir(),
    'operator':                 str,
    'bits':                     All(int, Range(min=1)),
    Optional('out_width'):      All(int, Range(min=1)),
    Optional('window_size'):    All(int, Range(min=1))
    })

data_schema_file = Schema({
    'type':     'file',
    'dir':      IsDir(),
    'filename': str
    })

sampling_schema = Schema({
    'dir':                      IsDir(),
    'Ns':                       All(int, Range(min=1)),
    'Ne':                       All(int, Range(min=1)),
    Optional('indices'):        [All(int, Range(min=0))],
    Optional('file_suffix'):    str
    })


# network_schema_given = Schema({
#     'method':   'given',
#     'gates':    [All([All(int, Range(min=0))], Length(min=3, max=3))],
#     })

network_schema_generated = Schema({
    'method':       'generated',
    'Ng':           All(int, Range(min=1)),
    'node_funcs':   [All(int, Range(min=0, max=15))]
    })

# network_schema = Any(network_schema_given, network_schema_generated)
network_schema = network_schema_generated


SA_schema = Schema({
    'name':                     'SA',
    'num_temps':                All(int, Range(min=1)),
    'init_temp':                Range(min=0.0),
    'temp_rate':                Range(min=0.0, max=1.0),
    'steps_per_temp':           All(int, Range(min=1)),
    'guiding_function':         In(guiding_functions),
    Optional('max_restarts'):   All(int, Range(min=0))
    })

HC_schema = Schema({
    'name':                     'HC',
    'max_iterations':           All(int, Range(min=1)),
    'guiding_function':         In(guiding_functions),
    Optional('max_restarts'):   All(int, Range(min=0))
    })

LAHC_schema = Schema({
    'name':                     'LAHC',
    'cost_list_length':         All(int, Range(min=1)),
    'max_iterations':           All(int, Range(min=1)),
    'guiding_function':         In(guiding_functions),
    Optional('max_restarts'):   All(int, Range(min=0))
    })


optimiser_schema = Any(SA_schema, HC_schema, LAHC_schema)


target_order_schema = Any('auto', 'msb', 'lsb', All(list, permutation))


learner_schema_stratified = Schema({
    'name':                         'stratified',
    'optimiser':                    optimiser_schema,
    'network':                      network_schema,
    'minfs_selection_method':       str,
    'target_order':                 target_order_schema,
    Optional('minfs_masking'):      bool,
    Optional('keep_files'):         bool,
    })


learner_schema_basic = Schema({
    'name':                             'basic',
    'network':                          network_schema,
    'optimiser':                        optimiser_schema,
    'target_order':                     target_order_schema,
    Optional('minfs_selection_method'): str,
    Entire:
        Msg(  # also override the message
            conditionally_required('target_order', 'auto',
                                   'minfs_selection_method'),
        '\'target_order\' = \'auto\' requires \'minfs_selection_method\'.')
    })


learner_schema = Any(learner_schema_basic, learner_schema_stratified)


experiment_schema = Schema({
    'name': str,
    'data': Any(data_schema_file, data_schema_generated),
    'logging': In(['none', 'warning', 'info', 'debug']),
    'learner': learner_schema,
    'sampling': sampling_schema,
    Optional('verbose_errors'): bool,
    Optional('verbose_timing'): bool,
    Optional('record_final_net'): bool,
    Optional('record_intermediate_nets'): bool,
    Optional('record_training_indices'): bool,
    Optional('seed'): All(int, Range(min=0)),
    })
