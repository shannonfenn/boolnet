from random import random
from math import exp
from copy import copy
from itertools import chain, repeat
from collections import deque
from BoolNet.BitError import metric_from_name
import sys
import logging


def stepped_exp_decrease(init_temp, rate, num_temps, steps_per_temp):
    return chain.from_iterable(repeat(t, steps_per_temp)
                               for t in geometric(init_temp, rate, num_temps))


def geometric(a, r, n):
    val = a
    for i in range(n):
        yield val
        val *= r


class SA:
    def accept(self, old_error, new_error, temperature):
        delta = new_error - old_error
        # DETERMINISTIC
        # return delta < -temperature or 0 <= delta < temperature
        # STOCHASTIC
        # Accept all movements for which dE is non-negative
        # or accept with probability P = e^(-dE/T)
        return (delta <= 0) or (temperature > 0 and
                                random() < exp(-delta/temperature))

    def move(self, state, temp):
        state.move_to_neighbour(state.random_move())

    def undo_move(self, state):
        state.revert_move()

    def initialise(self, parameters):
        # unpack options
        try:
            self.num_temps = parameters['num_temps']
            self.steps_per_temp = parameters['steps_per_temp']
            self.init_temp = parameters['init_temp']
            self.temp_rate = parameters['temp_rate']
            self.guiding_metric = metric_from_name(parameters['metric'])
        except KeyError:
            print('Optimiser parameters missing!', sys.stderr)
            raise
        self.temperatures = stepped_exp_decrease(
            self.init_temp, self.temp_rate, self.num_temps, self.steps_per_temp)

    def run(self, evaluator, parameters, end_condition):
        """This learns a network using SA."""
        # initialise
        self.initialise(parameters)

        state = evaluator.network

        # Calculate initial error
        error = evaluator.metric_value(self.guiding_metric)

        # Setup aspiration criteria
        best_error = error
        best_state = copy(state)
        best_iteration = 0

        best_error_for_temp = error

        last_temp = self.init_temp  # so we can check for temp changes

        # REMOVE
        # f_err = open('error.log', 'w')
        # print(error, file=f_err)
        # REMOVE

        # annealing loop
        for iteration, temp in enumerate(self.temperatures):
            # Stop on user defined condition
            if end_condition(evaluator, best_error):
                break

            # Log error on new temperature if logging
            best_error_for_temp = min(best_error_for_temp, error)
            if last_temp != temp:
                logging.debug('temp: %f best_err:%f ',
                              last_temp, best_error_for_temp)
                best_error_for_temp = error
                last_temp = temp

            # perform random move
            self.move(state, temp)

            # calculate error for new state
            new_error = evaluator.metric_value(self.guiding_metric)

            # Keep best state seen
            if new_error < best_error:
                best_state = copy(state)
                best_error = new_error
                best_iteration = iteration

            # Determine whether to accept the new state
            if self.accept(error, new_error, temp):
                error = new_error
            else:
                self.undo_move(state)

            # In any case, clear the move history to save memory
            # and prevent accidentally undoing accepted moves later
            state.clear_history()

            # REMOVE
            # print(error, file=f_err)
            # REMOVE

        # REMOVE
        # f_err.close()
        # REMOVE

        return (best_state, best_iteration, iteration)


class LAHC:
    def move(self, state, iteration):
        state.move_to_neighbour(state.random_move())

    def undo_move(self, state):
        state.revert_move()

    def initialise(self, parameters):
        # unpack options
        try:
            self.cost_list_len = parameters['cost_list_length']
            self.max_iterations = parameters['max_iterations']
            self.guiding_metric = metric_from_name(parameters['metric'])
        except KeyError:
            print('Optimiser parameters missing!', sys.stderr)
            raise

    def run(self, evaluator, parameters, end_condition):
        # unpack options
        self.initialise(parameters)

        # initialise state
        state = evaluator.network

        # Calculate initial error
        error = evaluator.metric_value(self.guiding_metric)

        # set up aspiration criteria
        best_error = error
        best_state = copy(state)
        best_iteration = 0

        # initialise cost list
        self.costs = deque(repeat(error, self.cost_list_len))

        # optimisation loop
        for iteration in range(self.max_iterations):
            # Stop on user defined condition
            if end_condition(evaluator, best_error):
                break

            # perform random move
            self.move(state, iteration)

            # calculate error for new state
            new_error = evaluator.metric_value(self.guiding_metric)

            # Keep best state seen
            if new_error < best_error:
                best_state = copy(state)
                best_error = new_error
                best_iteration = iteration

            # Determine whether to accept the new state
            if self.accept(new_error, error):
                error = new_error
            else:
                self.undo_move(state)

            # In any case, clear the move history to save memory
            # and prevent accidentally undoing accepted moves later
            state.clear_history()

        return (best_state, best_iteration, iteration)

    def accept(self, new_error, current_error):
        oldest_error = self.costs.popleft()
        if new_error <= current_error:
            self.costs.append(new_error)
            return True
        elif new_error < oldest_error:
            self.costs.append(min(new_error, self.costs[-1]))
            return True
        else:
            self.costs.append(min(oldest_error, self.costs[-1]))
            return False


# ###### Variable Neighbourhood versions ###### #
# class SA_VN(SA):
#     def move(self, state, temp):
#         num_moves = self.temp_to_move_map[temp]
#         for i in range(num_moves):
#             state.move_to_neighbour(state.random_move())

#     def undo_move(self, state):
#         state.revert_all_moves()

#     def initialise(self, parameters):
#         super().initialise(parameters)
#         try:
#             self.init_move_count = parameters['init_move_count']
#         except KeyError:
#             print('Optimiser parameters missing!', sys.stderr)
#             raise
#         # generates a table of evenly space integers starting in the
#         # range [init_move_count, 1] for each temperature as a LUT
#         move_schedule = np.ceil(np.linspace(
#             self.init_move_count, 1, self.num_temps,
#             endpoint=False)) - 1
#         move_schedule = np.array(move_schedule, dtype=int)
#         temp_schedule = np.unique(list(stepped_exp_decrease(
#             self.init_temp, self.temp_rate, self.num_temps, self.steps_per_temp)))
#         self.temp_to_move_map = dict(zip(temp_schedule, move_schedule))


# class LAHC_VN(LAHC):
#     def move(self, state, iteration):
#         num_moves = int(np.ceil(self.init_move_count * (
#             1 - iteration / self.max_iterations)))
#         for i in range(num_moves):
#             state.move_to_neighbour(state.random_move())

#     def undo_move(self, state):
#         state.revert_all_moves()

#     def initialise(self, parameters):
#         super().initialise(parameters)
#         # unpack options
#         try:
#             self.init_move_count = parameters['init_move_count']
#         except KeyError:
#             print('Optimiser parameters missing!', sys.stderr)
#             raise


# #### THIS IS NOT ACTUALLY TABU SEARCH ##### #
# class TabuSearch:
#     def initialise(self, parameters, state):
#         # unpack options
#         try:
#             self.tabu_period = parameters['tabu_period']
#             self.max_iterations = parameters['max_iterations']
#             self.guiding_metric = metric_from_name(parameters['metric'])
#         except KeyError:
#             print('Optimiser parameters missing!', sys.stderr)
#             raise

#         if self.tabu_period >= state.Ng:
#             raise ValueError('Tabu period too long!')

#         self.tabu_table = np.zeros(state.Ng, dtype=int)

#     def move(self, state):
#         move = state.random_move()
#         state.move_to_neighbour(move)
#         return move

#     def undo_move(self, state):
#         state.revert_move()

#     def accept(self, old_error, new_error, best_error, move):
#         # aspiration criteria
#         if new_error < best_error:
#             return True
#         # accept if improvement and not tabu
#         elif new_error <= old_error and self.tabu_table[move[0]] == 0:
#             return True
#         else:
#             return False

#     def run(self, evaluator, state_idx, parameters, end_condition):
#         """This learns a network using SA."""
#         # initialise
#         state = evaluator.network(state_idx)
#         self.initialise(parameters, state)

#         # Calculate initial error
#         error = evaluator.metric_value(state_idx, self.guiding_metric)

#         # Setup aspiration criteria
#         best_error = error
#         best_state = copy(state)
#         best_iteration = 0

#         # REMOVE
#         # f_err = open('error.log', 'w')
#         # print(error, file=f_err)
#         # REMOVE

#         # search loop
#         for iteration in range(self.max_iterations):
#             # Stop on user defined condition
#             if end_condition(evaluator, state_idx, best_error):
#                 break

#             # update tabu table
#             self.tabu_table = np.maximum(self.tabu_table - 1, 0)

#             # perform random move
#             move = self.move(state)

#             # calculate error for new state
#             new_error = evaluator.metric_value(state_idx, self.guiding_metric)

#             # Keep best state seen
#             if new_error < best_error:
#                 best_state = copy(state)
#                 best_error = new_error
#                 best_iteration = iteration

#             # Determine whether to accept the new state
#             if self.accept(error, new_error, best_error, move):
#                 error = new_error
#                 # expiration
#                 self.tabu_table[move[0]] = self.tabu_period
#             else:
#                 self.undo_move(state)

#             # In any case, clear the move history to save memory
#             # and prevent accidentally undoing accepted moves later
#             state.clear_history()

#             # REMOVE
#             # print(error, file=f_err)
#             # REMOVE

#         # REMOVE
#         # f_err.close()
#         # REMOVE

#         return (best_state, best_iteration, iteration)

# def anneal_with_gate_adding(seed_state, parameters, log_errors=False):
#     """This learns a network using SA."""
#     # unpack options
#     try:
#         num_t           = parameters['num temps']
#         steps_per_t     = parameters['steps per temp']
#         init_t          = parameters['init temp']
#         rate            = parameters['temp rate']
#         guiding_metric  = metric_from_name(parameters['metric'])
#     except KeyError:
#         print('Optimiser parameters missing!', sys.stderr)
#         raise

#     # initialise
#     state = copy(seed_state)
#     best_state = copy(seed_state)
#     temperatures = chain.from_iterable( repeat(t, steps_per_t)
#                                         for t in geometric(init_t, rate, num_t) )
#     gate_adding_on = ( final_Ng != state.Ng() )

#     # REMOVE LATER - For logging network as each new bit is learnt
#     # NOTPRINTED = [True]*state.No()

#     # Calculate initial error
#     best_error = error = state.error(guiding_metric)

#     if log_errors:
#         best_error_for_temp = error
#         print('BestErrorPerTemp: [')

#     if gate_adding_on:
#         gate_add_delay = (num_t * steps_per_t) / (final_Ng - state.Ng())
#         steps_until_gate_addition = gate_add_delay

#     last_temp = init_t # so we can check for temp changes

#     # actual annealing loop
#     for temp, iteration in zip(temperatures, count()):
#         if error <= 0:
#             break

#         # Log error on new temperature if logging
#         if log_errors:
#             best_error_for_temp = min(best_error_for_temp, error)
#             if last_temp != temp:
#                 print('{},'.format(best_error_for_temp), end=' ')
#                 sys.stdout.flush()
#                 best_error_for_temp = error

#         if gate_adding_on and iteration % gate_add_delay == 0:
#             state.addGates(1)
#             # Ensure adding gates hasn't changed the error
#             assert state.error(guiding_metric) == error

#         # perform random move
#         state.moveToNeighbour(state.randomMove())

#         # calculate error for new state
#         new_error = state.error(guiding_metric)

#         # REMOVE LATER - THIS SIMPLY PRINTS OUT THE TABLE OF ALL GATE OUTPUTS ONCE THE OPTIMISER
#         #                HITS 0 ERROR ON EACH BIT

#         # perBitErrs = state.errorPerBit()
#         # for i in xrange(state.No()):
#         #     if NOTPRINTED[i] and perBitErrs[i] == 0.0:
#         #         NOTPRINTED[i] = False
#         #         print >> sys.stderr, '\nbit {}'.format(i)
#         #         for stateVector in state.fullStateTableForSamples():
#         #             print >> sys.stderr, stateVector
#         #         print >> sys.stderr, state.gates()

#         # REMOVE UP TO HERE

#         # Keep best state seen
#         if new_error < best_error:
#             best_state = copy(state)
#             best_error = new_error
#             best_iteration = iteration

#         # Determine whether to accept the new state
#         if accept(error, new_error, temp):
#             error = new_error
#             state.clearHistory()
#         else:
#             state.revertMove()

#     # finish error log list if logging
#     if log_errors:
#         print(']', end=' ')
#         sys.stdout.flush()

#     return (best_state, best_iteration, iteration)
