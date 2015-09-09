from random import random
from math import exp
from copy import copy
from itertools import chain, repeat
from collections import deque
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
        state.move_to_random_neighbour()

    def undo_move(self, state):
        state.revert_move()

    def initialise(self, parameters):
        # unpack options
        try:
            self.num_temps = parameters['num_temps']
            self.steps_per_temp = parameters['steps_per_temp']
            self.init_temp = parameters['init_temp']
            self.temp_rate = parameters['temp_rate']
            self.guiding_function = parameters['guiding_function']
            self.stopping_criterion = parameters['stopping_criterion']
        except KeyError:
            print('Optimiser parameters missing!', sys.stderr)
            raise
        self.temperatures = stepped_exp_decrease(
            self.init_temp, self.temp_rate, self.num_temps, self.steps_per_temp)

    def _optimise(self, state):
        # Calculate initial error
        error = self.guiding_function(state)

        # Setup aspiration criteria
        best_error = error
        best_state = copy(state.representation())
        best_iteration = 0

        best_error_for_temp = error

        last_temp = self.init_temp  # so we can check for temp changes

        # annealing loop
        for iteration, temp in enumerate(self.temperatures):
            # Stop on user defined condition
            if self.stopping_criterion(state, best_error):
                break

            # Log error on new temperature if logging
            best_error_for_temp = min(best_error_for_temp, error)
            if last_temp != temp:
                logging.debug('temp: %f best_err:%f ', last_temp, best_error_for_temp)
                best_error_for_temp = error
                last_temp = temp

            # perform random move
            self.move(state, temp)

            # calculate error for new state
            new_error = self.guiding_function(state)

            # Keep best state seen
            if new_error < best_error:
                best_state = copy(state.representation())
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

        return (best_state, best_iteration, iteration)

    def run(self, state, parameters):
        """This learns a network using SA."""
        # initialise
        self.initialise(parameters)
        return self._optimise(state)


class LAHC:
    def move(self, state, iteration):
        state.move_to_random_neighbour()

    def undo_move(self, state):
        state.revert_move()

    def initialise(self, parameters):
        # unpack options
        try:
            self.cost_list_len = parameters['cost_list_length']
            self.max_iterations = parameters['max_iterations']
            self.guiding_function = parameters['guiding_function']
            self.stopping_criterion = parameters['stopping_criterion']
            self.reached_stopping_criterion = False
        except KeyError:
            print('Optimiser parameters missing!', sys.stderr)
            raise

    def _optimise(self, state):
        # Calculate initial error
        error = self.guiding_function(state)
        # error = evaluator.function_value(self.guiding_function)

        # set up aspiration criteria
        best_error = error
        best_state = copy(state.representation())
        best_iteration = 0

        # initialise cost list
        self.costs = deque(repeat(error, self.cost_list_len))

        self.reached_stopping_criterion = False
        # optimisation loop
        for iteration in range(self.max_iterations):
            # Stop on user defined condition
            if self.stopping_criterion(state, best_error):
                self.reached_stopping_criterion = True
                break

            # perform random move
            self.move(state, iteration)

            # calculate error for new state
            new_error = self.guiding_function(state)
            # new_error = evaluator.function_value(self.guiding_function)

            # Keep best state seen
            if new_error < best_error:
                best_state = copy(state.representation())
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

    def run(self, state, parameters):
        # unpack options
        self.initialise(parameters)

        return self._optimise(state)

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
