from random import random
from math import exp
from copy import copy
from itertools import chain, repeat
from collections import deque, namedtuple
import operator as op
import sys
import logging


OptimiserResult = namedtuple('OptimiserResult', [
    'representation', 'error', 'best_iteration', 'iteration', 'restarts'])


def stepped_exp_decrease(init_temp, rate, num_temps, steps_per_temp):
    return chain.from_iterable(repeat(t, steps_per_temp)
                               for t in geometric(init_temp, rate, num_temps))


def geometric(a, r, n):
    val = a
    for i in range(n):
        yield val
        val *= r


class RestartLocalSearch:

    def initialise(self, parameters):
        # unpack options
        try:
            self.guiding_function = parameters['guiding_function']
            if parameters['minimise']:
                self.is_as_good = op.le
                self.is_better = op.lt
            else:
                self.is_as_good = op.ge
                self.is_better = op.gt
            self.stopping_criterion = parameters['stopping_criterion']
            # self.max_restarts = parameters.get('max_restarts', 0)
            self.max_restarts = parameters.get('max_restarts', 0)
            self.reached_stopping_criterion = False
        except KeyError:
            print('Optimiser parameters missing!', file=sys.stderr)
            raise

    def move(self, state):
        state.move_to_random_neighbour()

    def undo_move(self, state):
        state.revert_move()

    def _optimise(self, state):
        raise NotImplemented

    def run(self, state, parameters):
        # unpack options
        self.initialise(parameters)

        # original_state = state.representation()

        for i in range(self.max_restarts + 1):
            # state.set_representation(original_state)
            step_result = self._optimise(state)
            if self.reached_stopping_criterion:
                break
            elif i < self.max_restarts:
                state.randomise()

        return OptimiserResult(
            representation=step_result.representation,
            error=step_result.error,
            best_iteration=step_result.best_iteration,
            iteration=step_result.iteration,
            restarts=i)


class HC(RestartLocalSearch):

    def initialise(self, parameters):
        super().initialise(parameters)
        try:
            self.max_iterations = parameters['max_iterations']
        except KeyError:
            print('Optimiser parameters missing!', file=sys.stderr)
            raise

    def _optimise(self, state):
        # Calculate initial error
        error = self.guiding_function(state)
        # error = evaluator.function_value(self.guiding_function)

        # set up aspiration criteria
        best_iteration = 0

        self.reached_stopping_criterion = False
        # optimisation loop
        for iteration in range(self.max_iterations):
            # Stop on user defined condition
            if self.stopping_criterion(state, error):
                self.reached_stopping_criterion = True
                break

            # perform random move
            self.move(state)

            # calculate error for new state
            new_error = self.guiding_function(state)
            # new_error = evaluator.function_value(self.guiding_function)

            # Determine whether to accept the new state
            if self.accept(new_error, error):
                error = new_error
                best_iteration = iteration
            else:
                self.undo_move(state)

            # In any case, clear the move history to save memory
            # and prevent accidentally undoing accepted moves later
            state.clear_history()

        return OptimiserResult(
            representation=state.representation,
            error=error,
            best_iteration=best_iteration,
            iteration=iteration,
            restarts=None)

    def accept(self, new_error, current_error):
        if self.is_as_good(new_error, current_error):
            return True


class LAHC(RestartLocalSearch):

    def initialise(self, parameters):
        super().initialise(parameters)
        try:
            self.cost_list_len = parameters['cost_list_length']
            self.max_iterations = parameters['max_iterations']
        except KeyError:
            print('Optimiser parameters missing!', file=sys.stderr)
            raise

    def _optimise(self, state):
        # Calculate initial error
        error = self.guiding_function(state)

        # set up aspiration criteria
        best_error = error
        best_representation = copy(state.representation)
        best_iteration = 0

        # initialise cost list
        self.costs = deque(repeat(error, self.cost_list_len))

        if self.stopping_criterion(state, best_error):
            self.reached_stopping_criterion = True
            return OptimiserResult(
                representation=best_representation, error=best_error,
                best_iteration=0, iteration=0, restarts=None)
        else:
            self.reached_stopping_criterion = False

        # optimisation loop
        for iteration in range(self.max_iterations):
            # perform random move
            self.move(state)
            # calculate error for new state
            new_error = self.guiding_function(state)

            # Keep best state seen
            if new_error < best_error:
                best_error = new_error
                best_representation = copy(state.representation)
                best_iteration = iteration

            # Determine whether to accept the new state
            if self.accept(new_error, error):
                error = new_error
            else:
                self.undo_move(state)

            # Clear the move history to save memory and prevent accidentally
            # undoing accepted moves later
            state.clear_history()

            # Stop on user defined condition
            if self.stopping_criterion(state, best_error):
                self.reached_stopping_criterion = True
                break

        return OptimiserResult(
            representation=best_representation,
            error=best_error,
            best_iteration=best_iteration,
            iteration=iteration,
            restarts=None)

    def accept(self, new_error, current_error):
        oldest_error = self.costs.popleft()
        if (self.is_as_good(new_error, current_error) or
           self.is_better(new_error, oldest_error)):
            self.costs.append(new_error)
            return True
        else:
            self.costs.append(current_error)
            return False


class SA(RestartLocalSearch):

    def accept(self, old_error, new_error, temperature):
        delta = abs(new_error - old_error)
        if self.is_better(new_error, old_error):
            delta *= -1
        # DETERMINISTIC
        # return delta < -temperature or 0 <= delta < temperature
        # STOCHASTIC
        # Accept when dE is non-negative or with probability P = e^(-dE/T)
        return (delta <= 0) or (temperature > 0 and
                                random() < exp(-delta/temperature))

    def initialise(self, parameters):
        super().initialise(parameters)
        try:
            self.num_temps = parameters['num_temps']
            self.steps_per_temp = parameters['steps_per_temp']
            self.init_temp = parameters['init_temp']
            self.temp_rate = parameters['temp_rate']
        except KeyError:
            print('Optimiser parameters missing!', file=sys.stderr)
            raise
        self.temperatures = stepped_exp_decrease(
            self.init_temp, self.temp_rate,
            self.num_temps, self.steps_per_temp)

    def _optimise(self, state):
        # Calculate initial error
        error = self.guiding_function(state)

        # Setup aspiration criteria
        best_error = error
        best_representation = copy(state.representation)
        best_iteration = 0

        best_error_for_temp = error

        last_temp = self.init_temp  # so we can check for temp changes

        self.reached_stopping_criterion = False

        # annealing loop
        for iteration, temp in enumerate(self.temperatures):
            # Stop on user defined condition
            if self.stopping_criterion(state, best_error):
                self.reached_stopping_criterion = True
                break

            # Log error on new temperature if logging
            best_error_for_temp = min(best_error_for_temp, error)
            if last_temp != temp:
                logging.debug('temp: %f best_err:%f ',
                              last_temp, best_error_for_temp)
                best_error_for_temp = error
                last_temp = temp

            # perform random move
            self.move(state)

            # calculate error for new state
            new_error = self.guiding_function(state)

            # Keep best state seen
            if new_error < best_error:
                best_representation = copy(state.representation)
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

        return OptimiserResult(
            representation=best_representation,
            error=best_error,
            best_iteration=best_iteration,
            iteration=iteration,
            restarts=None)
