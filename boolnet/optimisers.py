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

    def __init__(self, guide_functor, minimise, stop_functor,
                 return_option='best', max_restarts=0):
        self.guiding_function = guide_functor
        if minimise:
            self.is_as_good = op.le
            self.is_better = op.lt
        else:
            self.is_as_good = op.ge
            self.is_better = op.gt
        self.stopping_condition = stop_functor
        self.return_option = return_option
        self.max_restarts = max_restarts

    def move(self, state):
        state.move_to_random_neighbour()

    def undo_move(self, state):
        state.revert_move()

    def _optimise(self, state):
        raise NotImplementedError()

    def run(self, state):
        for i in range(self.max_restarts + 1):
            step_result, reached_stop_condition = self._optimise(state)
            if reached_stop_condition:
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

    def __init__(self, max_iterations, **kwargs):
        super().__init__(**kwargs)
        self.max_iterations = max_iterations

    def _optimise(self, state):
        # Calculate initial error
        error = self.guiding_function(state)
        # error = evaluator.function_value(self.guiding_function)

        # set up aspiration criteria
        best_iteration = 0

        # optimisation loop
        for iteration in range(self.max_iterations):
            # Stop on user defined condition
            if self.stopping_condition(state, error):
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

        result = OptimiserResult(
            representation=state.representation,
            error=error,
            best_iteration=best_iteration,
            iteration=iteration,
            restarts=None)
        return result, self.stopping_condition(state, error)

    def accept(self, new_error, current_error):
        if self.is_as_good(new_error, current_error):
            return True


class LAHC(RestartLocalSearch):

    def __init__(self, cost_list_length, max_iterations, **kwargs):
        super().__init__(**kwargs)
        self.cost_list_length = cost_list_length
        self.max_iterations = max_iterations

    def _optimise(self, state):
        # Calculate initial error
        error = self.guiding_function(state)

        # set up aspiration criteria
        best_error = error
        best_representation = copy(state.representation)
        best_iteration = 0

        # initialise cost list
        self.costs = deque(repeat(error, self.cost_list_length))

        if self.stopping_condition(state, best_error):
            return OptimiserResult(
                representation=best_representation, error=best_error,
                best_iteration=0, iteration=0, restarts=None), True

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

            # Stop on user defined condition
            if self.stopping_condition(state, best_error):
                break

            # Determine whether to accept the new state
            if self.accept(new_error, error):
                error = new_error
            else:
                self.undo_move(state)

            # Clear the move history to save memory and prevent accidentally
            # undoing accepted moves later
            state.clear_history()

        if self.return_option == 'last':
            best_representation = copy(state.representation)

        result = OptimiserResult(
            representation=best_representation,
            error=best_error,
            best_iteration=best_iteration,
            iteration=iteration,
            restarts=None)
        return result, self.stopping_condition(state, best_error)

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
    def __init__(self, num_temps, steps_per_temp, init_temp, temp_rate,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_t = num_temps
        self.steps_per_t = steps_per_temp
        self.init_t = init_temp
        self.t_rate = temp_rate

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

    def _optimise(self, state):
        # Calculate initial error
        error = self.guiding_function(state)

        # Setup aspiration criteria
        best_error = error
        best_representation = copy(state.representation)
        best_iteration = 0

        best_error_for_temp = error

        last_temp = self.init_temp  # so we can check for temp changes

        # annealing loop
        temperatures = stepped_exp_decrease(self.init_t, self.t_rate,
                                            self.num_t, self.steps_per_t)
        for iteration, temp in enumerate(temperatures):

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

            # Stop on user defined condition
            if self.stopping_condition(state, best_error):
                break

            # Determine whether to accept the new state
            if self.accept(error, new_error, temp):
                error = new_error
            else:
                self.undo_move(state)

            # In any case, clear the move history to save memory
            # and prevent accidentally undoing accepted moves later
            state.clear_history()

        if self.return_option == 'last':
            best_representation = copy(state.representation)

        result = OptimiserResult(
            representation=best_representation,
            error=best_error,
            best_iteration=best_iteration,
            iteration=iteration,
            restarts=None)

        return result, self.stopping_condition(state, best_error)
