Contains an optimised implementation of FBNs in Cython for learning applications. Experiments are specified in yaml files and can be given as a list of configurations or a cartesian product thereof (be careful of combinatorial explosion in the number of experiment instances).

Includes three base optimisers, which all allow restarts:
 - Hill Climbing
 - Late-Acceptance Hill Climbing
 - Simulated Annealing

Several multi-target methods can also be selected from including:
 - Individual Classifiers (aka Binary Relevance)
 - Classifier Chains (with and without target curricula)
 - Adaptive Learning Via Iterated Selection and Scheduling
 - Ensemble of Classifier Chains
Also includes wrappers for various a priori curriculum generation methods.

This currently includes scripts for configuring and running experiments on a PBS cluster, however these will be excised into another repository in the future.
