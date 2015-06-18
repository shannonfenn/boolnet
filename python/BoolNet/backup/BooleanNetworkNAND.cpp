#include "BooleanNetworkNAND.h"
#include <algorithm>
#include <functional>   // std::plus
#include <string>
#include <cmath>

// FOR GRAPHML
#include "boost/graph/directed_graph.hpp"
#include "boost/graph/graphml.hpp"

BooleanNetworkNAND::BooleanNetworkNAND(vector<vector<size_t> > initial_gates, 
                                       vector<vector<char> > inputs, 
                                       vector<vector<char> > target) :
    evaluated(false),
    first_changed_gate(0),
    gates(initial_gates)
{
    if(initial_gates.empty()) {
        throw invalid_argument("BooleanNetworkNAND - empty initial gate list.");
    }
    setExamples(inputs, target);
}

BooleanNetworkNAND::~BooleanNetworkNAND() {
}

void BooleanNetworkNAND::setExamples(vector<vector<char>> inputs, 
                                     vector<vector<char>> target) {
    // Check inputs
    if(inputs.empty() || inputs[0].empty()) 
        throw invalid_argument("BooleanNetworkNAND - empty input matrix.");
    if(target.empty() || target[0].empty())
        throw invalid_argument("BooleanNetworkNAND - empty target matrix.");

    Ni = inputs[0].size();
    No = target[0].size();

    if(pow(2, Ni) < inputs.size() )
        throw invalid_argument("BooleanNetworkNAND - Can't represent all examples (" + to_string(inputs.size()) + 
            ") with the number of inputs (" + to_string(Ni) + ").");

    // check target
    if(No > gates.size())
        throw invalid_argument("BooleanNetworkNAND - More bits in target than gates.");

    if(inputs.size() != target.size()) 
        throw invalid_argument("BooleanNetworkNAND::setExamples - number of input examples (" + 
            to_string(inputs.size()) + ") does not match number of output examples (" + 
            to_string(target.size()) + ").");

    // Initially copy over 
    state_matrix = move(inputs);
    target_matrix = move(target);
    error_matrix = target_matrix;   // no need to init, but error must match target in size
        
    for(auto& state : state_matrix) {
        state.resize(Ni + gates.size());
    }
    
    // state vectors have been reset, so start point must change as well
    evaluated = false;
    clearHistory();
    first_changed_gate = 0;
}

size_t BooleanNetworkNAND::getNi() const {
    return Ni;
}

size_t BooleanNetworkNAND::getNo() const {
    return No;
}

size_t BooleanNetworkNAND::getNg() const {
    return gates.size();
}


vector<vector<size_t>> BooleanNetworkNAND::getGates() const { 
    return gates; 
}

// void BooleanNetworkNAND::addGates(size_t N) {
//     evaluated = false;
//     clearHistory();

//     size_t Ng = gates.size();

//     // If the first changed gate is in the region to be moved then move the index to 
//     // the first gate of this region of new gates this ensures new gates are evaluated.
//     // Even though currently they are not connected to the output they may eventually 
//     // be and if they are not evaluated now they may not be in the future. (Not sure about this?)
//     if(first_changed_gate >= Ng - No) {
//         first_changed_gate = Ng - No;
//     }
    
//     // resize gate array
//     gates.resize(Ng + N);

//     // copy output gates to end
//     // must be done in reverse order in case N < No - in this case 
//     // the output gate region overlaps the new gate region
//     for(size_t o=No; o!=0; o--) {
//         // Move the gate to it's new position
//         gates[Ng - No + N + o - 1] = gates[Ng - No + o - 1];
        
//         // If the gate takes any gates to be moved as it's input
//         // then shift that index by the number of inserted gates
//         if(gates[Ng - No + N + o - 1][0] >= Ni + Ng - No) {
//             gates[Ng - No + N + o - 1][0] += N;
//         }
//         if(gates[Ng - No + N + o - 1][1] >= Ni + Ng - No) {
//             gates[Ng - No + N + o - 1][1] += N;
//         }
//     }
    
//     // randomize inserted gates
//     // Not done now - gates will be equal to the original gates

//     // expand the state vectors by the number of included gates
//     // no need to shift state values since the states will be re-evaluated
//     // from before the new gate region anyway
//     for(auto& state : state_matrix) {
//         state.resize(state.size() + N);
//     }
// }

void BooleanNetworkNAND::evaluate() {
    // double and triple check this is valid

    if(not evaluated) {

        size_t state_start = first_changed_gate;
        size_t output_start = 0;
        
        // If the gate modified is one of the output gates we don't want to be 
        // re-evaluating the error matrix for outputs before that one
        if(state_start > gates.size() - No) {
            output_start = state_start + No - gates.size();
        }
                
        auto state_it = state_matrix.begin();
        auto error_it = error_matrix.begin();
        auto target_it = target_matrix.begin();

        for(; state_it != state_matrix.end(); state_it++, error_it++, target_it++) {
            // Re-evaluate state
            auto g = gates.begin() + state_start;
            auto s_it = state_it->begin() + Ni + state_start;
            
            for(; g!= gates.end(); g++, s_it++) {
                *(s_it) = not ( (*state_it)[ (*g)[0] ] and (*state_it)[ (*g)[1] ] );
            }
            
            // Re-evaluate error matrix
            auto er_it = error_it->begin() + output_start;
            auto st_it = state_it->begin() + Ni + gates.size() - No + output_start;
            auto tg_it = target_it->begin() + output_start;
            for (;
                 er_it < error_it->end();
                 er_it++,
                 st_it++,
                 tg_it++) {
                (*er_it) = ( (*st_it) != (*tg_it) );
            }
        }
        
        evaluated = true;
    }
}

double BooleanNetworkNAND::accuracy() {
    evaluate();

    double numCorrect = 0.0;

    for(const auto& row : error_matrix) {
        if( all_of(row.begin(), row.end(), [](size_t i) { return i==0; }) ) {
            numCorrect += 1.0;
        }
    }

    return numCorrect / error_matrix.size();
}

double BooleanNetworkNAND::error(Metric metric) {
    evaluate();
       
    return BitError::metricValue(error_matrix, metric);
}

vector<double> BooleanNetworkNAND::errorPerBit() {
    evaluate();
    vector<double> bitErrs(No, 0.0);
    for(const auto& row : error_matrix) {
        transform(bitErrs.begin(), bitErrs.end(), row.begin(), bitErrs.begin(), plus<double>());
    }
    size_t Ne = error_matrix.size();
    transform(bitErrs.begin(), bitErrs.end(), bitErrs.begin(), [Ne](double d) { return d / Ne; });
    return bitErrs;
}

vector<vector<char> > BooleanNetworkNAND::errorMatrix() {
     evaluate();
                     
     return error_matrix;                      
}

vector<size_t> BooleanNetworkNAND::maxDepthPerBit() const {
    vector<size_t> depths(gates.size() + Ni, 0);
    size_t d1, d2;
   
    for (size_t g = 0; g < gates.size(); g++) {
        d1 = depths[gates[g][0]];
        d2 = depths[gates[g][1]];
        
        depths[g + Ni] = max(d1, d2) + 1;
    }
    
    return vector<size_t>(depths.end() - No, depths.end());
}

void BooleanNetworkNAND::moveToNeighbour(const Move& move) {
    if(evaluated)
        first_changed_gate = move.gate;
    else
        first_changed_gate = min(first_changed_gate, move.gate);

    // Record the inverse to this move
    Move inverse;
    inverse.gate = move.gate;
    inverse.connection = gates.at(move.gate).at(move.inp);
    inverse.inp = move.inp;
    inverse_moves.push_back(inverse);

    // actually modify the connection
    gates[move.gate][move.inp] = move.connection;

    // indicate that the network must be reevaluated
    evaluated = false;
}

void BooleanNetworkNAND::revertMove() {
    if(not inverse_moves.empty()) {
        auto inverse = inverse_moves.back();

        // indicate re-evaluation is needed
        evaluated = false;

        // change connection
        gates[inverse.gate][inverse.inp] = inverse.connection;

        inverse_moves.pop_back();
    }
    else {
        throw logic_error("BooleanNetworkNAND - tried to revert with empty move stack.");
    }
}

void BooleanNetworkNAND::revertAllMoves() {
    while(not inverse_moves.empty()) {
        auto inverse = inverse_moves.back();

        // indicate re-evaluation is needed
        evaluated = false;

        // change connection
        gates[inverse.gate][inverse.inp] = inverse.connection;

        inverse_moves.pop_back();
    }
}

void BooleanNetworkNAND::clearHistory() {
    inverse_moves.clear();
}

void BooleanNetworkNAND::forceReevaluation() { 
    evaluated = false; 
}

vector<vector<char> > BooleanNetworkNAND::getFullStateTableForSamples() {
    evaluate();

    return state_matrix;
}

vector<vector<char> > BooleanNetworkNAND::getTruthTable() const {
    vector<char> state(Ni + gates.size());
    size_t num_inputs = pow(2, Ni);
    vector<vector<char>> table;
    table.reserve(num_inputs);

    for(size_t i=0; i<num_inputs; i++) {
        // Generate next input
        for (size_t b = 0; b < Ni; b++) {
            state[b] = ( (1 << b & i) != 0 );
        }
        // Evaluate network state vector
        for(size_t g = 0; g < gates.size(); g++) {
            state[Ni + g] = not ( state[gates[g][0]] and state[gates[g][1]] );
        }

        // Copy output gate states to table
        table.push_back(vector<char>(state.end() - No, state.end()));
    }

    return table;
}
