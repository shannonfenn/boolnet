/* 
 * File:   Annealer.cpp
 * Author: Shannon
 * 
 * Created on 4 November 2013, 1:55 PM
 */

#include <math.h>
#include "Annealer.h"

using namespace std;

Annealer::Annealer()
{
}

Annealer::Annealer(map<string, size_t> posIntParams, map<string, double> realParams)
{
    setParameters(posIntParams, realParams);
}

Annealer::~Annealer() {
}


void Annealer::setParameters(map<string, size_t> posIntParams, 
                             map<string, double> realParams) {
    // Unpack arguments
    // If any of the following do not exist the map container with throw std::out_of_range
    numTemps = posIntParams.at("numTemps");
    stepsPerTemp = posIntParams.at("stepsPerTemp");
        
    finalNg = posIntParams.at("finalNg");
    
    initTemp = realParams.at("initTemp");
    tempRate = realParams.at("tempRate");
}

//Annealer::Result Annealer::run(size_t numTemps, size_t stepsPerTemp, double initTemp, double tempRate, BooleanNetworkNAND seed,
//                                 ostream& errorLog, ostream& progressLog, Metric guidingMetric) {
Optimiser::Result Annealer::run(BooleanNetworkNAND seed, Metric guidingMetric, bool logErrors) {
    BooleanNetworkNAND state(seed);
    BooleanNetworkNAND bestState(seed);
    //vector<InputType > outputs;
    double error;
    double bestError;
    double bestErrorForTemp;
    double newError;
    bool accepted;
    double temperature = initTemp;
    size_t iteration;
    size_t bestIteration = 0;
    size_t totalIterations = numTemps * stepsPerTemp;
    size_t tempStep = 0;
    
    bool gateAddingOn = finalNg != state.getNg();
    size_t gateAddIterations;
    size_t iterationsUntilGateAddition;
    
    vector<bool> NOTPRINTED;    //REMOVE LATER
    for(int i=0; i<state.getNo(); i++) {
        NOTPRINTED.push_back(true);
    }
    
    // Calculate initial error
    bestError = error = state.error(guidingMetric);
    
    if(logErrors) {
        bestErrorForTemp = error;
        cout << "\"BestErrorPerTemp\": [";
    }
    
    if(gateAddingOn) {
        gateAddIterations = totalIterations / (finalNg - state.getNg());
        iterationsUntilGateAddition = gateAddIterations;
    }
    
    for(iteration = 0; iteration < totalIterations && error > 0; iteration++, tempStep++) {

        if(gateAddingOn) {
            if(iterationsUntilGateAddition > 0) {
                iterationsUntilGateAddition--;
            }
            else {
                iterationsUntilGateAddition = gateAddIterations;
                state.addGates(1);
                if(state.error(guidingMetric) != error) {
                    cerr << "Adding gates changes error!" << state.error(guidingMetric) << " " << error << endl;
                }
            }
        }
        
        // Reduce temperature if required
        if(tempStep >= stepsPerTemp) {
            tempStep = 0;
            temperature *= tempRate;
            
            // Pertubation code
//            state.pertubExamples(generator);
//            error = state.error(guidingMetric);
                        
            // Tracking of error
            if(logErrors) {
                cout << bestErrorForTemp << ", " << std::flush;
                bestErrorForTemp = error;
            }
        }

//        for(int i=0; i < state.getNi() * double(totalIterations - iteration) / totalIterations; i++) //fixed
//            state.moveToNeighbour(state.getRandomMove(generator));
        state.moveToNeighbour(state.getRandomMove(generator));

        newError = state.error(guidingMetric);

        // REMOVE LATER - THIS SIMPLY PRINTS OUT THE TABLE OF ALL GATE OUTPUTS ONCE THE OPTIMISER
        //                HITS 0 ERROR ON EACH BIT
        for(int i=0; i<state.getNo(); i++) {
            auto perBitErrs = state.errorPerBit();
            if(NOTPRINTED[i] and perBitErrs[i] == 0.0) {
                NOTPRINTED[i] = false;
                cerr << endl << "bit " << i << endl;
                for(auto& stateVector: state.getFullStateTableForSamples()) {
                    cerr << stateVector << endl;
                }
                cerr << state.getGates() << endl;
            }
        }
        // REMOVE UP TO HERE

        // Keep best state seen
        if(newError < bestError) {
            bestState = state;
            bestError = newError;
            bestIteration = iteration;
        }
        
        // Determine whether to accept the new state
        accepted = accept(error, newError, temperature);
     
        if(accepted) {
            error = newError;
        }
        else {
//            state.revertMove();
            state.revertAllMoves();
        }
        state.clearHistory();
        
        // Tracking of error
        if(logErrors and error < bestErrorForTemp) {
            bestErrorForTemp = newError;
        }
    }
    
    if(logErrors)
        cout << "]" << std::flush;

    return {bestState, bestIteration, iteration};
}


//
//vector<InputType > Annealer::evaluateOnExamples(const StateType& net, const vector< pair<InputType, InputType > >& examples) {

//    vector<InputType > outputs;
//    outputs.reserve(examples.size());
//    for(const auto& example : examples) {
//        outputs.push_back(net.evaluate(example.first));
//    }
//    return outputs;
//}

//
//double Annealer::calculateTrainingError(StateType net, const vector< pair<InputType, InputType > >& examples, Metric metric) {
//    return calculateTrainingError(evaluateOnExamples(net, examples), examples, metric);
//}

//
//double Annealer::calculateTrainingError(const vector<InputType >& outputs, const vector< pair<InputType, InputType > >& examples, Metric metric) {
//    double totalError = 0;

//    if( outputs.size() != examples.size() ) {
//        cerr << "calculateTrainingError - output and example lengths don't match" << endl;
//        cerr << "output length: " << outputs.size() << " example length: " << examples.size() << endl;
//        throw new exception;
//    }

//    for(int i=0; i<examples.size(); i++) {
//        totalError += singleError(outputs[i], examples[i].second, metric);
//    }

//    return totalError / examples.size();
//}


//double Annealer::calculateTrainingError(const vector< pair<InputType, InputType > >& examples, Metric metric) {
//    double totalError = 0;

//    for(const auto& example : examples) {
//        totalError += singleError(example.first, example.second, metric);
//    }

//    return totalError / examples.size();
//}

//
//double Annealer::calculateFullError(StateType net, Metric metric) {
//    double totalError = 0;

//    for(size_t i = 0; i < goal.size(); i++) {
//        auto inp = toBinary(i, Ni);

//        auto out = net.evaluate(inp);

//        totalError += singleError(out, goal[i], metric);
//    }
    
//    return totalError / goal.size();
//}


//double Annealer::calculateFullError(StateType net, Metric metric) {
//    double totalError = 0;

//    for(size_t i = 0; i < goal.size(); i++) {
//        auto state = toBinary(i, Ni);
//        state.resize(Ni + net.getNg());

//        net.evaluate(state);

//        totalError += singleError(state, goal[i], metric);
//    }

//    return totalError / goal.size();
//}


//double Annealer::singleError(const InputType& state, const InputType& expected, Metric metric)
//{
//    if(state.size() < expected.size())
//        throw AnnealerException(" - singleError - state must be at least as long as expected output.");
//    double error = 0.0;

//    // for hierarchical methods
//    size_t n = expected.size();
//    auto outStart = state.begin() + state.size() - n;

//    vector<int> e(n);
//    vector<int> a(n);

//    switch(metric) {
//    case Metric::SIMPLE:
//        for(size_t k=0; k < n; k++) {
//            if(outStart[k] != expected[k]) {
//                error += 1.0;
//            }
//        }
//        break;

//    case Metric::WEIGHTED:
//        for(size_t k=0; k < n; k++) {
//            if(outStart[k] != expected[k]) {
//                error += k + 1.0;
//            }
//        }
//        break;

//    case Metric::INVERSE_WEIGHTED:
//        for(size_t k=0; k < n; k++) {
//            if(outStart[k] != expected[k]) {
//                error += n - k;
//            }
//        }
//        break;

//    case Metric::HIERARCHICAL:
//        for(size_t k=0; k < n; k++) {
//            e[k] = ( outStart[k] != expected[k] ? 1 : 0 );
//        }

//        a[0] = e[0];
//        for(size_t k=1; k < e.size(); k++) {
//            a[k] = a[k-1] * (1 - e[k]) + e[k];
//        }

//        for(int v : a) {
//            error += v;
//        }

//        break;
//    case Metric::INVERSE_HIERARCHICAL:
//        for(size_t k=0; k < n; k++) {
//            e[k] = ( outStart[n-k-1] != expected[n-k-1] ? 1 : 0 );
//        }

//        a[0] = e[0];
//        for(size_t k=1; k < e.size(); k++) {
//            a[k] = a[k-1] * (1 - e[k]) + e[k];
//        }

//        for(int v : a) {
//            error += v;
//        }

//        break;
//    }

//    return error;
//}

//double Annealer::singleError(const InputType& bits1, const InputType& bits2, Metric metric)
//{
//    if(bits1.size() != bits2.size())
//        throw AnnealerException(" - singleError - bitstring lengths must be equal.");
//    double error = 0.0;

//    // for hierarchical methods
//    size_t n = bits1.size();
//    vector<int> e(n);
//    vector<int> a(n);

//    switch(metric) {
//    case Metric::SIMPLE:
//        for(size_t k=0; k < n; k++) {
//            if(bits1[k] != bits2[k]) {
//                error += 1.0;
//            }
//        }
//        break;

//    case Metric::WEIGHTED:
//        for(size_t k=0; k < n; k++) {
//            if(bits1[k] != bits2[k]) {
//                error += k + 1.0;
//            }
//        }
//        break;

//    case Metric::INVERSE_WEIGHTED:
//        for(size_t k=0; k < n; k++) {
//            if(bits1[k] != bits2[k]) {
//                error += n - k;
//            }
//        }
//        break;

//    case Metric::HIERARCHICAL:
//        for(size_t k=0; k < n; k++) {
//            e[k] = ( bits1[k] != bits2[k] ? 1 : 0 );
//        }

//        a[0] = e[0];
//        for(size_t k=1; k < e.size(); k++) {
//            a[k] = a[k-1] * (1 - e[k]) + e[k];
//        }

//        for(int v : a) {
//            error += v;
//        }

//        break;
//    case Metric::INVERSE_HIERARCHICAL:
//        for(size_t k=0; k < n; k++) {
//            e[k] = ( bits1[n-k-1] != bits2[n-k-1] ? 1 : 0 );
//        }

//        a[0] = e[0];
//        for(size_t k=1; k < e.size(); k++) {
//            a[k] = a[k-1] * (1 - e[k]) + e[k];
//        }

//        for(int v : a) {
//            error += v;
//        }

//        break;
//    }

//    return error;
//}

bool Annealer::accept(double oldError, double newError, double temperature) {
    double delta = newError - oldError;
    
    
    // DETERMINISTIC
    
//    if(delta > temperature) {
//        return false;
//    }
//    else if (delta > 0) {
//        return true;
//    }
//    else if (delta > -temperature) {
//        return false;
//    }
//    else {
//        return true;
//    }
    
    
    // STOCHASTIC

    if(delta <= 0) {
        // Accept all movements for which dE is non-negative
        return true;
    }
    else if(temperature > 0){
        // Probability of acceptance based on standard formula P = e^(-dE/T)
        double P = exp( - delta / temperature);
        // Random uniformly distributed decimal in [0, 1)
        double r = uniform_real_distribution<>(0, 1)(generator);
        // Accept with probability P 
        return r < P;
    }
    else {
        // Reject if dE > 0 and T = 0
        return false;
    }
}


