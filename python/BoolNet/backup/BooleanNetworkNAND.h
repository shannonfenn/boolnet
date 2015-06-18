/*
 * File:   BooleanNetworkNAND.h
 * Author: Shannon Fenn (shannon.fenn@gmail.com)
 *
 * Purpose: Represents a boolean network constructed only from 2 input NAND gates
 *
 * Created on 16 December 2013
 */

#ifndef BOOLEANNETWORKNAND_H
#define BOOLEANNETWORKNAND_H

#include <string>
#include <random>
#include <iostream>
#include <vector>
#include <deque>
#include "BitError.hpp"
#include "VectorTools.h"

using namespace std;
using namespace BitError;

struct Move {
    size_t gate;        // [0, Ng) 
    size_t connection;  // [0, Ng + Ni)  gate 5 for a 4 inp network would be at 8
    size_t inp;         // [0, 2)
};

class BooleanNetworkNAND
{
public:
//    BooleanNetworkNAND(const BooleanNetworkNAND& orig);
    BooleanNetworkNAND(vector<vector<size_t>> initial_gates, 
                       vector<vector<char>> inputs, 
                       vector<vector<char>> target);
    virtual ~BooleanNetworkNAND();

    void setExamples(vector<vector<char>> inputs, vector<vector<char>> target);
    
    size_t getNi() const; 
    size_t getNo() const; 
    size_t getNg() const; 

    // void addGates(size_t N);
    vector<vector<size_t>> getGates() const;

//    vector<char> evaluate(vector<char> input) const;
//    void evaluate(vector<char>& stateVector) const;
//    void evaluateFromChangedGate(vector<char>& stateVector) const;
    
    double accuracy();
    double error(Metric metric);
    vector<double> errorPerBit();
    vector<vector<char>> errorMatrix();

    vector<vector<char>> getFullStateTableForSamples();

    vector<size_t> maxDepthPerBit() const;
    
    // void pertubExamples(default_random_engine& generator);
    
    vector<vector<char>> getTruthTable() const;

    void moveToNeighbour(const Move& move);
    void revertMove();
    void revertAllMoves();
    void clearHistory();

    void forceReevaluation();

    // void writeGML(ostream &out) const;
    // void writeGraphML(ostream& out) const;
    
//    BooleanNetworkNAND operator = (BooleanNetworkNAND other);
    
    // friend bool operator == (BooleanNetworkNAND lhs, BooleanNetworkNAND rhs);
    // friend bool operator != (BooleanNetworkNAND lhs, BooleanNetworkNAND rhs);
    // friend ostream& operator << (ostream& out, BooleanNetworkNAND net);
protected:
    BooleanNetworkNAND();
    void evaluate();
        
public:
    size_t Ni;
    size_t No;
    
    bool evaluated;
    size_t first_changed_gate;

    deque<Move> inverse_moves;

    vector<vector<size_t>> gates;

//    list<pair<vector<char>, vector<char> > > stateExamplePairs;
    vector<vector<char>> state_matrix;
    vector<vector<char>> target_matrix;
    vector<vector<char>> error_matrix;
};

#endif // BOOLEANNETWORKNAND_H
