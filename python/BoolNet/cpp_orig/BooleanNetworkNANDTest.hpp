#ifndef BOOLEANNETWORKNANDTEST_HPP
#define BOOLEANNETWORKNANDTEST_HPP

#include "catch.hpp"
#include "BooleanNet/BooleanNetworkNAND.h"
#include <random>
#include "common.hpp"

// ADD TEST FOR TRUTH TABLE

using namespace std;


TEST_CASE("getRandomNeighbourSimple/Output after change - single gate guaranteed to be different") {
    vector<pair<size_t, size_t>> gates;
    vector<vector<char>> inputs;
    vector<vector<char>> outputs;
    
    gates.push_back(make_pair(0, 1));
    
    for(int k=0; k<16; k++) {
        inputs.push_back(toBinary(k, 4));
        outputs.push_back(toBinary(0, 1));    // doesn't matter what this is, just that the error matrices are different
    }
    
    BooleanNetworkNAND net1(4, 1, gates, inputs, outputs);

    random_device rd;  // Seed with a real random value, if available
    default_random_engine generator(rd());

    for(int k=0; k < 100; k++) {
        for(int i=0; i<100; i++) {
            BooleanNetworkNAND net2 = net1;
            net1.moveToNeighbour(net1.getRandomMove(generator));
            REQUIRE(net1.errorMatrix() != net2.errorMatrix());
        }
    }
}

TEST_CASE("getRandomNeighbourSimple/Output after change - single layer guaranteed to be different") {
    vector<pair<size_t, size_t> > gates;
    vector<vector<char>> inputs;
    vector<vector<char>> outputs;
    
    gates.push_back(make_pair(0, 1));
    gates.push_back(make_pair(2, 3));
    
    for(int k=0; k<16; k++) {
        inputs.push_back(toBinary(k, 4));
        outputs.push_back(toBinary(0, 2));    // doesn't matter what this is, just that the error matrices are different
    }
    
    BooleanNetworkNAND net1(4, 2, gates, inputs, outputs);

    random_device rd;  // Seed with a real random value, if available
    default_random_engine generator(rd());

    for(int k=0; k < 100; k++) {
        for(int i=0; i<100; i++) {
            BooleanNetworkNAND net2 = net1;
            net1.moveToNeighbour(net1.getRandomMove(generator));
            REQUIRE(net1.errorMatrix() != net2.errorMatrix());
        }
    }
}


TEST_CASE("move") {
    vector<pair<size_t, size_t> > gates;
    vector<vector<char>> inputs;
    vector<vector<char>> outputs;

    gates.push_back(make_pair(0, 1));
    gates.push_back(make_pair(2, 3));

    for(int k=0; k<16; k++) {
        inputs.push_back(toBinary(k, 4));
        outputs.push_back(toBinary(0, 2));    // doesn't matter what this is, just that the error matrices are different
    }

    BooleanNetworkNAND net(4, 2, gates, inputs, outputs);

    BooleanNetworkNAND::Move move1 = {0, 1, false};
    vector<vector<char> > expected1 = {{1, 1},
                                       {1, 1},
                                       {0, 1},
                                       {0, 1},
                                       {1, 1},
                                       {1, 1},
                                       {0, 1},
                                       {0, 1},
                                       {1, 1},
                                       {1, 1},
                                       {0, 1},
                                       {0, 1},
                                       {1, 0},
                                       {1, 0},
                                       {0, 0},
                                       {0, 0}};
    net.moveToNeighbour(move1);
    REQUIRE(net.errorMatrix() == expected1);

    BooleanNetworkNAND::Move move2 = {1, 0, true};
    vector<vector<char> > expected2 = {{1, 1},
                                       {1, 1},
                                       {0, 1},
                                       {0, 1},
                                       {1, 1},
                                       {1, 0},
                                       {0, 1},
                                       {0, 0},
                                       {1, 1},
                                       {1, 1},
                                       {0, 1},
                                       {0, 1},
                                       {1, 1},
                                       {1, 0},
                                       {0, 1},
                                       {0, 0}};
    net.moveToNeighbour(move2);
    REQUIRE(net.errorMatrix() == expected2);

    BooleanNetworkNAND::Move move3 = {0, 0, true};
    vector<vector<char> > expected3 = {{1, 1},
                                       {1, 1},
                                       {1, 1},
                                       {0, 1},
                                       {1, 1},
                                       {1, 0},
                                       {1, 1},
                                       {0, 0},
                                       {1, 1},
                                       {1, 1},
                                       {1, 1},
                                       {0, 1},
                                       {1, 1},
                                       {1, 0},
                                       {1, 1},
                                       {0, 0}};
    net.moveToNeighbour(move3);
    REQUIRE(net.errorMatrix() == expected3);

    BooleanNetworkNAND::Move move4 = {1, 3, true};
    vector<vector<char> > expected4 = {{1, 1},
                                       {1, 1},
                                       {1, 1},
                                       {0, 1},
                                       {1, 1},
                                       {1, 1},
                                       {1, 1},
                                       {0, 1},
                                       {1, 1},
                                       {1, 1},
                                       {1, 1},
                                       {0, 1},
                                       {1, 0},
                                       {1, 0},
                                       {1, 0},
                                       {0, 0}};
    net.moveToNeighbour(move4);
    REQUIRE(net.errorMatrix() == expected4);

    BooleanNetworkNAND::Move move5 = {1, 4, true};
    vector<vector<char> > expected5 = {{1, 1},
                                       {1, 1},
                                       {1, 1},
                                       {0, 1},
                                       {1, 0},
                                       {1, 0},
                                       {1, 0},
                                       {0, 1},
                                       {1, 1},
                                       {1, 1},
                                       {1, 1},
                                       {0, 1},
                                       {1, 0},
                                       {1, 0},
                                       {1, 0},
                                       {0, 1}};
    net.moveToNeighbour(move5);
    REQUIRE(net.errorMatrix() == expected5);


    BooleanNetworkNAND::Move move6 = {1, 4, false};
    vector<vector<char> > expected6 = {{1, 0},
                                       {1, 0},
                                       {1, 0},
                                       {0, 1},
                                       {1, 0},
                                       {1, 0},
                                       {1, 0},
                                       {0, 1},
                                       {1, 0},
                                       {1, 0},
                                       {1, 0},
                                       {0, 1},
                                       {1, 0},
                                       {1, 0},
                                       {1, 0},
                                       {0, 1}};
    net.moveToNeighbour(move6);
    REQUIRE(net.errorMatrix() == expected6);

}

TEST_CASE("move without evaluation") {
    vector<pair<size_t, size_t> > gates;
    vector<vector<char>> inputs;
    vector<vector<char>> outputs;

    gates.push_back(make_pair(0, 1));
    gates.push_back(make_pair(2, 3));

    for(int k=0; k<16; k++) {
        inputs.push_back(toBinary(k, 4));
        outputs.push_back(toBinary(0, 2));    // doesn't matter what this is, just that the error matrices are different
    }

    BooleanNetworkNAND net(4, 2, gates, inputs, outputs);
    vector<vector<char> > expected = {{1, 1},
                                      {1, 1},
                                      {1, 1},
                                      {0, 1},
                                      {1, 1},
                                      {1, 1},
                                      {1, 1},
                                      {0, 1},
                                      {1, 1},
                                      {1, 1},
                                      {1, 1},
                                      {0, 1},
                                      {1, 0},
                                      {1, 0},
                                      {1, 0},
                                      {0, 0}};

    BooleanNetworkNAND::Move move1 = {1, 4, true};
    vector<vector<char> > expected1 = {{1, 1},
                                       {1, 1},
                                       {1, 1},
                                       {0, 1},
                                       {1, 0},
                                       {1, 0},
                                       {1, 0},
                                       {0, 1},
                                       {1, 1},
                                       {1, 1},
                                       {1, 1},
                                       {0, 1},
                                       {1, 0},
                                       {1, 0},
                                       {1, 0},
                                       {0, 1}};
    net.moveToNeighbour(move1);
    REQUIRE(net.errorMatrix() == expected1);
}

TEST_CASE("getRandomNeighbourSimple/100 steps") {
    vector<pair<size_t, size_t> > gates;
    vector<vector<char>> inputs;
    vector<vector<char>> outputs;
    
    gates.push_back(make_pair(0, 1));
    gates.push_back(make_pair(2, 3));
    gates.push_back(make_pair(4, 5));
    
    for(int k=0; k<16; k++) {
        inputs.push_back(toBinary(k, 4));
        outputs.push_back(toBinary(0, 2));    // doesn't matter what this is, we aren't testing that
    }

    random_device rd;  // Seed with a real random value, if available
    default_random_engine generator(rd());

    BooleanNetworkNAND net1(4, 2, gates, inputs, outputs);
    for(int i=0; i<100; i++) {
        BooleanNetworkNAND net2 = net1;
        net1.moveToNeighbour(net1.getRandomMove(generator));
        REQUIRE( net1 != net2 );
        net1 = net2;
    }
}

TEST_CASE("Evaluate/4-input AND") {
    // a ^ b ^ c ^ d   ==  !!( !!(a ^ b) ^ !!(c ^ d) ) 
    vector<pair<size_t, size_t> > gates;
    vector<vector<char>> inputs;
    vector<vector<char>> outputs;
    
    gates.push_back(make_pair(0, 1));
    gates.push_back(make_pair(2, 3));
    gates.push_back(make_pair(4, 4));
    gates.push_back(make_pair(5, 5));
    gates.push_back(make_pair(6, 7));
    gates.push_back(make_pair(8, 8));
        
    for(int k=0; k<16; k++) {
        inputs.push_back(toBinary(k, 4));
        outputs.push_back(toBinary(k == 15, 1));    // and
    }
    
    BooleanNetworkNAND net(4, 1, gates, inputs, outputs);
    
    REQUIRE( net.errorMatrix() == vector<vector<char>>(16, toBinary(0, 1)) );
    REQUIRE( net.error() == 0.0 );
}

TEST_CASE("Evaluate/4-input OR") {
    // a v b v c v d   ==  !( !!(!a ^ !b) ^ !!(!c ^ !d) ) 
    vector<pair<size_t, size_t> > gates;
    vector<vector<char>> inputs;
    vector<vector<char>> outputs;
    
    gates.push_back(make_pair(0, 0));
    gates.push_back(make_pair(1, 1));
    gates.push_back(make_pair(2, 2));
    gates.push_back(make_pair(3, 3));
    gates.push_back(make_pair(4, 5));
    gates.push_back(make_pair(6, 7));
    gates.push_back(make_pair(8, 8));
    gates.push_back(make_pair(9, 9));
    gates.push_back(make_pair(10, 11));
    
    for(int k=0; k<16; k++) {
        inputs.push_back(toBinary(k, 4));
        outputs.push_back(toBinary(k > 0, 1));    // or
    }
    
    BooleanNetworkNAND net(4, 1, gates, inputs, outputs);
    
    REQUIRE( net.errorMatrix() == vector<vector<char>>(16, toBinary(0, 1)) );
    REQUIRE( net.error() == 0.0 );
}

TEST_CASE("Evaluate/(1 ^ 2) v (3 ^ 4)") {
    // (1 ^ 2) v (3 ^ 4)    ==  !( !(a ^ b) ^ !(c ^ d) )
    vector<pair<size_t, size_t> > gates;
    vector<vector<char>> inputs;
    vector<vector<char>> outputs;
    
    gates.push_back(make_pair(0, 1));
    gates.push_back(make_pair(2, 3));
    gates.push_back(make_pair(4, 5));
    
    for(int k=0; k<16; k++) {
        inputs.push_back(toBinary(k, 4));
        outputs.push_back(toBinary(k == 3 || k == 7 || k > 10, 1));    // expression
    }

    BooleanNetworkNAND net(4, 1, gates, inputs, outputs);
        
    REQUIRE( net.errorMatrix() == vector<vector<char>>(16, toBinary(0, 1)) );
    REQUIRE( net.error() == 0.0 );
}

TEST_CASE("Evaluate/2 bit adder") {
    vector<pair<size_t, size_t> > gates;
    vector<vector<char>> inputs;
    vector<vector<char>> outputs;
    
    gates.push_back(make_pair(0, 2));   // 4
    gates.push_back(make_pair(1, 3));   // 5
    gates.push_back(make_pair(0, 4));   // 6
    gates.push_back(make_pair(2, 4));   // 7
    gates.push_back(make_pair(4, 4));   // 8
    gates.push_back(make_pair(1, 5));   // 9
    gates.push_back(make_pair(3, 5));   // 10
    gates.push_back(make_pair(9, 10));  // 11
    gates.push_back(make_pair(8, 11));  // 12
    gates.push_back(make_pair(8, 12));  // 13
    gates.push_back(make_pair(11, 12)); // 14
    gates.push_back(make_pair(6, 7));   // 15
    gates.push_back(make_pair(13, 14)); // 16
    gates.push_back(make_pair(5, 12));  // 17
    
    for(int k=0; k<16; k++) {
        inputs.push_back(toBinary(k, 4));
    }
    outputs.push_back(toBinary(0, 3));    // expression
    outputs.push_back(toBinary(1, 3));    // expression
    outputs.push_back(toBinary(2, 3));    // expression
    outputs.push_back(toBinary(3, 3));    // expression
    outputs.push_back(toBinary(1, 3));    // expression
    outputs.push_back(toBinary(2, 3));    // expression
    outputs.push_back(toBinary(3, 3));    // expression
    outputs.push_back(toBinary(4, 3));    // expression
    outputs.push_back(toBinary(2, 3));    // expression
    outputs.push_back(toBinary(3, 3));    // expression
    outputs.push_back(toBinary(4, 3));    // expression
    outputs.push_back(toBinary(5, 3));    // expression
    outputs.push_back(toBinary(3, 3));    // expression
    outputs.push_back(toBinary(4, 3));    // expression
    outputs.push_back(toBinary(5, 3));    // expression
    outputs.push_back(toBinary(6, 3));    // expression
    
    BooleanNetworkNAND net(4, 3, gates, inputs, outputs);
    
    REQUIRE( net.errorMatrix() == vector<vector<char>>(16, toBinary(0, 3)) );
    REQUIRE( net.error() == 0.0 );
}


#endif // BOOLEANNETWORKNANDTEST_HPP
