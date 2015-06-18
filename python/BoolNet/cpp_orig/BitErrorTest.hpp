#ifndef BITERRORTEST_HPP
#define BITERRORTEST_HPP

#include "catch.hpp"
#include "BooleanNet/BitError.hpp"
#include "common.hpp"

// ADD TEST FOR TRUTH TABLE

using namespace std;
using namespace BitError;

vector<vector<char> > getErrorMatrix6BitRand() {
                                            // 123456
                                            // 654321              worstlin
    vector<vector<char>> errorMatrix;       //         sim wlm wll hlm hll
    errorMatrix.push_back(toBinary(10, 6));   // 001010   2   6   8   4   5
    errorMatrix.push_back(toBinary(2, 6));    // 000010   1   2   5   2   5
    errorMatrix.push_back(toBinary(5, 6));    // 000101   2   4  10   3   6
    errorMatrix.push_back(toBinary(9, 6));    // 001001   2   5   9   4   6
    errorMatrix.push_back(toBinary(0, 6));    // 000000
    errorMatrix.push_back(toBinary(63, 6));   // 111111   6  21  21   6   6
    errorMatrix.push_back(toBinary(32, 6));   // 100000   1   6   1   6   1
    errorMatrix.push_back(toBinary(29, 6));   // 011101   4  13  15   5   6
    return errorMatrix;                     //         18  57  69  30  35
}

/*             worstlin
 *         sim wlm wll hlm hll
 * 0000     0   0   0   0   0
 * 0001     1   1   4   1   4
 * 0010     1   2   3   2   3
 * 0011     2   3   7   2   4
 * 0100     1   3   2   3   2
 * 0101     2   4   6   3   4
 * 0110     2   5   5   3   3
 * 0111     3   6   9   3   4
 * 1000     1   4   1   4   1
 * 1001     2   5   5   4   4
 * 1010     2   6   4   4   3
 * 1011     3   7   8   4   4
 * 1100     2   7   3   4   2
 * 1101     3   8   7   4   4
 * 1110     3   9   6   4   3
 * 1111     4   10  10  4   4
 *              80  80  49  49
 */

vector<vector<char> > getErrorMatrix4to1AND() {
    vector<vector<char>> errorMatrix;
    for(int k=0; k<16; k++) {
        errorMatrix.push_back(toBinary(k == 15, 1));
    }
    return errorMatrix;
}

vector<vector<char> > getErrorMatrix4to1OR() {
    vector<vector<char>> errorMatrix;
    for(int k=0; k<16; k++) {
        errorMatrix.push_back(toBinary(k > 0, 1));    // or
    }
    return errorMatrix;
}

vector<vector<char> > getErrorMatrix4to1Rand() {
    vector<vector<char>> errorMatrix;
    for(int k=0; k<16; k++) {
        errorMatrix.push_back(toBinary(k == 3 || k == 7 || k > 10, 1));
    }
    return errorMatrix;
}

vector<vector<char> > getErrorMatrix4BitCount() {
    vector<vector<char>> errorMatrix;
    for(int k=0; k<16; k++) {
        errorMatrix.push_back(toBinary(k, 4));
    }
    return errorMatrix;
}

TEST_CASE("SIMPLE") {
    auto metric = Metric::SIMPLE;
    REQUIRE( metricValue(getErrorMatrix6BitRand(), metric) == 18.0 / 8.0);
    REQUIRE( metricValue(getErrorMatrix4to1AND(), metric) == 1.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4to1OR(), metric) == 15.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4to1Rand(), metric) == 7.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4BitCount(), metric) == 32.0 / 16.0);
}


TEST_CASE("WEIGHTED_LIN_MSB") {
    auto metric = Metric::WEIGHTED_LIN_MSB;
    REQUIRE( metricValue(getErrorMatrix6BitRand(), metric) == 57.0 / 8.0);
    REQUIRE( metricValue(getErrorMatrix4to1AND(), metric) == 1.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4to1OR(), metric) == 15.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4to1Rand(), metric) == 7.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4BitCount(), metric) == 80.0 / 16.0);
}

TEST_CASE("WEIGHTED_LIN_LSB") {
    auto metric = Metric::WEIGHTED_LIN_LSB;
    REQUIRE( metricValue(getErrorMatrix6BitRand(), metric) == 69.0 / 8.0);
    REQUIRE( metricValue(getErrorMatrix4to1AND(), metric) == 1.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4to1OR(), metric) == 15.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4to1Rand(), metric) == 7.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4BitCount(), metric) == 80.0 / 16.0);
}

TEST_CASE("HIERARCHICAL_LIN_MSB") {
    auto metric = Metric::HIERARCHICAL_LIN_MSB;
    REQUIRE( metricValue(getErrorMatrix6BitRand(), metric) == 30.0 / 8.0);
    REQUIRE( metricValue(getErrorMatrix4to1AND(), metric) == 1.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4to1OR(), metric) == 15.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4to1Rand(), metric) == 7.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4BitCount(), metric) == 49.0 / 16.0);
}

TEST_CASE("HIERARCHICAL_LIN_LSB") {
    auto metric = Metric::HIERARCHICAL_LIN_LSB;
    REQUIRE( metricValue(getErrorMatrix6BitRand(), metric) == 35.0 / 8.0);
    REQUIRE( metricValue(getErrorMatrix4to1AND(), metric) == 1.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4to1OR(), metric) == 15.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4to1Rand(), metric) == 7.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4BitCount(), metric) == 49.0 / 16.0);
}

TEST_CASE("E4_LSB") {
    auto metric = Metric::E4_LSB;
    REQUIRE( metricValue(getErrorMatrix6BitRand(), metric) == 46.0 / 8.0);
    REQUIRE( metricValue(getErrorMatrix4to1AND(), metric) == 1.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4to1OR(), metric) == 15.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4to1Rand(), metric) == 13.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4BitCount(), metric) == 60.0 / 16.0);
}

TEST_CASE("E4_MSB") {
    auto metric = Metric::E4_MSB;
    REQUIRE( metricValue(getErrorMatrix6BitRand(), metric) == 38.0 / 8.0);
    REQUIRE( metricValue(getErrorMatrix4to1AND(), metric) == 1.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4to1OR(), metric) == 15.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4to1Rand(), metric) == 13.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4BitCount(), metric) == 49.0 / 16.0);
}


TEST_CASE("E5_LSB") {
    auto metric = Metric::E5_LSB;
    REQUIRE( metricValue(getErrorMatrix6BitRand(), metric) == 46.0 / 8.0);
    REQUIRE( metricValue(getErrorMatrix4to1AND(), metric) == 1.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4to1OR(), metric) == 15.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4to1Rand(), metric) == 13.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4BitCount(), metric) == 63.0 / 16.0);
}

//TEST_CASE("E5_MSB") {
//    auto metric = Metric::E5_MSB;
//    REQUIRE( metricValue(getErrorMatrix6BitRand(), metric) == 38.0 / 8.0);
//    REQUIRE( metricValue(getErrorMatrix4to1AND(), metric) == 1.0 / 16.0);
//    REQUIRE( metricValue(getErrorMatrix4to1OR(), metric) == 15.0 / 16.0);
//    REQUIRE( metricValue(getErrorMatrix4to1Rand(), metric) == 13.0 / 16.0);
//    REQUIRE( metricValue(getErrorMatrix4BitCount(), metric) == 49.0 / 16.0);
//}


TEST_CASE("E6_LSB") {
    auto metric = Metric::E6_LSB;
    REQUIRE( metricValue(getErrorMatrix6BitRand(), metric) == 44.0 / 8.0);
    REQUIRE( metricValue(getErrorMatrix4to1AND(), metric) == 1.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4to1OR(), metric) == 15.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4to1Rand(), metric) == 7.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4BitCount(), metric) == 56.0 / 16.0);
}

//TEST_CASE("E6_MSB") {
//    auto metric = Metric::E6_MSB;
//    REQUIRE( metricValue(getErrorMatrix6BitRand(), metric) == 38.0 / 8.0);
//    REQUIRE( metricValue(getErrorMatrix4to1AND(), metric) == 1.0 / 16.0);
//    REQUIRE( metricValue(getErrorMatrix4to1OR(), metric) == 15.0 / 16.0);
//    REQUIRE( metricValue(getErrorMatrix4to1Rand(), metric) == 13.0 / 16.0);
//    REQUIRE( metricValue(getErrorMatrix4BitCount(), metric) == 49.0 / 16.0);
//}


TEST_CASE("E7_LSB") {
    auto metric = Metric::E7_LSB;
    REQUIRE( metricValue(getErrorMatrix6BitRand(), metric) == 47.0 / 8.0);
    REQUIRE( metricValue(getErrorMatrix4to1AND(), metric) == 1.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4to1OR(), metric) == 15.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4to1Rand(), metric) == 13.0 / 16.0);
    REQUIRE( metricValue(getErrorMatrix4BitCount(), metric) == 60.0 / 16.0);
}

//TEST_CASE("E7_MSB") {
//    auto metric = Metric::E7_MSB;
//    REQUIRE( metricValue(getErrorMatrix6BitRand(), metric) == 38.0 / 8.0);
//    REQUIRE( metricValue(getErrorMatrix4to1AND(), metric) == 1.0 / 16.0);
//    REQUIRE( metricValue(getErrorMatrix4to1OR(), metric) == 15.0 / 16.0);
//    REQUIRE( metricValue(getErrorMatrix4to1Rand(), metric) == 13.0 / 16.0);
//    REQUIRE( metricValue(getErrorMatrix4BitCount(), metric) == 49.0 / 16.0);
//}

#endif // BITERRORTEST_HPP
