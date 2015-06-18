#ifndef BITERROR_HPP
#define BITERROR_HPP

#include <vector>
#include <numeric>
#include <algorithm>
#include <ostream>

using namespace std;

namespace BitError {
    enum Metric {SIMPLE,
                 WEIGHTED_LIN_MSB,        WEIGHTED_LIN_LSB,
                 WEIGHTED_EXP_MSB,        WEIGHTED_EXP_LSB,
                 HIERARCHICAL_LIN_MSB,    HIERARCHICAL_LIN_LSB,
                 HIERARCHICAL_EXP_MSB,    HIERARCHICAL_EXP_LSB,
                 WORST_SAMPLE_LIN_MSB,    WORST_SAMPLE_LIN_LSB,
                 WORST_SAMPLE_EXP_MSB,    WORST_SAMPLE_EXP_LSB,
                 E4_MSB,                  E4_LSB,
                 E5_MSB,                  E5_LSB,
                 E6_MSB,                  E6_LSB,
                 E7_MSB,                  E7_LSB};
    
    Metric metricFromName(const string& name);
    
    ostream& operator<<(std::ostream& out, Metric m);
    
    // Many of the calculations in this method rely on errorMatrix only being comprised of 1s and 0s
//    double metricValue(const vector<vector<char>>& errorMatrix, Metric metric = Metric::SIMPLE, bool print=false);
    double metricValue(const vector<vector<char>>& errorMatrix, Metric metric = Metric::SIMPLE);
}
#endif // BITERROR_HPP
