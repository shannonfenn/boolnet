#include "BitError.hpp"

#include <iostream>

namespace BitError {
    Metric metricFromName(const string& name) {
        if(name.compare("simple") == 0)                     return Metric::SIMPLE;
        else if(name.compare("weighted lin msb") == 0)      return Metric::WEIGHTED_LIN_MSB;
        else if(name.compare("weighted lin lsb") == 0)      return Metric::WEIGHTED_LIN_LSB;
        else if(name.compare("hierarchical lin msb") == 0)  return Metric::HIERARCHICAL_LIN_MSB;
        else if(name.compare("hierarchical lin lsb") == 0)  return Metric::HIERARCHICAL_LIN_LSB;
        else if(name.compare("worst sample lin msb") == 0)  return Metric::WORST_SAMPLE_LIN_MSB;
        else if(name.compare("worst sample lin lsb") == 0)  return Metric::WORST_SAMPLE_LIN_LSB;
    
        else if(name.compare("weighted exp msb") == 0)      return Metric::WEIGHTED_EXP_MSB;
        else if(name.compare("weighted exp lsb") == 0)      return Metric::WEIGHTED_EXP_LSB;
        else if(name.compare("hierarchical exp msb") == 0)  return Metric::HIERARCHICAL_EXP_MSB;
        else if(name.compare("hierarchical exp lsb") == 0)  return Metric::HIERARCHICAL_EXP_LSB;
        else if(name.compare("worst sample exp msb") == 0)  return Metric::WORST_SAMPLE_EXP_MSB;
        else if(name.compare("worst sample exp lsb") == 0)  return Metric::WORST_SAMPLE_EXP_LSB;
        else if(name.compare("e4 msb") == 0)                    return Metric::E4_MSB;
        else if(name.compare("e4 lsb") == 0)                    return Metric::E4_LSB;

        else if(name.compare("e5 msb") == 0)                    return Metric::E5_MSB;
        else if(name.compare("e5 lsb") == 0)                    return Metric::E5_LSB;

        else if(name.compare("e6 msb") == 0)                    return Metric::E6_MSB;
        else if(name.compare("e6 lsb") == 0)                    return Metric::E6_LSB;

        else if(name.compare("e7 msb") == 0)                    return Metric::E7_MSB;
        else if(name.compare("e7 lsb") == 0)                    return Metric::E7_LSB;

        else throw invalid_argument(string("metricFromName - ") + name);
    }
    
    ostream& operator<<(std::ostream& out, Metric m) {
        switch (m) {
        case Metric::SIMPLE:                out << "simple"; break;
        case Metric::WEIGHTED_LIN_MSB:      out << "weighted lin msb"; break;
        case Metric::WEIGHTED_LIN_LSB:      out << "weighted lin lsb"; break;
        case Metric::HIERARCHICAL_LIN_MSB:  out << "hierarchical lin msb"; break;
        case Metric::HIERARCHICAL_LIN_LSB:  out << "hierarchical lin lsb"; break;
        case Metric::WORST_SAMPLE_LIN_MSB:  out << "worst sample lin msb"; break;
        case Metric::WORST_SAMPLE_LIN_LSB:  out << "worst sample lin lsb"; break;
        case Metric::WEIGHTED_EXP_MSB:      out << "weighted exp msb"; break;
        case Metric::WEIGHTED_EXP_LSB:      out << "weighted exp lsb"; break;
        case Metric::HIERARCHICAL_EXP_MSB:  out << "hierarchical exp msb"; break;
        case Metric::HIERARCHICAL_EXP_LSB:  out << "hierarchical exp lsb"; break;
        case Metric::WORST_SAMPLE_EXP_MSB:  out << "worst sample exp msb"; break;
        case Metric::WORST_SAMPLE_EXP_LSB:  out << "worst sample exp lsb"; break;
        case Metric::E4_MSB:                out << "e4 msb"; break;
        case Metric::E4_LSB:                out << "e4 lsb"; break;
        case Metric::E5_MSB:                out << "e5 msb"; break;
        case Metric::E5_LSB:                out << "e5 lsb"; break;
        case Metric::E6_MSB:                out << "e6 msb"; break;
        case Metric::E6_LSB:                out << "e6 lsb"; break;
        case Metric::E7_MSB:                out << "e7 msb"; break;
        case Metric::E7_LSB:                out << "e7 lsb"; break;
        }
        return out;
    }
    
    // Many of the calculations in this method rely on errorMatrix only being comprised of 1s and 0s
//    double metricValue(const vector<vector<char>>& errorMatrix, Metric metric, bool print) {
    double metricValue(const vector<vector<char>>& errorMatrix, Metric metric) {
        double error = 0.0;
        // for weighted and hierarchical methods
        size_t n = errorMatrix.front().size();
        size_t pos;

        switch(metric) {
        case Metric::SIMPLE:
            // mean valud for error matrix
            for(const auto& sample : errorMatrix) {
                error += accumulate(sample.begin(), sample.end(), 0);
            }
            return error / errorMatrix.size();
    
        case Metric::WEIGHTED_LIN_MSB:
            // mean value for error matrix with columns weighted increasingly
            // in a linear manner
            for(const auto& sample : errorMatrix) {
                for(size_t k=0; k < n; k++) {
                    error += sample[k] * (k + 1.0);
                }
            }
            return error / errorMatrix.size();
    
        case Metric::WEIGHTED_LIN_LSB:
            // mean value for error matrix with columns weighted decreasingly
            // in a linear manner
            for(const auto& sample : errorMatrix) {
                for(size_t k=0; k < n; k++) {
                    error += sample[k] * (n - k);
                }
            }
            return error / errorMatrix.size();
    
        case Metric::WEIGHTED_EXP_MSB:
            // mean value for error matrix with columns weighted increasingly
            // in a exponential manner
            for(const auto& sample : errorMatrix) {
                for(size_t k=0; k < n; k++) {
                    error += sample[k] << k;
                }
            }
            return error / errorMatrix.size();
                
        case Metric::WEIGHTED_EXP_LSB:
            // mean value for error matrix with columns weighted decreasingly
            // in a exponential manner
            for(const auto& sample : errorMatrix) {
                for(size_t k=0; k < n; k++) {
                    error += sample[k] << (n - k - 1);
                }
            }
            return error / errorMatrix.size();
                
        case Metric::HIERARCHICAL_LIN_MSB:
            // each row is given the index of the highest column for which
            // there is an error the average of these is returned
            for(const auto& sample : errorMatrix) {
                // Find the highest bit for which this disagrees
                error += sample.rend() - find(sample.rbegin(), sample.rend(), 1);
            }
            return error / errorMatrix.size();
                    
        case Metric::HIERARCHICAL_LIN_LSB:      
            // each row is given the inverse index of the lowest column for which
            // there is an error the average of these is returned
            for(const auto& sample : errorMatrix) {
                // Find the lowest bit for which this disagrees
                error += sample.end() - find(sample.begin(), sample.end(), 1);
            }
            return error / errorMatrix.size();
            
        case Metric::HIERARCHICAL_EXP_MSB:        
            for(const auto& sample : errorMatrix) {
                // Find the highest bit for which this disagrees
                auto pos = find(sample.rbegin(), sample.rend(), 1);
                
                if(pos != sample.rend()) {
                    error += 1 << (sample.rend() - pos);
                }
            }
            return error / errorMatrix.size();
            
        case Metric::HIERARCHICAL_EXP_LSB:      
            for(const auto& sample : errorMatrix) {
                // Find the lowest bit for which this disagrees
                auto pos = find(sample.begin(), sample.end(), 1);
                
                if(pos != sample.end()) {
                    error += 1 << (sample.end() - pos);
                }
            }
            return error / errorMatrix.size();
            
        case Metric::WORST_SAMPLE_LIN_MSB:
            pos = 0;
            for(const auto& sample : errorMatrix) {
                // Only bother checking while the error could be worse
                for(size_t k=n; k > pos; k--) {
                    if(sample[k]) {
                        pos = k;    // This will force a break
                    }
                }
                
                // Break early if error cannot be increased
                if(pos == n) {
                    break;
                }
            }
            
            return pos;
            
        case Metric::WORST_SAMPLE_LIN_LSB:
            // Find the lowest bit for which this disagrees
            pos = 0;
            for(const auto& sample : errorMatrix) {
                // Only bother checking while the error could be worse
                for(size_t k=0; k < n - pos; k++) {
                    if(sample[k]) {
                        pos = n - k;    // This will force a break
                    }
                }
                
                // Break early if error cannot be increased
                if(pos == n) {
                    break;
                }
            }
            
            return pos;
    
        case Metric::WORST_SAMPLE_EXP_MSB:
            pos = 0;
            for(const auto& sample : errorMatrix) {
                // Only bother checking while the error could be worse
                for(size_t k=n; k > pos; k--) {
                    if(sample[k]) {
                        pos = k;    // This will force a break
                    }
                }
                
                // Break early if error cannot be increased
                if(pos == n) {
                    break;
                }
            }
            
            return 1 << pos;
    
        case Metric::WORST_SAMPLE_EXP_LSB:
            // Find the lowest bit for which this disagrees
            pos = 0;
            for(const auto& sample : errorMatrix) {
                // Only bother checking while the error could be worse
                for(size_t k=0; k < n - pos; k++) {
                    if(sample[k]) {
                        pos = n - k;    // This will force a break
                    }
                }
                
                // Break early if error cannot be increased
                if(pos == n) {
                    break;
                }
            }
            
            return 1 << pos;
        case Metric::E4_MSB:
            pos = 0;

            for(auto s_it = errorMatrix.begin(); s_it < errorMatrix.end(); s_it++) {

                // Only bother checking while the error could be worse
                for(size_t k=n; k > pos; k--) {
                    if((*s_it)[k-1]) {
                        pos = k;    // This will force a break
                    }
                }

                error += pos;
                if(pos == n) {
                    error += pos * (errorMatrix.end() - s_it - 1);
                    s_it = errorMatrix.end();
                }
            }

            return error / errorMatrix.size();
        case Metric::E4_LSB:
        {
            // Find the lowest bit for which this disagrees
            pos = 0;
            for(auto s_it = errorMatrix.begin(); s_it < errorMatrix.end(); s_it++) {
                // Only bother checking while the error could be worse
                for(size_t k=0; k < n - pos; k++) {
                    if((*s_it)[k]) {
                        pos = n - k;    // This will force a break
                    }
                }

                error += pos;
                if(pos == n) {
                    error += pos * (errorMatrix.end() - s_it - 1);
                    s_it = errorMatrix.end();
                }
            }

            return error / errorMatrix.size();
        }
        case Metric::E5_MSB:
        {
            pos = 0;
            vector<vector<char>>::const_iterator samp;
            for(auto s_it = errorMatrix.begin(); s_it < errorMatrix.end(); s_it++) {

                // Only bother checking while the error could be worse
                for(size_t k=n; k > pos; k--) {
                    if((*s_it)[k-1]) {
                        pos = k;    // This will force a break
                        samp = s_it;    // record which sample had this error first
                    }
                }

                if(pos == n) {
                    break;
                }
            }

            if(pos > 0) {
                error += pos * (errorMatrix.end() - samp);
                error += (pos-1) * (samp - errorMatrix.begin());
            }

            return error / errorMatrix.size();
        }
        case Metric::E5_LSB:
        {
            pos = 0;
            vector<vector<char>>::const_iterator samp;
            for(auto s_it = errorMatrix.begin(); s_it < errorMatrix.end(); s_it++) {

                // Only bother checking while the error could be worse
                for(size_t k=0; k < n - pos; k++) {
                    if((*s_it)[k]) {
                        pos = n - k;    // This will force a break
                        samp = s_it;    // record which sample had this error first
                    }
                }

                if(pos == n) {
                    break;
                }
            }

            if(pos > 0) {
                error += pos * (errorMatrix.end() - samp);
                error += (pos-1) * (samp - errorMatrix.begin());
            }

            return error / errorMatrix.size();
        }
        case Metric::E6_MSB:
        {
            for(pos = n; pos > 0; pos--) {
                for(auto s_it = errorMatrix.begin(); s_it < errorMatrix.end(); s_it++) {
                    // count the number of samples with an error in this bit position
                    error += (*s_it)[pos-1];
                }
                if(error > 0) {
                    // if there were any samples in error for this bit, then all further bits are counted as
                    // in error for all samples
                    error += (pos-1) * errorMatrix.size();
                    break;
                }
            }

            return error / errorMatrix.size();
        }
        case Metric::E6_LSB:
        {
            for(pos = 0; pos < n; pos++) {
                for(auto s_it = errorMatrix.begin(); s_it < errorMatrix.end(); s_it++) {
                    // count the number of samples with an error in this bit position
                    error += (*s_it)[pos];
                }
                if(error > 0) {
                    // if there were any samples in error for this bit, then all further bits are counted as
                    // in error for all samples
                    error += (n-pos-1) * errorMatrix.size();
                    break;
                }
            }

            return error / errorMatrix.size();
        }
        case Metric::E7_MSB:
            // Forces the learner to learn the first example completely (from msb to lsb) before
            // being rewarded for future examples
            for(auto s_it = errorMatrix.begin(); s_it < errorMatrix.end(); s_it++) {
                // Find the highest bit for which this disagrees
                auto pos = find(s_it->rbegin(), s_it->rend(), 1);

                if(pos != s_it->rend()) {
                    error += (s_it->rend() - pos);                 // Error for the rest of this row
                    error += n * (errorMatrix.end() - s_it - 1);    // Error for the rest of the matrix
                    break;                       // Terminate loop
                }
            }

            return error / errorMatrix.size();
        case Metric::E7_LSB:
            // Forces the learner to learn the first example completely (from lsb to msb) before
            // being rewarded for future examples
        {
            for(auto s_it = errorMatrix.begin(); s_it < errorMatrix.end(); s_it++) {
                // Find the lowest bit for which this disagrees
                auto pos = find(s_it->begin(), s_it->end(), 1);

                if(pos != s_it->end()) {
                    error += (s_it->end() - pos);                 // Error for the rest of this row
                    error += n * (errorMatrix.end() - s_it - 1);    // Error for the rest of the matrix
                    break;                       // Terminate loop
                }
            }

            return error / errorMatrix.size();
        }
        default:
            throw invalid_argument("Metric");
        }
    }
}
