#ifndef VECTORTOOLS_H
#define VECTORTOOLS_H

#include <vector>
#include <iostream>

using std::vector;
using std::ostream;

/*
 *  ostream operator for a vector of bits, MSB to the right
 */
inline ostream& operator<<(ostream& out, vector<char> bits) {
    for(auto b = bits.begin(); b < bits.end(); b++)
        out << (bool)*b;
    return out;
}

inline ostream& operator<<(ostream& out, vector<double> vals) {
    out << "[";
    for(auto v = vals.begin(); v < vals.end() - 1; v++)
        out << *v << ", ";
    out << vals.back() << "]";
    return out;
}

inline ostream& operator<<(ostream& out, vector<size_t> vals) {
    out << "[";
    for(auto v = vals.begin(); v < vals.end() - 1; v++)
        out << *v << ", ";
    out << vals.back() << "]";
    return out;
}

inline ostream& operator<<(ostream& out, vector<pair<size_t, size_t>> vals) {
    out << "[";
    for(auto v = vals.begin(); v < vals.end() - 1; v++)
        out << "[" << v->first << ", " << v->second << "], ";
    out << "[" << vals.back().first << ", " << vals.back().second << "]]";
    return out;
}

/*
 *  generates the binary expression for the given number in the given number of bits
 *  MSB at highest index
 */
inline vector<char> toBinary(int n, int base=2) {
    vector<char> bits(base, false);
    for (int i = 0; i < base; i++) {
        bits[i] = (1 << i & n) != 0;
    }
    return bits;
}

/*
 *  returns integral value expressed by bitset, assumes MSB at highest index
 */
inline size_t fromBinary(vector<char> bits) {
    size_t n = 0;
    size_t pow = 1;
    for(auto b : bits) {
        if(b)
            n += pow;
        pow = pow << 1;
    }
    return n;
}

#endif // VECTORTOOLS_H
