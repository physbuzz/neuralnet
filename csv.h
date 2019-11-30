#ifndef FNET_H
#define FNET_H

#include <iostream>
#include <vector>

/* Struct describing a rectangular CSV. Note that there's no 
 * internal condition enforcing data.size()==w*h, even though 
 * this should be the case.*/
struct RectCSV {
    int w;
    int h;
    std::vector<double> data;
};

/* Struct describing weights and biases in the context of a 
 * feedforward neural network. 
 *
 * n[i] is one plus the number of neurons in the layer. 
 *
 * weights is a list of weights, data.size() should be n.size()-1 
 * (weights connect layers, n[i] are the fenceposts). Each 
 * each data[i] should be a n[i] x n[i+1] row-major matrix. */
struct WeightCSV {
    std::vector<int> n;
    std::vector<std::vector<double> > weights;
};

/* Load a file specified by fname and populate the RectCSV* passed in.
 * Returns true on success, false otherwise. This function simply 
 * opens an fstream and calls loadRectCSVStream on it. */
bool loadRectCSV(std::string fname,RectCSV *data, int expected_w=0, int expected_h=0);

/* Load data from the specified stream and populate the RectCSV* passed in.
 * Returns true on success, false on failure.
 *
 * If expected_w is nonzero, the function demands the file only has 
 * expected_w columns to it, and fails otherwise.
 *
 * If expected_h is nonzero, the function only reads expected_h lines
 * from the stream (using std::getline), and fails if there aren't enough lines.
 *
 * If strict is false, expected_w and expected_h are just used to preallocate memory,
 * and if the dataset turns out to be bigger the method runs fine.
 *
 * If strict is true, only expected_h lines are read from the input file, and the method
 * fails (returns false) if expected_w is different from the width of the loaded data. */
bool loadRectCSVStream(std::istream& st,RectCSV *data,int expected_w=0, int expected_h=0);

/* save a rectangle of double floating point data as a csv.
 * Returns true on success, false on failure.*/
bool saveRectCSV(std::string fname, const RectCSV& data);
bool writeRectCSVStream(std::ostream& stream,const RectCSV& data);

/* Save neural network weight data as a csv. The format is as follows:
 * 1st row is the number of neurons in each layer, n[i].
 * The next n[0] rows and n[1] columns are the first weight matrix.
 * The next n[1] rows and n[2] columns are the second weight matrix.
 * etc. */
bool saveWeightCSV(std::string fname, const WeightCSV& data);
bool writeWeightCSVStream(std::ostream& stream, const WeightCSV& data);

/* Load neural network weight data according to the spec in 
 * saveWeightCSV. */
bool loadWeightCSV(std::string fname, WeightCSV* data);
bool loadWeightCSVStream(std::istream& stream, WeightCSV* data);





#endif
