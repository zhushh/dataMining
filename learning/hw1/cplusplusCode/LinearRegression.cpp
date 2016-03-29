#ifndef __LINEARREGRESSION
#define __LINEARREGRESSION

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
using namespace std;

#include <cstdio>
#include <cstdlib>
#include <cstring>

class LinearRegression {
    public:
        LinearRegression(int n, double a = 0.01, int runs=1000) :
            feature_size(n), alpha(a), runTimes(runs) {
                data_size = 0;
                for (int i = 0; i < feature_size; i++ ) {
                    theta.push_back(0.0);
                }
            }

        virtual ~LinearRegression() {}

        void train(const string &fileName) {
            readData(fileName);

            // std::vector<double> tempTheta(feature_size);
            // std::vector<double> hx(data_size);
            double tempTheta[feature_size];
            double hx[data_size];

            cout << "Start training..." << endl;
            for (int i = 0; i < runTimes; i++) {
                // cout << i  << "runs" << endl;
                // computing hx
                for (int j = 0; j < data_size; j++) {
                    hx[j] = 0.0;
                    for (int k = 0; k < feature_size; k++) {
                        hx[j] += theta[k] * dataMat[j][k];
                    }
                }

                // computing theta
                for (int j = 0; j < feature_size; j++) {
                    double hx_sum = 0.0;
                    for (int k = 0; k < data_size; k++) {
                        hx_sum += (hx[k] - y[k]) * dataMat[k][j];
                    }
                    tempTheta[j] = theta[j] - (alpha / data_size) * hx_sum;
                }

                // updating theta
                for (int j = 0; j < feature_size; j++) {
                    theta[j] = tempTheta[j];
                }
            }
            cout << "Training Finished." << endl;
        }

        // default storing the prediction result into prediction.csv
        void prediction(const string &fileName, const string outfile = "prediction.csv") {
            readData(fileName);
            std::fstream fout;

            fout.open(outfile.c_str(), std::fstream::in | std::fstream::out);
            fout << "Id,reference" << endl;

            cout << "Start prediction:" << endl;
            for (int i = 0; i < data_size; i++) {
                double prediction = 0.0;
                for (int j = 0; j < feature_size; j++) {
                    prediction += theta[j] * dataMat[i][j];
                }
                if (outfile == "") {
                    cout << i << "\t" << prediction << endl;
                } else {
                    fout << i << "," << prediction << endl;
                }
            }
            cout << "Prediction Finished." << endl;


            fout.close();
        }

    private:
        int data_size;
        int feature_size;
        double alpha;
        int runTimes;

        std::vector< std::vector<double> > dataMat;
        std::vector<double> theta;
        std::vector<double> y;

        std::vector<string> split(const string &str, char delim) {
            std::vector<std::string> elems;
            std::stringstream ss(str);
            std::string item;
            while (getline(ss, item, delim)) {
                elems.push_back(item);
            }
            return elems;
        }

        void readData(const string &fileName) {
            std::fstream fin(fileName.c_str(), std::ifstream::in);

            cout << "Reading data from file..." << endl;
            int curRow = 0;
            dataMat.clear();
            string buffer;
            getline(fin, buffer);       // ignore the first line
            while (getline(fin, buffer)) {
                std::vector<std::string> elems = split(buffer, ',');
                if (elems.size() >= feature_size) {
                    std::vector<double> row_data(feature_size);
                    row_data.push_back(1.0);
                    for (int i = 1; i < feature_size; i++) {
                        row_data.push_back(atof(elems[i].c_str()));
                    }

                    dataMat.push_back(row_data);

                    if (elems.size() > feature_size) {
                        y.push_back(atof(elems[feature_size].c_str()));
                    }
                    curRow++;
                }
            }
            data_size = curRow;
            cout << "Finished read data." << endl;
            cout << "Sum rows: " << data_size << endl << endl;
            fin.close();
        }
};

#endif
