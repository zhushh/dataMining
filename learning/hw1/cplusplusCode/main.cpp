/*************************************************************************
 > File Name: main.cpp
 > Author: 
 > Mail: 
 > Created Time: Mon 28 Mar 2016 11:35:31 PM CST
 ************************************************************************/

#include <iostream>
using namespace std;

#include "LinearRegression.cpp"

int main() {
    LinearRegression linear(385, 0.08, 500);

    linear.train("train.csv");

    linear.prediction("test.csv", "ans.csv");
    return 0;
}
