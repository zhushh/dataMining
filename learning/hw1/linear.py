#!/usr/bin/python
# coding: utf-8

import csv
import numpy as np

# return m, n, dataMat
def createMat(fileName):
    # read data from file
    csvfile = file(fileName)            # construct file obj
    reader = csv.reader(csvfile)        # open csv file
    data = []                           # store the training data
    flag = False
    for line in reader:
        if flag:
            data.append(map(float, line))
        else:
            flag = True

    dataMat = np.mat(data)
    csvfile.close()
    return dataMat

def costEstimation(theta, dataMat):
    m, n = np.shape(dataMat)
    cost = theta * dataMat[0:,0:n-1].T - dataMat[0:,n-1].T
    result = 0.0
    for i in range(m):
        result += (cost[0,i]*cost[0,i]) / float(2*m)
    return result

######################### start training##########################
dataMat = createMat('train.csv')
m, n = np.shape(dataMat)
dataMat[0:,0] = np.ones((m, 1), float)
theta = np.mat([0 for i in range(n-1)])
alpha = 0.07

i = 0
cost = costEstimation(theta, dataMat)
while abs(cost) > 35:
    print "the", i, "times, cost = ", cost
    i += 1
    theta = theta - ((alpha / float(m)) * ((theta * dataMat[0:,0:n-1].T - dataMat[0:, n-1].T) * dataMat[0:,0:n-1]))
    # print theta
    cost = costEstimation(theta, dataMat)

print theta

######################### prediction##############################
dataMat = createMat('test.csv')
m, n = np.shape(dataMat)
data = []
for i in range(m):
    reference = (theta * dataMat[i, 0:n].T)[0,0]
    data.append((i, reference))
    print i, reference

prediction = file('prediction.csv', 'w')
writer = csv.writer(prediction)
writer.writerow(['Id', 'reference'])
writer.writerows(data)
prediction.close()
print 'finish.'
