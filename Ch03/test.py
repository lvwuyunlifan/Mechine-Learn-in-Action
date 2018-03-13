#!/usr/bin/python2

from trees import *


myDat, labels = createDataSet()
print('I am myDat: ', myDat)
print('I am labels: ', labels)

ShonnonEntropy = calcShannonEnt(myDat)
print('the result ShonnonEntropy is: ', ShonnonEntropy)

retDataSet = splitDataSet(myDat, 0, 1)
print('retDataSet: ', retDataSet)
