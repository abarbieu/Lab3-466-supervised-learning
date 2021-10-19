#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import math
import json
import sys
from InduceC45 import c45, readFiles
from classifier import classify, readArrange


# In[85]:
def getArgs():
    restr=None
    if len(sys.argv) < 3:
        print("Not enough arguments.")
        exit(1)
    elif len(sys.argv) == 3:
        file1 = sys.argv[1]
        k = sys.argv[2]
    elif len(sys.argv) == 4:
        file1 = sys.argv[1]
        k = sys.argv[2]
        restr = sys.argc[3]
    
    df,tmp = readFiles(file1,restr)
    
    return df, int(k)

def predict_kfold(df, numSplits, threshold):
    prev=None
    kfoldPreds = []
    
    # all but one cross validation
    if numSplits == -1:
        numSplits = len(df)-1
    
    # split dataset kfold and generate predictions
    if numSplits <= 1:
        kfoldPreds += classify(df, c45(df, df.columns[:-1], threshold), silent=True)
    else:
        splitnum=0
        # go through indecies by fold length
        for i in range(0,len(df),int(len(df)/numSplits)):
            splitnum+=1
            if prev is None:
                prev=i
            else:
                trainingData = pd.concat([df[:prev], df[i:]])
                classifyData = df[prev:i]
                print("running C45 on split #", splitnum)
                kfoldPreds += classify(classifyData, c45(trainingData, df.columns[:-1], threshold), silent=True)
                print("completed split #",splitnum)
                prev=i
        
        trainingData = df[:prev]
        classifyData = df[prev:]
        kfoldPreds += classify(classifyData, c45(trainingData, df.columns[:-1], threshold), silent=True)
    
    ret = pd.DataFrame(kfoldPreds, columns=['index', 'prediction']).set_index('index')
    ret['actual'] = df.loc[:,df.columns[-1]:]
    return ret


if __name__ == '__main__':
    df,k = getArgs()
    print(predict_kfold(df, k, 0.2))

# pierce, consulting firm customers, base for later jobs