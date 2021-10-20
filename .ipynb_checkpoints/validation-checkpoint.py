#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import math
import json
import sys
from InduceC45 import c45, readFiles
from classifier import classify, readArrange, initializeConfusion


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
    
    df,tmp,labeled = readFiles(file1,restr)
    
    return df, int(k), labeled

def predict_kfold(df, numSplits, threshold, labeled):
    prev=None
    kfoldPreds = []
    
    confusion=initializeConfusion(df)
    foldAccuracies=[]
    accErr=[0,0]
    
    # all but one cross validation
    if numSplits == -1:
        numSplits = len(df)-1
    
    # split dataset kfold and generate predictions
    if numSplits <= 1:
        kfoldPreds += classify(confusion, accErr, df, c45(df, df.columns[:-1], threshold), silent=True, labeled=labeled)
        foldAccuracies.append(accErr)
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
                
                tree=c45(trainingData, df.columns[:-1], threshold)
                
                foldAccuracies.append(accErr)
                
                print("completed split #",splitnum)
                prev=i
        
        trainingData = df[:prev]
        classifyData = df[prev:]
        print(confusion)
        kfoldPreds += classify(confusion, accErr, classifyData, c45(trainingData, df.columns[:-1], threshold), silent=True, labeled=labeled)
        foldAccuracies.append(accErr)
    
    ret = pd.DataFrame(kfoldPreds, columns=['index', 'prediction']).set_index('index')
    ret['actual'] = df.loc[:,df.columns[-1]:]
    return ret


if __name__ == '__main__':
    df,k,labeled = getArgs()
    print(predict_kfold(df, k, 0.2, labeled))

# pierce, consulting firm customers, base for later jobs