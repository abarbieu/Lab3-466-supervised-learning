#!/usr/bin/env python
# coding: utf-8

# ## Classifier

# Takes JSON input (from tree induction) and CSV file

# In[2]:


import numpy as np
import pandas as pd
import json
import sys


# In[3]:


def readArrange(filename):
    df = pd.read_csv(filename)
    aclass = df.iloc[1,0]
    labels = True
    if not isinstance(aclass, str):
        labels = False
    df = df.drop([0,1], axis=0)
    
    if labels:
        df = df[[c for c in df if c not in [aclass]] + [aclass]]
    return df, labels


# In[6]:


# returns a pandas dataframe from the csvfile and a dictionary from the jsonfile
def readFiles(file1=None, file2=None):
    if file1 is None and file2 is None:
        if len(sys.argv) != 3:
            print("Not enough arguments.")
            exit(1)
        else:
            file1 = sys.argv[1]
            file2 = sys.argv[2]
    
    data, isLabeled = readArrange(file1)
    tree = None
    with open(file2) as f:
        tree = json.load(f)
    
    return data, tree, isLabeled


# In[31]:


def traverseTree(row, tree, nodeType):
    if nodeType == "leaf":
        return tree["decision"]        
        
    elif nodeType == "node":
        val = row[tree["var"]]
        for obj in tree["edges"]:
            if obj["edge"]["value"] == val:
                newType = "leaf" if "leaf" in obj["edge"].keys() else "node"
                return traverseTree(row, obj["edge"][newType], newType)

def initializeConfusion(df):
    labels = resultDf.iloc[:, -1].unique() # labels are in last column (not using result df from classify)
    zeros = np.zeros(shape=(len(labels), len(labels)))
    confusion = pd.DataFrame(zeros, labels, labels)

    return confusion
           
    
def classify(confusion, vals, data=None, tree=None, silent=False, labeled=False):
    numErrors = 0
    numCorrect = 0
    totalClassified = 0
    accuracy = 0
    errorRate = 0
    
    if data is None and tree is None:
        data, tree = readFiles()  

    out = []
    for i, row in data.iterrows():
        prediction = traverseTree(row, tree["node"], "node")
        
        if silent:
            out.append([i,prediction])
        else:
            newLine = []
            for c in row:
                newLine.append(c)
            newLine.append(prediction)
            out.append(newLine)

        if labeled:
            actual = row[data.columns[-1]]
            confusion[actual][prediction] += 1
            if prediction != actual:
                numErrors += 1
            else:
                numCorrect += 1

        totalClassified += 1
            
    accuracy = 0
    errorRate = 0
    if labeled:
        accuracy = numCorrect / totalClassified
        errorRate = numErrors / totalClassified
        vals[0] += accuracy
        vals[1] += errorRate
#         print("Total Records Classifed: ", totalClassified)
#         print("Total Classified Correctly: ", numCorrect)
#         print("Total Classified Incorrectly: ", numErrors)
#         print("Accuracy: ", accuracy)
#         print("Error Rate: ", errorRate)
    
    if silent:
        return out
    else:
        cols = [c for c in data.columns] + ["Prediction"]
        print(pd.DataFrame(out, columns=cols))
    
if __name__ == "__main__":
    classify()
