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
    df = df.drop([0,1], axis=0)
    df = df[[c for c in df if c not in [aclass]] + [aclass]]
    return df


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
    
    data = readArrange(file1)
    tree = None
    with open(file2) as f:
        tree = json.load(f)
    
    return data, tree


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

            
# In[59]:
def confusionMatrix(resultDf):
    predictions = resultDf["Prediction"].unique()
    print(predictions)

def classify(d=None, t=None, silent=False):
    numErrors = 0
    numCorrect = 0
    totalClassified = 0
    accuracy = 0
    errorRate = 0
    
    data=None
    tree=None
    if d is None and t is None:
        data, tree = readFiles()
    else:
        data=d
        tree=t

    out = []
    for i, row in data.iterrows():
        prediction = traverseTree(row, tree["node"], "node")
        
        if silent:
            out.append([i,prediction])
        else:
            actual = row[data.columns[-1]] 
            newLine = []
            for c in row:
                newLine.append(c)
            newLine.append(prediction)
            out.append(newLine)

            if prediction != actual:
                numErrors += 1
            else:
                numCorrect += 1

            totalClassified += 1
    
    if silent:
        return out
    else:
        cols = [c for c in data.columns] + ["Prediction"]

        accuracy = numCorrect / totalClassified
        errorRate = numErrors / totalClassified


        print(pd.DataFrame(out, columns=cols))
        print("Total Records Classifed: ", totalClassified)
        print("Total Classified Correctly: ", numCorrect)
        print("Total Classified Incorrectly: ", numErrors)
        print("Accuracy: ", accuracy)
        print("Error Rate: ", errorRate)
    
if __name__ == "__main__":
    classify()
