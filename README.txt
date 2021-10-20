CSC 466
Fall 2021
Lab 3 Part 1

Eric Inman (eainman@calpoly.edu)
Aidan Barbieux (abarbieu@calpoly.edu)

Instructions to Run the Programs:

InduceC45.py:
    1. In the command line, run "python3 InduceC45.py <file.csv> <restrictions.txt>" (without parentheses). 
       The restrictions file is optional
       
    2. This will print out the generated decision tree as well as output a tree.json file that you can use as input
       for classifier.py. That file is always going to be named tree, so make sure you run InduceC45 on a new data set
       before trying that dataset on the classifier.
       
Classifier.py:
    1. In the command line, run "python3 classifier.py <file.csv> <file.json>". You can use the generated tree json file
       from running the InduceC45.py program. 
       
    2. If the data is labeled, classify.py will with output:
            1. total number of records classified
            2. total number of records classified correctly
            3. total number of records incorrectly classified
            4. overall accuracy and error rate of the classifier
            5. a confusion matrix
       If not, you will only see a resulting dataframe of the records along with a new column indicating their predictions.
       
Validation.py:
    1. In the command line, run "python3 validation.py <trainingFile.csv> <Number of Folds> <restrictions.txt>" (again 
       without the parentheses and restrictions are optional). 
       
    2. This will output the overall and average accuracy, the overall confusion matrix, and the predictions versus their 
       actual values.
       
    3. If you want this output in its own file, you'll have to pipe it through the commandline. 
       Ex: python3 validation.py nurses.csv 4 > nurses-results.out
       