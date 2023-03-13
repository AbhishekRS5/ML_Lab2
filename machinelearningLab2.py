#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd #imports two Python libraries, NumPy and Pandas, and note them as "np" and "pd", respectively.
col_names=["sepal_length","sepal_width","petal_length","petal_width","type"] #creates a list called "col_names" containing the column names for the data that will be loaded from the CSV file.
data=pd.read_csv("iris_dataset.csv",skiprows=1,header=None,names=col_names) # load data from a CSV file called "iris_dataset.csv" using the Pandas library.
#The "skiprows" argument tells Pandas to skip the first row which contains headers.
# "header" argument is set to None to indicate that there are no headers in the file.
# "names" argument is set to the "col_names" list to provide the column names.
data.head(10)#displays the first 10 rows of the loaded data


# In[2]:


X=data.iloc[:,:-1].values #extracts the feature matrix X from the DataFrame.# The ".iloc" function is used to select rows and columns by their integer index.
#".values" function is used to convert the resulting DataFrame to a NumPy array.
Y1=data.iloc[:,-1].values 
Y=Y1.reshape(-1,1) #.reshape to convert Y into a column vector
from  sklearn.model_selection import train_test_split #split the data into training and testing sets
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=41) #"test_size" argument specifies the data to be used for testing # "random_state" argument for reproducibility.
from sklearn import tree #import the "DecisionTreeClassifier" class
classifier=tree.DecisionTreeClassifier(min_samples_split=3,max_depth=3,criterion="entropy") #create a new decision tree classifier called "classifier"
#"min_samples_split"  specifies the minimum number of samples required to split a node, "max_depth" specifies the maximum depth of the tree, and "criterion" specify to use the entropy criterion.
classifier.fit(X_train,Y_train) #trains the decision tree classifier on the training data.
classifier.score(X_test,Y_test) #compute the accuracy score and evaluates the performance of the trained classifier
tree.plot_tree(classifier) #plots the decision tree


# In[ ]:




