# Wine_Quality-Prediction
Predicting the quality of red wine using machine learning

![](https://media.phillyvoice.com/media/images/09122019_red_wine_pexels.2e16d0ba.fill-735x490.jpg)

**Let us predict the quality of wine using a decision tree classifier**

Let us start by importing certain libraries:
```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
```

**NumPy** is not another programming language but a Python extension module. It provides fast and efficient operations on arrays of homogeneous data.**Pandas** is a software library written for the Python programming language for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series. the third import, **from sklearn.model_selection import train_test_split** is used to split our dataset into training and testing data, more of which will be covered later. The next import, **from sklearn import preprocessing** is used to preprocess the data before fitting into predictor, or converting it to a range of -1,1, which is easy to understand for the machine learning algorithms. The last import, **from sklearn import tree** is used to import our decision tree classifier, which we will be using for prediction.


Let us now import the dataset that we will be using for this project:
```
df=pd.read_csv('winequality-red.csv',sep=';')
```
Let us now see the contents inside the dataset:
```
df.head()
```

We see a bunch of columns with some values in them. Now, in every machine learning program, there are two things, **features** and **labels**. **Features** are the part of a dataset which are used to predict the label. And **labels** on the other hand are mapped to features. After the model has been trained, we give features to it, so that it can predict the labels.
So, if we analyse this dataset, since we have to predict the wine quality, the attribute **quality** will become our label and the rest of the attributes will become the features.
Our next step is to separate the features and labels into two different dataframes.
```
y = df.quality
X = df.drop('quality', axis=1)
```
Next, we have to split our dataset into test and train data, we will be using the train data to to train our model for predicting the quality. The next part, that is the test data will be used to verify the predicted values by the model.
```
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
```
We have used, train_test_split() function that we imported from sklearn to split the data. Notice we have used test_size=0.2 to make the test data 20% of the original data. The rest 80% is used for training.
Now let’s print and see the first five elements of data we have split using head() function.
```
X_train.head()
```
The next step is scaling the data.data is converted to fit in a range of -1 and 1. 
```
X_train_scaled = preprocessing.scale(X_train)
X_train_scaled
```

It is now time to predict the wine quality.Let us use Decision Tree Algorithm here.
```
dtree=tree.DecisionTreeClassifier()
dtree.fit(X_train, y_train)
```
We can use a function called score().This function checks how efficiently this algorithm helps in predicting the wine quality.
```
confidence = dtree.score(X_test, y_test)
print("\nThe confidence score:\n")
print(confidence)
```
Our predicted information will be stored in y_pred.
Let us print the classification report and the confusion matrix
```
y_pred = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred)).
```
It shows an accuracy of 61%



We will just take first five entries of both y_pred and y_test, print them and compare them.
Here the numpy array is converted to a list.
```
#converting the numpy array to list
x=np.array(y_pred).tolist()

#printing first 5 predictions
print("\nThe prediction:\n")
for i in range(0,5):
    print(x[i])
    
#printing first five expectations
print("\nThe expectation:\n")
y_test.head()
```
Here,the predictor got it wrong only once for the first five examples.this provides an accuracy of 80% for the first five examples.
With the increasing examples,the accuracy goes down,which was shown previously as 0.61

This brings us to the end of the blog.

Assignment during Online Internship with DLithe(www.dlithe.com)

