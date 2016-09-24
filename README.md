# Titanic: Machine Learning from Disaster

## Goal and Motivation 
The goal in this challenge is to predict passenger survival using a set of given features.  There are a few N/As in the dataset provided, and it is important to take care of those to avoid errors.  The "Cleaning Data" section explains to methods used to fill in these N/As.  Some features are used to generate new features, this will be further explained in the "Feature Engineering" section.  

## Cleaning Data
The data cleaning can be summarized as the following:

1. Combine the training and testing set
2. Fill the N/As
  * "Age" column with its median
  * "Embarked" column with the most common location, S
  * "Fare" column with its median
3. Transform categorical data to numbers

## Feature Engineering
The following features were obtained from feature engineering:
* Child - 1 for Age < 18, 0 otherwise
* Family_Size - SibSp + Parch + 1(self)
* Title - Extracted from Name
* LastName - Extracted from Name
* Family_ID - Family_Size + LastName, for family size less than 2 it is categorized as small

## Choosing Classifiers
The following five classifiers were chosen:
* Random Forest (RF)
* Extra Trees (ET)
* Support Vector Machine (SVM)
* K Nearest Neighbors
* Logistic Regression (LR)

## Result
By using the random forest classifier, the accuracy obtained is between 0.785 and 0.795.  This puts me on the Kaggle ranking of the top 25% of the competitors.
