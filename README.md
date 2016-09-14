# Titanic: Machine Learning from Disaster

## Goal and Motivation 
The goal in this challenge is to predict passenger survival using a set of given features.  There are a few N/As in the dataset provided, and it is important to take care of those to avoid errors.  The "Cleaning Data" section explains to methods used to fill in these N/As.  Some features are used to generate new features, this will be further explained in the "Feature Engineering" section.  

## Cleaning Data
The first thing I did was to combine the training and testing set.  In order to do this, I added a "Survived" column to the testing set so the number of columns are equal.  I decided to fill the N/As in the "Age" column with the median age of the combined data.  The N/As in the "Embarked" column are filled with the most common location, which is S.  The missing "Fare" are calculated using the median of the fare from the dataset.  All the categorical datas are then transformed to numbers.

## Feature Engineering
To improve the accuracy of the predictoin, I followed Trevor Stephens's tips to engineer some extra features.  I first categorized the passengers into Children and Adults.  For an individual under the age of 18, the feature "Child" will be 1, otherwise 0.  I then used a combination of Sibling/Spouse count and Parents/Children count to obtain the feature "Family_Size".  The +1 at the end is to include the passenger himself/herself.  It is suspected that family size might be related to the chance of escaping the sinking boat since they might not want to be separated.
The "Title" feature is extracted from the name of the passenger.  It is suspected that individuals with more noble titles are more likely to have access to life boats.  The feature "Family_ID" is used create a link between family size and last names.  The goal here is to group families by last names and their relative family size.

## Choosing Classifiers
There are five classifiers used here: random forest, extra trees, support vector machine, k nearest neighbors, and logistic regression.  The optimal parameters are obtained by using GridSearchCV.  It turned out that the random forest classifier provided the optimal results for this dataset.

## Result
By using the random forest classifier, the accuracy obtained is between 0.785 and 0.795.  This puts me on the Kaggle ranking of the top 30% of the competitors.
