import numpy as np
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def transform_categorical_data(col, col_name):
  le = preprocessing.LabelEncoder()
  le.fit(col)
  print('The categorical variables in %s has been converted to the following:' %col_name)
  for class_name in list(le.classes_):
    print(list(le.classes_).index(class_name), class_name)
  print('\n')
  return(le.transform(col))

def obtain_parameters(classifier_name, X_train, y):
  if classifier_name == 'RF':
    classifier = RandomForestClassifier()
    param_grid = dict(max_depth = [10, 20, 30, None], min_samples_split = [2,4,6], n_estimators = [500])
  elif classifier_name == 'ET':
    classifier = ExtraTreesClassifier()
    param_grid = dict(criterion = ['gini', 'entropy'],
                      max_depth = [10, 20, 30, None], 
                      min_samples_split = [2,4,6,8], 
                      min_samples_leaf = [1,2,3,4,5,6,7,8,9,10], 
                      n_estimators = [500])
  elif classifier_name == 'SVM':
    steps = [('scl', StandardScaler()), 
             ('clf', SVC())]
    classifier = Pipeline(steps)
    param_grid = dict(clf__C = [1,5,10,15,20,25,30], 
                      clf__kernel = ['rbf'], 
                      clf__gamma = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1])
  elif classifier_name == 'KNN':
    steps = [('scl', StandardScaler()), 
             ('clf', KNeighborsClassifier())]
    classifier = Pipeline(steps)
    param_grid = dict(clf__n_neighbors = list(range(1,31)))
  elif classifier_name == 'LR':
    steps = [('scl', StandardScaler()), 
         ('clf', LogisticRegression())]
    classifier = Pipeline(steps)
    param_grid = dict(clf__penalty = ['l1', 'l2'], clf__C = [0.1,1,5,6,7,8,10,15,20,25,30])
  grid = GridSearchCV(classifier, param_grid, cv = 10, scoring = 'accuracy', n_jobs = -1)
  grid.fit(X_train,y)
  grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
  print(grid_mean_scores)
  print(grid.best_estimator_)
  print(grid.best_params_)

def classify(classifier_name, X_train, y, X_test):
  # The parameters are obtained using the obtain_parameters function above
  if classifier_name == 'RF':
    clf = RandomForestClassifier(max_depth = 10, min_samples_split = 2, n_estimators = 500)
  elif classifier_name == 'ET':
    clf = ExtraTreesClassifier(criterion = 'entropy', 
                               min_samples_leaf = 1,
                               max_depth = 20, 
                               min_samples_split = 8, 
                               n_estimators = 500)
  elif classifier_name == 'SVM':
    steps = [('scl', StandardScaler()), ('clf', SVC(C = 5, gamma = 0.05, kernel = 'rbf'))]
    clf = Pipeline(steps)
  elif classifier_name == 'KNN':
    steps = [('scl', StandardScaler()), ('clf', KNeighborsClassifier(n_neighbors = 16))]
    clf = Pipeline(steps)
  elif classifier_name == 'LR':
    steps = [('scl', StandardScaler()), ('clf', LogisticRegression(C = 5, penalty = 'l2'))]
    clf = Pipeline(steps)
  fit = clf.fit(X_train,y)
  training_accuracy = fit.score(X_train, y)
  prediction = fit.predict(X_test)
  return (training_accuracy, prediction)

