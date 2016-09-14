import util
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, AdaBoostClassifier
import re

# Disable the warning for false positive chained assignment
pd.options.mode.chained_assignment = None  # default='warn'

# Import data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Clean data
test['Survived'] = np.NaN
combined_data = pd.concat([train,test], ignore_index = True)
combined_data['Sex'] = util.transform_categorical_data(combined_data['Sex'], 'Sex')
combined_data['Age'] = combined_data['Age'].fillna(value = combined_data.Age.median())
combined_data['Embarked'] = combined_data["Embarked"].fillna('S')
combined_data['Embarked'] = util.transform_categorical_data(combined_data['Embarked'], 'Embarked')
combined_data['Fare'] = combined_data["Fare"].fillna(value = combined_data.Fare.median())

# Feature engineering
combined_data['Child'] = float(0)
combined_data['Child'][combined_data['Age'] < 18] = 1
combined_data['Child'][combined_data['Age'] >= 18] = 0

combined_data['Family_Size'] = combined_data['SibSp'] + combined_data['Parch'] + 1

combined_data['Title'] = ''
for row in list(range(0,len(combined_data))):
  combined_data['Title'][row] = re.findall("([A-Z]+[a-z]+)\.", combined_data['Name'][row])[0]
combined_data['Title'][combined_data['Title'] == 'Mme'] = 'Mlle'
combined_data['Title'][combined_data['Title'] == 'Dona'] = 'Lady'
combined_data['Title'][combined_data['Title'] == 'Countess'] = 'Lady'
combined_data['Title'][combined_data['Title'] == 'Jonkheer'] = 'Sir'
combined_data['Title'][combined_data['Title'] == 'Dr'] = 'Sir'
combined_data['Title'][combined_data['Title'] == 'Capt'] = 'Sir'
combined_data['Title'][combined_data['Title'] == 'Don'] = 'Sir'
combined_data['Title'][combined_data['Title'] == 'Major'] = 'Sir'
combined_data['Title'] = util.transform_categorical_data(combined_data['Title'], 'Title')

combined_data['LastName'] = ''
for row in list(range(0,len(combined_data))):
  combined_data['LastName'][row] = re.findall("(.+),", combined_data['Name'][row])[0]

combined_data['Family_ID'] = ''
for row in list(range(0,len(combined_data))):
  combined_data['Family_ID'][row] = str(combined_data['Family_Size'][row]) + combined_data['LastName'][row]
combined_data['Family_ID'][combined_data['Family_Size'] <= 2] = 'Small'
ID_Freq = pd.DataFrame(combined_data.Family_ID.value_counts())
ID_Freq['Family_ID'] = ID_Freq[ID_Freq['Family_ID'] <= 2]
ID_Freq = ID_Freq.dropna()
for ID in ID_Freq.index.tolist():
  combined_data['Family_ID'][combined_data['Family_ID'] == ID] = 'Small' 
combined_data['Family_ID'] = util.transform_categorical_data(combined_data['Family_ID'], 'Family_ID')

train_clean = combined_data.iloc[:891,:]
test_clean = combined_data.iloc[891:,:]
train_clean['Survived'] = train_clean['Survived'].astype(int)
test_clean = test_clean.drop('Survived', axis = 1)

'''
Choose features, available features are:

Original Features
  PassengerId:  ID
  Pclass:       Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)
  Name:         Name
  Sex:          Sex
  Age:          Age
  SibSp:        Number of Siblings/Spouses Aboard
  Parch:        Number of Parents/Children Aboard
  Ticket:       Ticket Number
  Fare:         Passenger Fare
  Cabin:        Cabin
  Embarked:     Port of Embarkation
Generated Features:
  Child:        1 = under 18, 0 = over 18
  Family_Size:  SibSp + Parch + 1(self)
  Title:        Title of the passenger
  LastName:     Passenger last name
  Family_ID:    Family_Size + LastName, for family size less than 2 it is categorized as small
'''

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'Family_Size', 'Family_ID']

X_train = train_clean[features].values
X_test = test_clean[features].values
y = train_clean['Survived'].values

'''
There are five types of classifier investigated, the abbreviation are noted as below: 
RF: Random Forest
ET: Extra Trees
SVM: Support Vector Machine
KNN: K Nearest Neighbors
LR: Logisic Regression
'''

# Find best parameters for classifier
classifier_name = 'RF'
util.obtain_parameters(classifier_name, X_train, y)

# Choose classifier
(training_accuracy, prediction) = util.classify(classifier_name, X_train, y, X_test)
print('The training accuracy obtained using %s classifier is %f.' %(classifier_name, training_accuracy))

# Export data for submission
filename = 'prediction_%s.csv' %classifier_name
PassengerId = np.array(test["PassengerId"]).astype(int)
df_prediction = pd.DataFrame(prediction, index = PassengerId, columns = ["Survived"])
df_prediction_csv = df_prediction.to_csv(filename, index_label = ["PassengerId"])
