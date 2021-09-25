from imblearn.over_sampling import SMOTE

import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,ExtraTreesClassifier,ExtraTreesRegressor
from xgboost import XGBRegressor,XGBClassifier
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor #KNN
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,KFold
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report,mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report,mean_squared_error
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor,plot_tree,export_text
import imblearn
import os



### Data Preprocessing

# importing dataset
dataset = pd.read_csv("Data//diabetic_data.csv")

# dropping unwanted columns
dataset = dataset.drop(['encounter_id','patient_nbr','weight','payer_code','medical_specialty','diag_1','diag_2','diag_3'],axis=1)

# dropping duplicate raws
dataset.drop_duplicates(inplace=True)

# replacing values
dataset["race"].replace({"?":"Unknown"}, inplace=True)

# removing rows that contains 'Unknown/Invalid' for gender column
dataset = dataset[dataset.gender != 'Unknown/Invalid']

# data encoding

dataset['race'] = dataset['race'].map({'Caucasian':0, 'AfricanAmerican':1, 'Asian':2, 'Hispanic':3, 'Other':4, 'Unknown':5})
dataset['gender'] = dataset['gender'].map({'Male':1,'Female':0})
dataset['age'] = dataset['age'].map({'[0-10)':1,'[10-20)':2, '[20-30)':3, '[30-40)':4, '[40-50)':5, '[50-60)':6, 
                                    '[60-70)':7, '[70-80)':8, '[80-90)':9, '[90-100)':10})


# data encoding (col 16-38)

for col in dataset.iloc[:,16:39]:
    dataset[col] = dataset[col].map({'No':0, 'Steady':1, 'Up':2, 'Down':3})

# data encoding
dataset['change'] = dataset['change'].map({'No':0, 'Ch':1})
dataset['diabetesMed'] = dataset['diabetesMed'].map({'No':0, 'Yes':1})
dataset['readmitted'] = dataset['readmitted'].map({'NO':0, '>30':1, '<30':2})

dataset['max_glu_serum'] = dataset['max_glu_serum'].map({'None':0, '>300':1, 'Norm':2, '>200':3})
dataset['A1Cresult'] = dataset['A1Cresult'].map({'None':0, '>7':1, '>8':2, 'Norm':3})

# Splitting

# splitting data
x = dataset.drop('readmitted', axis=1).values# Input features (attributes)
y = dataset['readmitted'].values # Target vector
print('X shape: {}'.format(np.shape(x)))
print('y shape: {}'.format(np.shape(y)))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Random Forest

# transform the dataset
oversample = SMOTE()
x, y = oversample.fit_resample(x, y)

rfcl = RandomForestClassifier(n_estimators = 100, min_samples_split = 10,
min_samples_leaf = 4, max_features = 'auto',
max_depth = 70, bootstrap = True)

rfcl.fit(x_train, y_train)
prediction_test = rfcl.predict(x_test)

y_pred=rfcl.predict(x_test)

print(accuracy_score(y_test,y_pred))
pickle.dump(rfcl, open('model.pkl','wb'))