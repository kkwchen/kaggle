
# coding: utf-8

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# for jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

from sklearn.preprocessing import Imputer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

# save the target and ID separately
train_ID = train_df['Id']
test_ID = test_df['Id']
target_df = train_df['SalePrice']


# create a list of headings, remove SalePrice because target
possible_features = train_df.columns.tolist()
possible_features.remove('SalePrice')
possible_features.remove('Id')

# save the length of each dataframe so we can combine them for
# pre-processing and then split them later
index_train = train_df.shape[0]
index_test = test_df.shape[0]

# combine for pre-processing, drop target so we do not manipulate it
# drop ID because uneeded for prediction
preprocess_df = pd.concat((train_df, test_df)).reset_index(drop=True)
preprocess_df.drop(['SalePrice'], axis=1, inplace=True)
preprocess_df.drop(['Id'], axis=1, inplace=True)


for feature in possible_features:
    print(feature)
    print(preprocess_df[feature].value_counts())
    print('\n')

preprocess_df.describe()

preprocess_df.hist(bins=50,figsize=(40,30))
plt.show()

preprocess_df.corr()

# separate dataframe by types for preprocessing
g = preprocess_df.columns.to_series().groupby(preprocess_df.dtypes).groups

# separate into groups by object, convert to a dictionary
g_dict = {k.name: v for k, v in g.items()}
print(g_dict['object'])
print('\n')

# use dictionary to get unique values for each categorical feature
object_features = []
total_features = 0
for item in g_dict['object']:
    print(item)
    object_features.append(item)
    print(preprocess_df[item].unique())
    list_length = len(preprocess_df[item].unique())
    print(list_length)
    total_features += list_length

    print('\n')

# for future, non object features
non_object = list(set(preprocess_df.columns.tolist()) - set(object_features))
print(non_object)

# data cleaning non object features
# impute the median value for non object features
imputer = Imputer(strategy="median")

imputer.fit(preprocess_df[non_object])

# find median
X = imputer.transform(preprocess_df[non_object])

# set median
preprocess_df[non_object] = pd.DataFrame(X, columns=non_object)

preprocess_df[non_object].describe()


# because lazy, fill missing categorical features with most frequent value
for of in object_features:
    frequent_value = preprocess_df[of].value_counts().index[0]
    preprocess_df[of] = preprocess_df[of].fillna(frequent_value)


preprocess_df[object_features].describe()

# one hot encoding of categorical values
pre_process_1hot = pd.get_dummies(preprocess_df)

pre_process_1hot.head()


# test to see if this is enough feature engineering
#split data sets again
processed_train = pre_process_1hot[:index_train]
processed_test = pre_process_1hot[index_train:]

processed_train.describe()


# normalize target

normalize_target = target_df.copy()

min_max_target = (normalize_target - normalize_target.min())/(normalize_target.max()-normalize_target.min())


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor


def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())



from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
gbr.fit(processed_train,target_df)
gbr_scores = cross_val_score(gbr,processed_train,target_df,scoring='neg_mean_squared_error',cv=10)

gbr_rmse=np.sqrt(-gbr_scores)

display_scores(gbr_rmse)



# parameter tuning
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'learning_rate':[.01,.05,.1,.5,.75],'n_estimators': [30,50,70,90],'max_features':[10,14,18,20]}
]

grid_search = GridSearchCV(gbr, param_grid, cv=5,
                           scoring='neg_mean_squared_error')

grid_search.fit(processed_train,target_df)


param_dict = {'n_estimators': [30,50,70,90],'max_features':[10,14,18,20], 'max_depth':[2,3,4,5,6]}


best_params = grid_search.best_params_

# some initial values
score_dict = {}
i = 0
last_score = 100000

param_dict = {'n_estimators': [30,50,70,90],'max_features':[10,14,18,20], 'max_depth':[2,3,4,5,6]}
new_param_grid = [param_dict]


# run once to get values
grid_search = GridSearchCV(gbr, new_param_grid, cv=5,
                           scoring='neg_mean_squared_error')
grid_search.fit(processed_train,target_df)

best_params = grid_search.best_params_
best_score = np.sqrt(-grid_search.best_score_)
last_param_grid = [param_dict]

print('Iteration: ', i)
print('Current Score: ', best_score)
print('Current Settings: ', best_params)

# iterate
while i < 20:
    if best_score < last_score:
        # param_dict['learning_rate'] = [max(.1,best_params['learning_rate']-.01),best_params['learning_rate'],best_params['learning_rate']+.01,best_params['learning_rate']+.05]
        param_dict['n_estimators'] = [max(1,best_params['n_estimators']-1),best_params['n_estimators'],best_params['n_estimators']+1,best_params['n_estimators']+5]
        param_dict['max_features'] = [max(1,best_params['max_features']-1),best_params['max_features'],best_params['max_features']+1,best_params['max_features']+5]
        param_dict['max_depth'] = [max(1,best_params['max_depth']-1),best_params['max_depth'],best_params['max_depth']+1,best_params['max_depth']+5]
        new_param_grid = [param_dict]
        i += 1
    else:
        # param_dict['learning_rate'] = [max(.1,best_params['learning_rate']-.05),best_params['learning_rate'],best_params['learning_rate'],best_params['learning_rate']+.01]
        param_dict['n_estimators'] = [max(1,best_params['n_estimators']-5),max(1,best_params['n_estimators']-1),best_params['n_estimators'],best_params['n_estimators']+1]
        param_dict['max_features'] = [max(1,best_params['max_features']-5),max(1,best_params['max_features']-1),best_params['max_features'],best_params['max_features']+1]
        param_dict['max_depth'] = [max(1,best_params['max_depth']-5),max(1,best_params['max_depth']-1),best_params['max_depth'],best_params['max_depth']+1]
        new_param_grid = [param_dict]
        i += 1

    grid_search = GridSearchCV(gbr, new_param_grid, cv=5,
                           scoring='neg_mean_squared_error')
    grid_search.fit(processed_train,target_df)

    best_params = grid_search.best_params_
    best_score = np.sqrt(-grid_search.best_score_)

    score_dict[best_score] = best_params

    print('Iteration: ', i)
    print('Current Score: ', best_score)
    print('Current Settings: ', best_params)




bestest_params = score_dict[min(score_dict.keys())]


gbr = GradientBoostingRegressor(max_depth=bestest_params['max_depth'],max_features=bestest_params['max_features'],n_estimators=bestest_params['n_estimators'])



gbr.fit(processed_train,target_df)


gbr_predict = gbr.predict(processed_test)

submission = test_df.copy()


submission['SalePrice'] = gbr_predict


submission[['Id','SalePrice']].to_csv('submission.csv', index=False)
