import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

# Models from Scikit-learn
from sklearn.tree import DecisionTreeClassifier
# Model Evaluations
from sklearn.model_selection import train_test_split, cross_val_score
#load dataset
datas = pd.read_csv('final_test.csv')
#datas = datas.astype(np.float,errors ='ignore')
mean_s = datas.mean()
std_s = datas.std()
#we will be using mean and vareance in the data later in the code to convert back from the z-scrore dataset
#2> Removing Outliers
#this step is taken in sevral more steps :
# a. identifying the outliers
dfs = [] # list of dataframes giving Z-scores odered according to size
sizes = [] # list of diffrent sizes
for size_type in datas['size'].unique():
    sizes.append(size_type)
    ndf = datas[['weight','age','height']][datas['size'] == size_type] #sorting
    zscore = (ndf - ndf.mean()) / ndf.std()#calculating zscore
    #zscore = (ndf - mean_s) / std_s#calculating zscore

    dfs.append(zscore)
# b. removing the z-scores of features  that are not inbetween the range (-3,3)
for i in range(len(dfs)):
    dfs[i]['age'] = dfs[i]['age'][(dfs[i]['age']>-3) & (dfs[i]['age']<3)]
    dfs[i]['height'] = dfs[i]['height'][(dfs[i]['height']>-3) & (dfs[i]['height']<3)]
    dfs[i]['weight'] = dfs[i]['weight'][(dfs[i]['weight']>-3) & (dfs[i]['weight']<3)]
#c. assigning sizes accordingly and updating our dataframe pandas
for i in range(len(sizes)):
    dfs[i]['size'] = sizes[i]
datas = pd.concat(dfs)
print('this is z-score dataframe : datas')
print(datas)
#d. converting from z-score to orignal dataset after removing outliers
datas['age'] = mean_s['age'] + std_s['age']*datas['age']
datas['age'] = datas['age'].apply(np.ceil)
datas['weight'] = mean_s['weight'] + datas['weight']*std_s['weight']
datas['weight'] = datas['weight'].apply(np.ceil)
datas['height'] = mean_s['height'] + std_s['height']*datas['height']
datas['height'] = datas['height'].apply(np.ceil)
print('checking null values : ')
print(datas.isna().sum())
print(" ")
datas["age"] = datas["age"].fillna(datas['age'].median())
datas["height"] = datas["height"].fillna(datas['height'].median())
datas["weight"] = datas["weight"].fillna(datas['weight'].median())
#datas['size'] = datas['size'].map({"XXS": 1,"S": 2,"M" : 3,"L" : 4,"XL" : 5,"XXL" : 6,"XXXL" : 7})

X = datas.drop("size", axis=1)
# Target
y = datas["size"]
X_train, X_test, y_train, y_test, = train_test_split(X,y, test_size=0.10)
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
# making a pickle file
pickle.dump(classifier, open("model.pkl","wb"))