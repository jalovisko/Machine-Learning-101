#libraries

import pandas as pd #data Analysis
import numpy as np  #scientific compution
import seaborn as sns #statistical plotting
import matplotlib.pyplot as plt  #plot
#    % matplotlib inline
import math  #BASE MATHEMATICS
 
# IMPORT DATA
non=[" ?", "?"]
adult_data = pd.read_csv("adult.csv" , names=['age','workclass','fnlwgt', 'education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','class'])
adult_data.head(10)
adult_data=adult_data.replace(non, np.nan)  #replace '?' values to nan

#Analyzing Data
#For categorixed Data
sns.countplot(x="class" , data = adult_data)
#in comparison with Class
sns.countplot(x="class" , hue="sex",  data = adult_data)
sns.countplot(x="class" , hue="race", data=adult_data)
sns.countplot(x="class" , hue="relationship", data=adult_data)
sns.countplot(x="class" , hue="workclass", data=adult_data)
sns.countplot(x="class" , hue="education", data=adult_data)
sns.countplot(x="class" , hue="occupation", data=adult_data)
sns.countplot(x="class" , hue="marital-status", data=adult_data)
sns.countplot(x="class" , hue="native-country", data=adult_data)
#alone
sns.countplot(x="sex",  data = adult_data)
sns.countplot(x="race", data=adult_data)
sns.countplot(x="relationship", data=adult_data)
sns.countplot(x="workclass", data=adult_data)
sns.countplot(x="education", data=adult_data)
sns.countplot(x="occupation", data=adult_data)
sns.countplot(x="marital-status", data=adult_data)
sns.countplot(x="native-country", data=adult_data)      #~99% are from US
adult_data=adult_data.drop("native-country" , axis=1)   #deleting a maifold feature
#For Continius Data     ??
adult_data["age"].plot.hist()
adult_data["fnlwgt"].plot.hist(bins=20, figsize=(10,5))
adult_data.info()
adult_data["education-num"].plot.hist()
adult_data["capital-gain"].plot.hist()
adult_data["capital-loss"].plot.hist()
adult_data["hours-per-week"].plot.hist()
adult_data.isin([0]).sum()[10:12] #showing the number of 0 in column 1
(adult_data.sum()[10:12]-adult_data.isin([0]).sum()[10:12])/(adult_data.sum()[10:12])   #as we sww 99.91% of datas in this column are 0
adult_data=adult_data.drop("capital-gain", axis=1)
adult_data=adult_data.drop("capital-loss"  , axis=1)

#Cleaning Data
adult_data.isnull()     #showing false/Treu values, True means it is naN
adult_data.isnull().sum()   #showing the number of True values in each column
sns.heatmap(adult_data.isnull(), yticklabels=False, cmap="viridis") #seeing naN values
adult_data.dropna(inplace=True) #remove instances with missing or malformed features

#categorical one-hot encoding
sex=pd.get_dummies(adult_data['sex'], drop_first=True)
races=pd.get_dummies(adult_data['race'], drop_first=True)
relation=pd.get_dummies(adult_data['relationship'], drop_first=True)
workclass=pd.get_dummies(adult_data['workclass'], drop_first=True)
education=pd.get_dummies(adult_data['education'], drop_first=True)
occupation=pd.get_dummies(adult_data['occupation'], drop_first=True)
marital=pd.get_dummies(adult_data['marital-status'], drop_first=True)
income=pd.get_dummies(adult_data['class'], drop_first=True)
adult_data=pd.concat([adult_data,sex,races,relation,workclass,education,occupation,marital,income], axis=1 )
adult_data=adult_data.drop(['sex','race','relationship','workclass','education','occupation','marital-status','class'], axis=1)
adult_data.head(5)

# Feature_scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
adult_data[['age', 'fnlwgt','education-num','hours-per-week']] = scaler.fit_transform(adult_data[['age', 'fnlwgt','education-num','hours-per-week']])

#train data
X=adult_data.drop(' >50K', axis=1)
y=adult_data[' >50K']

 
        