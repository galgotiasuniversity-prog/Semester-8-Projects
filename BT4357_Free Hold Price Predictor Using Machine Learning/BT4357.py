#!/usr/bin/env python
# coding: utf-8

# #  1 Import relevant libraries

# In[4]:


import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.pipeline import Pipeline
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
get_ipython().run_line_magic('matplotlib', 'inline')


# # 2   Load dataset

# In[5]:


# Load the dataset and read the dataset

data = pd.read_csv("Solar Power dataset.csv")


# In[6]:


data


# In[7]:


data.info()


# In[8]:


data.isna().any()


# # 3  Explore the data

# # 3-i Identify available columns in the data

# In[9]:


data.columns


# # 3-ii Check for missing values, Impute missing values if present 

# In[10]:


data.isnull().sum()


# In[8]:


data.fillna(data['Pressure'].mean(), inplace =True)

data


# In[72]:


data.isnull().sum()


# # 3-iii Calculate summary statistics of numerical columns

# In[31]:


data.describe()


# 
# # 3-iv  Visualize data distribution

# In[11]:


plt.boxplot(data['Pressure'])


# In[12]:


sns.jointplot(x = data['PolyPwr'],y = data['Pressure'])


# In[13]:


sns.jointplot(x = data['PolyPwr'],y = data['Humidity'])


# In[14]:


sns.jointplot(x = data['Humidity'],y = data['Pressure'])


# # 3-v Correlation analysis

# In[37]:


data.drop('YRMODAHRMI',axis='columns',inplace=True)


# In[19]:


# "YRMODAHRMI" column was dropped because it is not intutive and no description is provided.


# In[29]:


data.corr()


# In[38]:


data_corr = data[['Location', 'Time', 'Latitude', 'Longitude', 'Altitude','Month', 'Hour', 'Season', 'Humidity', 'AmbientTemp',
              'Wind.Speed', 'Visibility', 'Pressure', 'Cloud.Ceiling', 'PolyPwr']].corr()


# In[39]:


mask = np.triu(np.ones_like(data_corr, dtype=bool))


# In[40]:


f, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(data_corr, mask=mask, cmap='jet', vmax=.3, center=0, annot=True, fmt='.2f',square=True, linewidths=.5, cbar_kws={"shrink": .8});
plt.title('Correlation analysis for all sites');


# In[65]:


sns.pairplot(data = data)


# # Average of PolyPwr with respective location

# In[ ]:


data1 =pd.pivot_table(data,index= 'Location',values = 'PolyPwr',aggfunc='mean')


# In[27]:


data1


# # Pie chart draw in respective columns

# In[17]:


plt=data.groupby(data['Location']).sum().plot(kind='pie',y='PolyPwr', subplots=True, shadow = True,startangle=90,
figsize=(15,10), autopct='%1.1f%%')


# In[20]:


plt=data.groupby(data['Season']).sum().plot(kind='pie',y='PolyPwr', subplots=True, shadow = True,startangle=90,
figsize=(15,10), autopct='%1.1f%%')


# # one hot encoding

# In[45]:


# Encode location data
df_with_location_en = pd.get_dummies(data, columns=['Location'], drop_first=True)
df_with_location_en


# In[44]:


# Encode season data
df_with_loc_season_en = pd.get_dummies(data, columns=['Season'], drop_first=True)
df_with_loc_season_en 


# In[50]:


# Define time bounds in data
min_hour_of_interest = 10
max_hour_of_interest = 15
# Calculate time lapse since onset of power generation
df_with_loc_season_en['delta_hr']= df_with_loc_season_en.Hour - min_hour_of_interest
df_with_loc_season_en['delta_hr']


# In[ ]:


mask2 = np.triu(np.ones_like(df_with_loc_season_en.corr(), dtype=bool))


# In[53]:


f, ax = plt.subplots(figsize=(30, 20))
sns.heatmap(df_with_loc_season_en.corr(method='spearman'), mask=mask2, cmap='jet', vmax=.3, center=0, annot=True, fmt='.2f',
            square=True, linewidths=.5, cbar_kws={"shrink": .8});
plt.title('Correlation analysis including encoded features for all sites');


# # Cyclic

# In[54]:


# Create cyclic month features
df_with_loc_season_en['sine_mon']= np.sin((df_with_loc_season_en.Month - 1)*np.pi/11)
df_with_loc_season_en['sine_mon']


# In[13]:


df_with_loc_season_en['cos_mon']= np.cos((df_with_loc_season_en.Month - 1)*np.pi/11)
df_with_loc_season_en['cos_mon']


# In[16]:


# Create cyclic hour features
df_with_loc_season_en['sine_hr']= np.sin((df_with_loc_season_en.delta_hr*np.pi/(max_hour_of_interest - min_hour_of_interest)))
df_with_loc_season_en['sine_hr']


# In[15]:


df_with_loc_season_en['cos_hr']= np.cos((df_with_loc_season_en.delta_hr*np.pi/(max_hour_of_interest - min_hour_of_interest)))
df_with_loc_season_en['cos_hr']


# # adding columns

# In[39]:


data[['delta_hr', 'sin_mon', 'cos_mon','sine_hr','cos_hr']] = pd.DataFrame([[df_with_loc_season_en['delta_hr'],df_with_loc_season_en['sine_mon'],df_with_loc_season_en['cos_mon'],df_with_loc_season_en['sine_hr'],df_with_loc_season_en['cos_hr']]], index=data.index)
data


# # Linear Regression

# In[66]:


from sklearn.model_selection import train_test_split
predictors=["Humidity"]
target=["PolyPwr"]
x=data[predictors]
y=data[target]


# In[67]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[68]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)
print(model.intercept_)


# In[78]:


model.coef_
rsq=model.score(x_test,y_test)
print(rsq)


# In[79]:


rsq1=model.score(x_train,y_train)
rsq1


# # KNN Regression

# In[118]:


predictors=["Humidity"]
target=["PolyPwr"]
X=data[predictors]
Y=data[target]


# In[119]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.2, random_state=0)


# In[120]:


from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors=3)


# In[121]:


knn_model.fit(X_train, y_train)


# In[122]:


from sklearn.metrics import mean_squared_error
from math import sqrt
# x_Train
train_preds = knn_model.predict(X_train)
mse = mean_squared_error(y_train, train_preds)
rmse = sqrt(mse)
rmse


# # Make model inferences on test set

# In[ ]:


test_preds = knn_model.predict(X_test)


# # Evaluate model performance on test set

# In[123]:


#RMSE
mse = mean_squared_error(y_test, test_preds)
rmse = sqrt(mse)
rmse


# In[124]:


#R-Squared
from sklearn.metrics import r2_score
r2 = r2_score(y_test,test_preds)
r2


# In[127]:


#MAE
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,test_preds)


# # Random Forest

# In[53]:


data['PolyPwr']=data['PolyPwr'].astype(int)


# In[54]:


from sklearn.model_selection import train_test_split

X=data[['Altitude']]  # Features
y=data['PolyPwr']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test


# In[55]:


from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


# In[56]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[77]:


import matplotlib.pyplot as plt
import seaborn as sns
# Creating a bar plot
# Model Accuracy, how often is the classifier correct?
sns.barplot(y_test, y_pred)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('polypower')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# # Make model inferences on test set

# In[140]:


test_preds = clf.predict(X_test)


# # Evaluate model performance on test set

# In[60]:


#RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
test_preds = clf.predict(X_test)
mse = mean_squared_error(y_test, test_preds)
rmse = sqrt(mse)
rmse


# In[61]:


#R-Squared
from sklearn.metrics import r2_score
r2 = r2_score(y_test,test_preds)
r2


# In[62]:


#MAE
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred)


# # Cross-validation

# In[33]:


# Predicting th input features , x=Predictotrs
x = np.array(data[['AmbientTemp','Humidity','Pressure','Cloud.Ceiling']])
x.shape


# In[34]:


# proving the output feature, y = regressand
y =np.array(data['PolyPwr'])
y.shape


# In[35]:


# splitting the dataset into train and test datasets
# importing the train_test_split method
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(x, y, test_size=0.30)
y_train.shape


# In[36]:


# importig the linear regerssion class
from sklearn.linear_model import LinearRegression
# Instantiation
model =  LinearRegression()
# Fitting the model
mymodel = model.fit(X_train,y_train)


# In[40]:


# k- fold cv
# Importing cross_val_score function
from sklearn.model_selection import cross_val_score
# 10-fold cv
score = cross_val_score(mymodel,X_train,y_train,scoring ='r2',cv = 10) # scoring ='r2' random cross validation
score


# In[42]:


#printing the average score
print(np.mean(score))


# In[45]:


# printing the score on the test dataset
# first:prediction
# importing cross_val_predict function
from sklearn.model_selection import cross_val_predict
# The predictions
pred = cross_val_predict(model,X_test,y_test)
pred


# In[49]:


# 10-fold cv on test data
score_test = cross_val_score(model,X_test,y_test,cv = 10)
score_test


# In[50]:


# The average score
print(np.mean(score_test))


# # stacking regressor

# In[71]:


from sklearn.ensemble import StackingRegressor


# In[101]:


# Define the base models
base0 = list()
base0.append(('Cross', model ))
base0.append(('knn_model', knn_model))


# In[102]:


# Define meta learner model
base1 = LinearRegression()


# In[103]:


# Define the stacking ensemble
stacked_model = StackingRegressor(estimators=base0, final_estimator=base1, cv=4, passthrough=True)


# In[132]:


get_ipython().run_cell_magic('time', '', '# Fit the model on the training data\nstacked_model.fit(X_test, y_test)')


# # Make Model Interface on test set

# In[133]:


test_preds = clf.predict(X_test)


# # Evaluate Model performance on test set

# In[136]:


# R2 score
r2_score(y_test,test_preds)


# In[137]:


# Mean absolute error
mean_absolute_error(y_test,test_preds)


# In[139]:


# Root mean square error
np.sqrt(mean_squared_error(y_test,test_preds))

