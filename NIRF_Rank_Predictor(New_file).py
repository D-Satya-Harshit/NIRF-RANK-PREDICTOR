#!/usr/bin/env python
# coding: utf-8

# # NIRF Rank Predictor
# *The goal of this machine learning problem is to build a predictive model that can accurately estimate the NIRF ranking of colleges and universities based on a set of relevant features and historical ranking data.
# By doing so, we aim to assist colleges and universities in assessing and enhancing their performance in various NIRF ranking parameters.*
# 
# It is carried out in following steps:-
# * Data Preprocessing and data cleaning
# * Transforming raw data into features that can be used to create predictive models
# * Exploratory Data Analysis 
# * Assessing various Machine Learning models
# * Training and Testing ML models
# * Finalizing the best model suited.

# As per the methodology of NIRF, there are five ranking parameters which are as follows:
# 
# * Teaching, learning, and resources
# * Research and professional practice
# * Graduation outcomes
# * Outreach and inclusivity
# * Peer perception
# 

# In[5]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge


# Data of year 2016, 2017, 2018, 2019, 2020, 2021 is arranged from kaggle.
# The dataset contains:-
# * Institute ID
# * Institute Name
# * City where it is located
# * State
# * Rank of various years
# * TLR (Teaching, Learning and Resources)
# * RPC (Research And Professional Practice)
# * GO (Graduation Outcomes)
# * OI (Outreach And Inclusivity)
# * Perception
# 
# The dataset is then combined into one dataset.
# 

# In[6]:


data_2016 = pd.read_csv('EngineeringRanking_2016.csv')


# In[7]:


data_2016.head()


# In[8]:


data_2017 = pd.read_csv('EngineeringRanking_2017.csv')


# In[9]:


data_2017.head()


# In[10]:


data_2018 = pd.read_csv('EngineeringRanking_2018.csv')


# In[11]:


data_2018.head()


# In[12]:


data_2019 = pd.read_csv('EngineeringRanking_2019.csv')


# In[13]:


data_2019.head()


# In[14]:


data_2020 = pd.read_csv('EngineeringRanking_2020.csv')


# In[15]:


data_2020.head()


# In[16]:


data_2021 = pd.read_csv('EngineeringRanking_2021.csv')


# In[17]:


data_2021.head()


# In[18]:


data_2016['year'],data_2017['year'],data_2018['year'],data_2019['year'],data_2020['year'],data_2021['year'] = '2016' , '2017' , '2018' , '2019' , '2020' , '2021'


# In[19]:


df = [data_2016,data_2017,data_2018,data_2019,data_2020,data_2021]
df_combined = pd.concat(df , axis =0 , ignore_index = 'True')


# In[20]:


# Combined dataset
df_combined.head()


# In[21]:


df_combined


# In[22]:


df_combined.info()


# 
# 
# *There are certain anomalies present in the data like in some fields Rank 21A , 26A is present. The letter is removed using lambda function and the datatype of rank is converted from string to float.*
# 
# 

# In[23]:


df_combined[219:228]


# In[24]:


df_combined['Rank'] = df_combined['Rank'].apply( lambda x: x if str(x).isdigit() else x[:-1])


# In[25]:


df_combined['Rank'] = df_combined['Rank'].astype('float64')


# In[26]:


df_combined[219:228]


# In[27]:


df_combined.info()


# *The Institute name in 2016 is different from other year datasets so a operation is carried out to make it same.*

# In[28]:


data_2016.head(1)


# In[29]:


data_2017.head(1)


# In[30]:


data_2016['Institute Name'] = data_2016['Institute Name'].str.replace(',', '')


# In[31]:


data_2016.head(1)


# # EXPLORATORY DATA ANALYSIS

# In[32]:


sns.pairplot(df_combined)


#         Mean RPC is low implies research sector is weak in colleges..

# In[33]:


df_combined.describe()


# In[34]:


sns.distplot(df_combined['Rank'])


# In[35]:


df_combined.columns


# #Feature Extractions

# In[36]:


df_parameters = df_combined.drop(columns = ['Institute Id', 'Institute Name', 'City', 'State' , 'year', 'Rank'])


# In[37]:


df_parameters.head()


# In[38]:


corrmat = df_parameters.corr()
ax = sns.heatmap(corrmat, annot = True , cmap = 'Reds')
plt.figure(figsize=(15,15))
plt.show()


# *Here correlation of rpc with score is 0.94 it implies it greatly affects the overall score.*

# # Linear Regressor

# In[39]:


X = df_combined[['TLR' , 'RPC' , 'GO' , 'OI' , 'Perception']]
y = df_combined['Score']


# In[40]:


from sklearn.model_selection import train_test_split


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# In[42]:


from sklearn.linear_model import LinearRegression


# In[43]:


lm = LinearRegression()


# In[44]:


lm.fit(X_train,y_train)


# In[45]:


# print the intercept
print(lm.intercept_)


# In[46]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[47]:


predictions = lm.predict(X_test)


# In[48]:


plt.scatter(y_test,predictions)


# # Evaluating the Model

# In[49]:


sns.distplot((y_test-predictions),bins=50);


# In[50]:


from sklearn import metrics


# In[51]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# # R Square Value

# In[54]:


lm.score(X_test,y_test)


# # Ajusted R Square
# 
# $R^2_{adj.} = 1 - (1-R^2)*\frac{n-1}{n-p-1}$
# 
# n = Number of observations and 
# P = Features 

# #### So the shape the Test Data set is n = 270 , P=5

# In[56]:


X_test.shape


# In[65]:


r2 = lm.score(X_test,y_test)
n = X_test.shape[0]
p = X_test.shape[1]

adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
adjusted_r2


# In[53]:


ridge_regressor= Ridge()
parameters = {'alpha' : [10,20,30,40,50,60,70,80,90]}
ridgecv = GridSearchCV(ridge_regressor,parameters,scoring='neg_mean_squared_error')
ridgecv.fit(X_train,y_train)


# In[54]:


print(ridgecv.best_params_)


# In[56]:


predicitions_R2 = ridgecv.predict(X_test)
sns.displot(predicitions_R2-y_test,kind='kde')


# In[57]:


from sklearn.metrics import r2_score
r2_score_ridge = r2_score(predicitions_R2,y_test)
r2_score_ridge


# ### Adjusted R Square

# In[58]:


n = X_test.shape[0]
p = X_test.shape[1]

adjusted_r2 = 1-(1-r2_score_ridge)*(n-1)/(n-p-1)
adjusted_r2


# # Decision Tree Regressor

# In[79]:


from sklearn.tree import DecisionTreeRegressor


# In[80]:


dtree = DecisionTreeRegressor()


# In[81]:


dtree.fit(X_train,y_train)


# In[54]:


sns.distplot((y_test-pred1),bins=50);


# In[55]:


pred1 = dtree.predict(X_test)


# In[56]:


print('MAE:', metrics.mean_absolute_error(y_test, pred1))
print('MSE:', metrics.mean_squared_error(y_test, pred1))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred1)))


# # Random Forest Regressor

# In[82]:


from sklearn.ensemble import RandomForestRegressor
rfc = RandomForestRegressor(n_estimators=100)
rfc.fit(X_train, y_train)


# In[58]:


rfc_pred = rfc.predict(X_test)


# In[59]:


sns.distplot((y_test-pred1),bins=50);


# In[60]:


print('MAE:', metrics.mean_absolute_error(y_test, rfc_pred))
print('MSE:', metrics.mean_squared_error(y_test, rfc_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, rfc_pred)))


# # KNN Regressor
# 

# In[84]:


from sklearn.neighbors import KNeighborsRegressor


# Elbow Method 

# In[85]:


error_rate = []

for i in range(1,40):
    
    knn = KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    exp_i = list(y_test)
    error_rate.append((np.square(np.subtract(pred_i, exp_i))).mean())


# In[86]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[88]:


knn_model= KNeighborsRegressor(n_neighbors=11)

knn_model.fit(X_train,y_train)
pred_new_KNN = knn_model.predict(X_test)

print('WITH K=11')
print('\n')
print('MAE:', metrics.mean_absolute_error(y_test, pred_new_KNN))
print('MSE:', metrics.mean_squared_error(y_test, pred_new_KNN))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred_new_KNN)))


# In[65]:


df_score_rank = df_combined[['Score' , 'Rank']]
df_rel_score_rank = df_score_rank.groupby('Rank').mean()
df_rel_score_rank.plot(kind = 'line' , ylabel = 'Score')


# *Here it is shown the relation between Overall Score and the Rank of the institute.*

# # Linear Regression Accuracy

# In[78]:


arr_test = y_test.values


# In[117]:


(arr_test)


# In[79]:


sum_test =0


# In[80]:


for i in arr_test:
    sum_test = sum_test + i


# In[81]:


sum_test


# In[82]:


predictions = lm.predict(X_test)


# In[96]:


len(predictions)


# In[84]:


sum_predictions =0


# In[97]:


for i in predictions:
    sum_predictions = sum_predictions + i


# In[98]:


sum_predictions


# In[101]:


bias= (sum_test - sum_predictions) / (sum_test)


# In[102]:


bias


# In[103]:


100 -(-1 *bias)


# In[105]:


df_combined.info()


# In[138]:


new_predicitions = ((abs(y_test - predictions))/y_test)   


# In[83]:


predictions


# In[139]:


new_predicitions


# In[140]:


new_predicitions =new_predicitions*100


# In[141]:


new_predicitions


# In[112]:


sum_linear_regression=0


# In[114]:


for i in new_predicitions:
    sum_linear_regression = sum_linear_regression + i


# # Mean Absolute Percentage Error Linear Regression

# In[132]:


mape = (sum_linear_regression/270)


# In[155]:


accuracy_linear_regression = 100-mape
accuracy_linear_regression


# # Decision Tree Regressor Accuracy

# In[160]:


predicitions_Decision_tree = dtree.predict(X_test)


# In[161]:


new_predicitions_decision_tree = ((abs(y_test - predicitions_Decision_tree))/y_test)*100


# In[165]:


sum_decision_tree = 0
for i in new_predicitions_decision_tree:
    sum_decision_tree = sum_decision_tree + i
mape_decision_tree = sum_decision_tree/270
mape_decision_tree


# In[167]:


accuracy_decision_tree = 100-mape_decision_tree
accuracy_decision_tree


# # Random Forest Accuracy

# In[168]:


rfc_pred = rfc.predict(X_test)


# In[169]:


new_predicitions_random_forest = ((abs(y_test -rfc_pred))/y_test)*100


# In[170]:


sum_random_forest = 0
for i in new_predicitions_random_forest:
    sum_random_forest = sum_random_forest + i
mape_random_forest = sum_random_forest/270
mape_random_forest


# In[172]:


accuracy_random_forest = 100-mape_random_forest
accuracy_random_forest


# # KNN Accuracy

# knn_model= KNeighborsRegressor(n_neighbors=11)
# 
# knn_model.fit(X_train,y_train)
# pred_new_KNN = knn_model.predict(X_test)
# 
# print('WITH K=11')
# print('\n')
# print('MAE:', metrics.mean_absolute_error(y_test, pred_new_KNN))
# print('MSE:', metrics.mean_squared_error(y_test, pred_new_KNN))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred_new_KNN)))

# In[196]:


Predicitons_KNN = knn_model.predict(X_test)


# In[197]:


new_predicitions_KNN = ((abs(y_test -Predicitons_KNN))/y_test)*100


# In[198]:


sum_KNN= 0
for i in new_predicitions_KNN:
    sum_KNN = sum_KNN + i
mape_KNN = sum_KNN/270
mape_KNN


# In[199]:


accuracy_KNN = 100-mape_KNN
accuracy_KNN


# # Overall Accuracy of the models

# In[201]:


print("The Accuracy of the models are ")
print("Linear_Regression:", accuracy_linear_regression)
print("Decision Tree:", accuracy_decision_tree)
print("Random Forest:", accuracy_random_forest )
print("KNN:",accuracy_KNN )

