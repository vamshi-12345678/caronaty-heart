#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd # linear algebra
import numpy as np # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns # statistical data visualization
import matplotlib.pyplot as plt # data visualization


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[177]:


data=pd.read_csv("sample_submission.csv")
data.head()


# In[3]:


#preview the dataset
data1=pd.read_csv("train_data1.csv")
data1


# In[4]:


data1.head()


# In[5]:


data1.tail()


# In[6]:


data1['is_fraud'].isnull().sum()


# In[7]:


data1.shape


# In[8]:


data1.describe()


# In[9]:


data1.info()


# In[10]:


missing_vals=data1.isnull().sum()
missing_vals


# In[11]:


missing_vals=data1.isnull().sum().sum()
missing_vals


# In[12]:


# find categorical variables
categorical = [var for var in data1.columns if data1[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)


# In[13]:


# view the categorical variables

data1[categorical].head()


# In[14]:


# view frequency of categorical variables

for var in categorical: 
    
    print(data1[var].value_counts())


# In[15]:


# view frequency distribution of categorical variables

for var in categorical: 
    
    print(data1[var].value_counts()/np.float(len(data1)))


# In[16]:


# check for cardinality in categorical variables

for var in categorical:
    
    print(var, ' contains ', len(data1[var].unique()), ' labels')


# # Feature Engineering of Date Variable

# In[17]:


data1['transaction_initiation'].dtypes


# In[18]:


# parse the dates, currently coded as strings, into datetime format

data1['transaction_initiation'] = pd.to_datetime(data1['transaction_initiation'])


# In[19]:


# extract year from date

data1['year'] = data1['transaction_initiation'].dt.year

data1['year'].head()


# In[20]:


# extract month from date

data1['month'] = data1['transaction_initiation'].dt.month

data1['month'].head()


# In[21]:


# extract day from date

data1['day'] = data1['transaction_initiation'].dt.day

data1['day'].head()


# In[22]:


#check the columns of dataset
col_names=data1.columns
col_names


# In[23]:


# again view the summary of dataset

data1.info()


# In[24]:


# drop the original transaction_initiation variable

data1.drop('transaction_initiation', axis=1, inplace = True)


# In[25]:


# preview the dataset again

data1.head()


# ### Explore Categorical Variables
# 
# 
# Now, I will explore the categorical variables one by one. 

# In[26]:


# find categorical variables

categorical = [var for var in data1.columns if data1[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)


# We can see that there are 4 categorical variables in the dataset. The `transaction_initiation` variable has been removed. First, I will check missing values in categorical variables.

# In[27]:


# check for missing values in categorical variables 

data1[categorical].isnull().sum()


# we can see that there is no missing values.i will explore these variables one by one

# # Explore 'payment_method' variable

# In[28]:


# print number of labels in payment_method variable

print('payment_method contains', len(data1.payment_method.unique()), 'labels')


# In[29]:


# check labels in payment_method variable

data1.payment_method.unique()


# In[30]:


# check frequency distribution of values in payment_method variable

data1.payment_method.value_counts()


# In[31]:


# let's do One Hot Encoding of payment_method variable
# get k-1 dummy variables after One Hot Encoding 
# preview the dataset with head() method

pd.get_dummies(data1.payment_method, drop_first=True).head()


# # Explore 'partner_category' variable

# In[32]:


# print number of labels in partner_category variable


print('partner_category contains', len(data1.partner_category.unique()), 'labels')


# In[33]:


# check labels in partner_category variable

data1.partner_category.unique()


# In[34]:


# check frequency distribution of values in partner_category variable

data1.partner_category.value_counts()


# In[35]:


# let's do One Hot Encoding of partner_category variable
# get k-1 dummy variables after One Hot Encoding 
# preview the dataset with head() method

pd.get_dummies(data1.partner_category, drop_first=True).head()


# # Explore 'country' variable

# In[36]:


# print number of labels in country variable


print('country contains', len(data1.country.unique()), 'labels')


# In[37]:


# check labels in country variable

data1.country.unique()


# In[38]:


# check frequency distribution of values in country variable

data1.country.value_counts()


# In[39]:


# let's do One Hot Encoding of country variable
# get k-1 dummy variables after One Hot Encoding 
# preview the dataset with head() method

pd.get_dummies(data1.country, drop_first=True).head()


# # Explore 'device_type' variable

# In[40]:


# print number of labels in device_type variable


print('device_type contains', len(data1.device_type.unique()), 'labels')


# In[41]:


# check labels in device_type variable

data1.device_type.unique()


# In[42]:


# check frequency distribution of values in device_type variable

data1.device_type.value_counts()


# In[43]:


# let's do One Hot Encoding of device_type variable
# get k-1 dummy variables after One Hot Encoding 
# preview the dataset with head() method

pd.get_dummies(data1.device_type, drop_first=True).head()


# # Explore Numerical Variables

# In[44]:


# find numerical variables

numerical = [var for var in data1.columns if data1[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)


# In[45]:


# view the numerical variables

data1[numerical].head()


# ## Explore problems within numerical variables
# 
# 
# Now, I will explore the numerical variables.
# 
# 
# ### Missing values in numerical variables

# In[46]:


# check missing values in numerical variables

data1[numerical].isnull().sum()


# In[47]:


len(data1[numerical].isnull().sum())


# # Outliers in numerical variables

# In[48]:


# view summary statistics in numerical variables
data1[numerical].describe().T


# In[49]:


# draw boxplots to visualize outliers

plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = data1.boxplot(column='transaction_number')
fig.set_title('')
fig.set_ylabel('transaction_number')


plt.subplot(2, 2, 2)
fig = data1.boxplot(column='user_id')
fig.set_title('')
fig.set_ylabel('user_id')


plt.subplot(2, 2, 3)
fig = data1.boxplot(column='partner_id')
fig.set_title('')
fig.set_ylabel('partner_id')


plt.subplot(2, 2, 4)
fig = data1.boxplot(column='money_transacted')
fig.set_title('')
fig.set_ylabel('money_transacted')


# In[50]:


plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
fig = data1.boxplot(column='partner_pricing_category')
fig.set_title('')
fig.set_ylabel('partner_pricing_category')



# # Check the distribution of variables
# Now, I will plot the histograms to check distributions to find out if they are normal or skewed. If the variable follows normal distribution, then I will do Extreme Value Analysis otherwise if they are skewed, I will find IQR (Interquantile range).

# In[51]:


# plot histogram to check distribution

plt.figure(figsize=(15,10))


plt.subplot(2, 3, 1)
fig = data1.transaction_number.hist(bins=10)
fig.set_xlabel('transaction_number')
fig.set_ylabel('is_fraud')


plt.subplot(2, 3, 2)
fig = data1.user_id.hist(bins=10)
fig.set_xlabel('user_id')
fig.set_ylabel('is_fraud')


plt.subplot(2, 3, 3)
fig = data1.partner_id.hist(bins=10)
fig.set_xlabel('partner_id')
fig.set_ylabel('is_fraud')


plt.subplot(2, 3, 4)
fig = data1.money_transacted.hist(bins=10)
fig.set_xlabel('money_transacted')
fig.set_ylabel('is_fraud')

plt.subplot(2, 3, 5)
fig = data1.partner_pricing_category.hist(bins=10)
fig.set_xlabel('partner_pricing_category')
fig.set_ylabel('is_fraud')


# In[52]:


# find outliers for transaction_number variable

IQR = data1.transaction_number.quantile(0.75) - data1.transaction_number.quantile(0.25)
Lower_fence = data1.transaction_number.quantile(0.25) - (IQR * 3)
Upper_fence = data1.transaction_number.quantile(0.75) + (IQR * 3)
print('transaction_number outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))


# In[53]:


print('transaction_number min and max values are : {min_value} , {max_value}'.format(min_value=data1.transaction_number.min(), max_value=data1.transaction_number.max()))


# In[54]:


# find outliers for user_id variable

IQR = data1.user_id.quantile(0.75) - data1.user_id.quantile(0.25)
Lower_fence = data1.user_id.quantile(0.25) - (IQR * 3)
Upper_fence = data1.user_id.quantile(0.75) + (IQR * 3)
print('user_id outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))


# In[55]:


print('user_id min and max values are : {min_value} , {max_value}'.format(min_value=data1.user_id.min(), max_value=data1.user_id.max()))


# In[56]:


# find outliers for partner_id variable

IQR = data1.partner_id.quantile(0.75) - data1.partner_id.quantile(0.25)
Lower_fence = data1.partner_id.quantile(0.25) - (IQR * 3)
Upper_fence = data1.partner_id.quantile(0.75) + (IQR * 3)
print('partner_id outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))


# In[57]:


print('partner_id min and max values are : {min_value} , {max_value}'.format(min_value=data1.partner_id.min(), max_value=data1.partner_id.max()))


# In[58]:


# find outliers for money_transacted variable

IQR = data1.money_transacted.quantile(0.75) - data1.money_transacted.quantile(0.25)
Lower_fence = data1.money_transacted.quantile(0.25) - (IQR * 3)
Upper_fence = data1.money_transacted.quantile(0.75) + (IQR * 3)
print('money_transacted outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))


# In[59]:


print('money_transacted min and max values are : {min_value} , {max_value}'.format(min_value=data1.money_transacted.min(), max_value=data1.money_transacted.max()))


# In[60]:


#preview the dataset
data2=pd.read_csv("test_data1.csv")
data2


# In[61]:


data2.head()


# In[62]:


data2.tail()


# In[63]:


data2.shape


# In[64]:


data2.describe()


# In[65]:


data2.info()


# In[66]:


# find categorical variables

categorical = [var for var in data2.columns if data2[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)


# In[67]:


# view the categorical variables

data2[categorical].head()


# ## Explore problems within categorical variables
# 
# 
# First, we will explore the categorical variables.
# 
# 
# ### Missing values in categorical variables

# In[68]:


# check missing values in categorical variables

data2[categorical].isnull().sum()


# In[69]:


# view the categorical variables

data2[categorical].head()


# In[70]:


# view frequency of categorical variables

for var in categorical: 
    
    print(data2[var].value_counts())


# # Feature Engineering of Date Variable¶
# 
# 

# In[71]:


data2['transaction_initiation'].dtypes


# In[72]:


# parse the dates, currently coded as strings, into datetime format

data2['transaction_initiation'] = pd.to_datetime(data2['transaction_initiation'])
data2['transaction_initiation']


# In[73]:


# extract year from date

data2['year'] = data2['transaction_initiation'].dt.year

data2['year'].head()


# In[74]:


# extract month from date

data2['month'] = data2['transaction_initiation'].dt.month

data2['month'].head()


# In[75]:


# extract month from date

data2['day'] = data2['transaction_initiation'].dt.day

data2['day'].head()


# In[76]:


# again view the summary of dataset

data2.info()


# In[77]:


missing_vals=data2.isnull().sum()
missing_vals


# In[78]:


# drop the original transaction_initiation variable

data2.drop('transaction_initiation', axis=1, inplace = True)


# In[79]:


# preview the dataset again

data2.head()


# # Explore Categorical Variables
# Now, I will explore the categorical variables one by one.

# In[80]:


# find categorical variables

categorical = [var for var in data2.columns if data2[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :', categorical)


# In[81]:


# check for missing values in categorical variables 

data2[categorical].isnull().sum()


# # Explore 'payment_method' variable¶

# In[82]:


# print number of labels in partner_category variable


print('partner_category contains', len(data2.partner_category.unique()), 'labels')


# In[83]:


# check labels in payment_method variable

data2.payment_method.unique()


# In[84]:


# check frequency distribution of values in payment_method variable

data2.payment_method.value_counts()


# In[85]:


# let's do One Hot Encoding of payment_method variable
# get k-1 dummy variables after One Hot Encoding 
# preview the dataset with head() method

pd.get_dummies(data2.payment_method, drop_first=True).head()


# # Explore 'partner_category' variable

# In[86]:


# print number of labels in partner_category variable


print('partner_category contains', len(data2.partner_category.unique()), 'labels')


# In[87]:


# check labels in partner_category variable
data2.partner_category.unique()


# In[88]:


# check frequency distribution of values in partner_category variable

data2.partner_category.value_counts()


# In[89]:


# let's do One Hot Encoding of partner_category variable
# get k-1 dummy variables after One Hot Encoding 
# preview the dataset with head() method

pd.get_dummies(data2.partner_category, drop_first=True).head()


# # Explore 'country' variable

# In[90]:


# print number of labels in country variable


print('country contains', len(data2.country.unique()), 'labels')


# In[91]:


# check labels in country variable

data2.country.unique()


# In[92]:


# check frequency distribution of values in country variable

data2.country.value_counts()


# In[93]:


# let's do One Hot Encoding of country variable
# get k-1 dummy variables after One Hot Encoding 
# preview the dataset with head() method

pd.get_dummies(data2.country, drop_first=True).head()


# # Explore 'device_type' variable

# In[94]:


# print number of labels in device_type variable


print('device_type contains', len(data2.device_type.unique()), 'labels')


# In[95]:


# check labels in device_type variable

data2.device_type.unique()


# In[96]:


# check frequency distribution of values in device_type variable

data2.device_type.value_counts()


# In[97]:


# let's do One Hot Encoding of device_type variable
# get k-1 dummy variables after One Hot Encoding 
# preview the dataset with head() method

pd.get_dummies(data2.device_type, drop_first=True).head()


# # Explore Numerical Variables

# In[98]:


# find numerical variables

numerical = [var for var in data2.columns if data2[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)


# In[99]:


# view the numerical variables

data2[numerical].head()


# # Explore problems within numerical variables
# Now, I will explore the numerical variables.
# 
# Missing values in numerical variables¶

# In[100]:


# check missing values in numerical variables

data2[numerical].isnull().sum()


# In[101]:


len(data2[numerical].isnull().sum())


# # Outliers in numerical variables

# In[102]:


# view summary statistics in numerical variables
data2[numerical].describe().T


# In[103]:


# draw boxplots to visualize outliers

plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = data2.boxplot(column='transaction_number')
fig.set_title('')
fig.set_ylabel('transaction_number')


plt.subplot(2, 2, 2)
fig = data2.boxplot(column='user_id')
fig.set_title('')
fig.set_ylabel('user_id')


plt.subplot(2, 2, 3)
fig = data2.boxplot(column='partner_id')
fig.set_title('')
fig.set_ylabel('partner_id')


plt.subplot(2, 2, 4)
fig = data2.boxplot(column='money_transacted')
fig.set_title('')
fig.set_ylabel('money_transacted')


# In[104]:


plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
fig = data2.boxplot(column='partner_pricing_category')
fig.set_title('')
fig.set_ylabel('partner_pricing_category')


# # Check the distribution of variables¶
# Now, I will plot the histograms to check distributions to find out if they are normal or skewed. If the variable follows normal distribution, then I will do Extreme Value Analysis otherwise if they are skewed, I will find IQR (Interquantile range).

# In[105]:


# plot histogram to check distribution

plt.figure(figsize=(15,10))


plt.subplot(2, 3, 1)
fig = data2.transaction_number.hist(bins=10)
fig.set_xlabel('transaction_number')
fig.set_ylabel('is_fraud')


plt.subplot(2, 3, 2)
fig = data2.user_id.hist(bins=10)
fig.set_xlabel('user_id')
fig.set_ylabel('is_fraud')


plt.subplot(2, 3, 3)
fig = data2.partner_id.hist(bins=10)
fig.set_xlabel('partner_id')
fig.set_ylabel('is_fraud')


plt.subplot(2, 3, 4)
fig = data2.money_transacted.hist(bins=10)
fig.set_xlabel('money_transacted')
fig.set_ylabel('is_fraud')

plt.subplot(2, 3, 5)
fig = data2.partner_pricing_category.hist(bins=10)
fig.set_xlabel('partner_pricing_category')
fig.set_ylabel('is_fraud')


# In[106]:


# find outliers for transaction_number variable

IQR = data2.transaction_number.quantile(0.75) - data2.transaction_number.quantile(0.25)
Lower_fence = data2.transaction_number.quantile(0.25) - (IQR * 3)
Upper_fence = data2.transaction_number.quantile(0.75) + (IQR * 3)
print('transaction_number outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))


# In[107]:


print('transaction_number min and max values are : {min_value} , {max_value}'.format(min_value=data2.transaction_number.min(), max_value=data2.transaction_number.max()))


# In[108]:


# find outliers for user_id variable

IQR = data2.user_id.quantile(0.75) - data2.user_id.quantile(0.25)
Lower_fence = data2.user_id.quantile(0.25) - (IQR * 3)
Upper_fence = data2.user_id.quantile(0.75) + (IQR * 3)
print('user_id outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))


# In[109]:


print('user_id min and max values are : {min_value} , {max_value}'.format(min_value=data2.user_id.min(), max_value=data2.user_id.max()))


# In[110]:


# find outliers for partner_id variable

IQR = data2.partner_id.quantile(0.75) - data2.partner_id.quantile(0.25)
Lower_fence = data2.partner_id.quantile(0.25) - (IQR * 3)
Upper_fence = data2.partner_id.quantile(0.75) + (IQR * 3)
print('partner_id outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))


# In[111]:


print('partner_id min and max values are : {min_value} , {max_value}'.format(min_value=data2.partner_id.min(), max_value=data2.partner_id.max()))


# In[112]:


# find outliers for money_transacted variable

IQR = data2.money_transacted.quantile(0.75) - data2.money_transacted.quantile(0.25)
Lower_fence = data2.money_transacted.quantile(0.25) - (IQR * 3)
Upper_fence = data2.money_transacted.quantile(0.75) + (IQR * 3)
print('money_transacted outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))


# In[113]:


print('money_transacted min and max values are : {min_value} , {max_value}'.format(min_value=data2.money_transacted.min(), max_value=data2.money_transacted.max()))


# # Declare feature vector and target variable

# In[114]:


import sklearn
new_features=data1[['transaction_number', 'user_id', 'payment_method', 'partner_id',
       'partner_category', 'country', 'device_type', 'money_transacted',
       'partner_pricing_category', 'year', 'month', 'day','is_fraud']]
x_train=new_features.iloc[:,:-1]
y_train=new_features.iloc[:,-1]


# In[115]:


x_train


# In[116]:


y_train


# In[117]:


# display categorical variables

categorical = [col for col in x_train.columns if x_train[col].dtypes == 'O']

categorical


# In[118]:


# display numerical variables

numerical = [col for col in x_train.columns if x_train[col].dtypes != 'O']

numerical


# In[119]:


import sklearn
new_features=data2[['transaction_number', 'user_id', 'payment_method', 'partner_id',
      'partner_category', 'country', 'device_type', 'money_transacted',
       'partner_pricing_category', 'year', 'month', 'day']]
x_test=new_features.iloc[:,:]


# In[120]:


x_test


# # Encode categorical variable

# In[121]:


categorical


# In[122]:


x_train[categorical].head()


# In[123]:


x_train


# In[124]:


x_train = pd.concat([x_train[numerical],
                     pd.get_dummies(x_train.payment_method), 
                     pd.get_dummies(x_train.partner_category),
                     pd.get_dummies(x_train.device_type),], axis=1)


# In[125]:


x_train.head()


# In[126]:


x_test = pd.concat([x_test[numerical],
                     pd.get_dummies(x_test.payment_method), 
                     pd.get_dummies(x_test.partner_category),
                     pd.get_dummies(x_test.device_type),], axis=1)


# In[127]:


x_test


# In[128]:


x_train.drop('cat_9', axis=1, inplace = True)


# # Feature Scaling

# In[129]:


x_train.describe().T


# In[130]:


x_test.describe().T


# # Model training

# In[131]:


y_train.value_counts()


# In[132]:


# shows counts of observations
sns.countplot(x='is_fraud',data=data1)


# In[133]:


# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression


# instantiate the model
logreg = LogisticRegression(solver='liblinear', random_state=0)


# fit the model
logreg.fit(x_train, y_train)


# In[134]:


variables = ['*'.join(tup).rstrip('*') for tup in x_train.columns.values]
coef = pd.concat([pd.Series(variables), pd.Series(logreg.coef_.reshape(-1))], axis=1)
coef.columns=["Variable", "coefficient"]
coef


# In[135]:


model = LogisticRegression().fit(x_train, y_train)
model


# # Predict results

# In[136]:


y_pred_test = logreg.predict(x_test)

y_pred_test


# In[137]:


y_test = model.predict(x_test)  # create y-labels through the learned model
print(y_test)


# In[138]:


# probability of getting output as 0 - no fraud

logreg.predict_proba(x_test)[:,0]


# In[139]:


y_test = model.predict(x_test)  # create y-labels through the learned model
print(y_test)


# In[140]:


# probability of getting output as 1 - fraud

logreg.predict_proba(x_test)[:,1]


# # Check accuracy score

# In[141]:


from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))


# # Compare the train set and test set accuracy

# In[142]:


y_pred_train = logreg.predict(x_train)

y_pred_train


# In[143]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


# # CONFUSION MATRIX

# In[144]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred_test)
cm=confusion_matrix(y_test,y_pred_test)
print ("Confusion Matrix : \n", cm)


# # F1_SCORE CALCULATE 

# In[145]:


from sklearn.metrics import classification_report

print(classification_report(y_train, y_pred_train))


# In[146]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred_test)
cm=confusion_matrix(y_test,y_pred_test)
print ("Confusion Matrix : \n", cm)


# In[181]:


pd.DataFrame(confusion_matrix(y_pred_train, y_train),columns=['Predict-NO','Predict-YES'],index=['NO','YES'])


# In[ ]:





# In[148]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_train,y_pred_train)
conf_matrix=pd.DataFrame(data=cm,columns=['Actual:1','Actual:0'],index=['predicted:NO','predicted:YES'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")


# # CLASSIFICATION ACCURACY

# In[149]:


TN=cm[0,0]
TP=cm[1,1]
FN=cm[1,0]
FP=cm[0,1]
sensitivity=TP/float(TP+FN)
specificity=TN/float(TN+FP)


# In[150]:


sensitivity


# In[151]:


specificity


# # CLASSIFICATION REPORT

# In[189]:


from sklearn.metrics import classification_report

print(classification_report(y_train, y_pred_train))


# # CLASSIFICATION ERROR

# In[153]:


# print classification error

classification_error = (FP + FN) / float(TP + TN + FP + FN)

print('Classification error : {0:0.4f}'.format(classification_error))


# # MODEL EVALUATION

# In[154]:


print('The acuuracy of the model = TP+TN/(TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n',

'The Missclassification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',

'Sensitivity or True Positive Rate = TP/(TP+FN) = ',TP/float(TP+FN),'\n',

'Specificity or True Negative Rate = TN/(TN+FP) = ',TN/float(TN+FP),'\n',

'Positive Predictive value = TP/(TP+FP) = ',TP/float(TP+FP),'\n',

'Negative predictive Value = TN/(TN+FN) = ',TN/float(TN+FN),'\n',

'Positive Likelihood Ratio = Sensitivity/(1-Specificity) = ',sensitivity/(1-specificity),'\n',

'Negative likelihood Ratio = (1-Sensitivity)/Specificity = ',(1-sensitivity)/specificity)


# In[194]:


y_pred_prob=logreg.predict_proba(x_test)[:,:]
y_pred_prob_data1=pd.DataFrame(data=y_pred_prob, columns=['Prob of fraud  (NO)','Prob of fraud (YES)'])
y_pred_prob_data1['Decision']=(y_pred_prob_data1['Prob of fraud (YES)']>prob_threshold).apply(int)
y_pred_prob_data1.head(1500)


# In[195]:


y_pred_prob=logreg.predict_proba(x_train)[:,:]
y_pred_prob_data1=pd.DataFrame(data=y_pred_prob, columns=['Prob of fraud  (NO)','Prob of fraud (YES)'])
y_pred_prob_data1['Decision']=(y_pred_prob_data1['Prob of fraud (YES)']>prob_threshold).apply(int)
y_pred_prob_data1.head(1500)


# In[164]:


print('Positive Likelihood Ratio = Sensitivity/(1-Specificity) = ',sensitivity/(1-specificity),'\n',

'Negative likelihood Ratio = (1-Sensitivity)/Specificity = ',(1-sensitivity)/specificity)


# In[184]:


# print the first 10 predicted probabilities of two classes- 0 and 1

y_pred_test_prob = logreg.predict_proba(x_test)[0:10]

y_pred_test_prob


# # store the predicted probabilities for class 1 - Probability of fraud

# In[185]:


y_pred_test = logreg.predict_proba(x_test)[:, 1]
y_pred_test


# # PREDICTED PROBABILITIES OF fraud

# In[187]:


# plot histogram of predicted probabilities


# adjust the font size 
plt.rcParams['font.size'] = 12


# plot histogram with 10 bins
plt.hist(y_pred_train, bins = 10)


# set the title of predicted probabilities
plt.title('Histogram of predicted is_fraud')


# set the x-axis limit
plt.xlim(0,1)


# set the title
plt.xlabel('Predicted probabilities of fraud')
plt.ylabel('Frequency')


# # COMMENTS
# In binary problems, the threshold of 0.5 is used by default to convert predicted probabilities into class predictions.
# 
# Threshold can be adjusted to increase sensitivity or specificity.
# 
# Sensitivity and specificity have an inverse relationship. Increasing one would always decrease the other and vice versa.
# 
# We can see that increasing the threshold level results in increased accuracy.
# 
# Adjusting the threshold level should be one of the last step you do in the model-building process

# # K-FOLD CROSS VALIDATION

# In[196]:


# calculate cross-validated ROC AUC 

from sklearn.model_selection import cross_val_score

Cross_validated_ROC_AUC = cross_val_score(logreg, x_test, y_test, cv=5, scoring='roc_auc').mean()

print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))


# In[197]:


## k-Fold Cross Validation
# Applying 5-Fold Cross Validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(logreg, x_test, y_test, cv = 5, scoring='accuracy')

print('Cross-validation scores:{}'.format(scores))

# compute Average cross-validation score

print('Average cross-validation score: {:.4f}'.format(scores.mean()))
# Our, original model score is found to be 0.8295. The average cross-validation score is 0.. So, we can conclude that cross-validation does not result in performance improvement.

# # COMMENTS
# Our original model test accuracy is 1.000.

# # RESULTS AND CONCLUSION
# The logistic regression model accuracy score is 0.8298. So, the model does a very good job in predicting fraud or not
# 
# Small number of observations predict that there will be fraud. Majority of observations predict that there will be no fraud.
# 
# The model shows no signs of overfitting.
# 
# Increasing the value of C results in higher test set accuracy and also a slightly increased training set accuracy. So, we can conclude that a more complex model should perform better.
# 
# 
# Our original model accuracy score is 1.000 
# In the oiginal model, we have TP = 76375 whereas FN =0 , FP = 154 whereas TN = 0.
# 
# 

# # REFERENCES
# The work done in this project is inspired from following books and websites:-
# 
# Hands on Machine Learning with Scikit-Learn and Tensorflow by Aurélién Géron
# 
# Introduction to Machine Learning with Python by Andreas C. Müller and Sarah Guido
# 
# Udemy course – Machine Learning – A Z by Kirill Eremenko and Hadelin de Ponteves
# 
# Udemy course – Feature Engineering for Machine Learning by Soledad Galli
# 
# Udemy course – Feature Selection for Machine Learning by Soledad Galli
# 
# https://en.wikipedia.org/wiki/Logistic_regression
# 
# https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html
# 
# https://en.wikipedia.org/wiki/Sigmoid_function
# 
# https://www.statisticssolutions.com/assumptions-of-logistic-regression/
# 
# https://www.kaggle.com/mnassrib/titanic-logistic-regression-with-python
# 
# https://www.kaggle.com/neisha/heart-disease-prediction-using-logistic-regression
# 
# https://www.ritchieng.com/machine-learning-evaluate-classification-model/
# 
# So, now we will come to the end of this session.
# 
# I hope you find session is useful and enjoyable.
# 
# Your comments and feedback are most welcome.
# 
# Thank you
# 
# Go to Top

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




