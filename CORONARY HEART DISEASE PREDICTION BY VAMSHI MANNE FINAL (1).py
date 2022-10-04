#!/usr/bin/env python
# coding: utf-8

# # HEART DISEASE PREDICTION USING LOGISTIC REGRESSION 

# A healthcare organization together with a couple of government hospitals in a city has collectedinformation about the vitals that would reveal if the person might have a coronary heart disease inthe next ten years or not. This study is useful in early identification of disease and have medicalintervention if necessary. This would help not only in improving the health conditions but also theeconomy as it has been identified that health performance and economic performance areinterlinked. This research intends to develop appropriate models to identify/predict if the personlikely to have heart disease or not

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
get_ipython().run_line_magic('matplotlib', 'inline')


# # Data 
# 
# The total number of records is 34281 with 24 independent attributes and the 25th column is the target which needs to be predicted. The variables are masked, and we get little information from their names. Missing values are represented as NA in some columns and as -99 in some other columns

# In[2]:


Train_df=pd.read_csv("C:/Users/vamshi/Downloads/csv files heart/Problem1_train.csv")
Train_df


# In[3]:


Train_df.head(10)


# In[4]:


Train_df.isnull().sum()


# In[5]:


sn.heatmap(Train_df.isnull())


# In[6]:


Train_df.loc[Train_df['A2'].isnull()]


# In[7]:


count=0
for i in Train_df.isnull().sum(axis=1):
    if i>0:
        count=count+1
print('Total number of rows with missing values is ', count)
print('since it is only',round((count/len(Train_df.index))*100), 'percent of the entire dataset the rows with missing values are excluded.')


# since it is only 5 percent of the entire dataset the rows with missing values are excluded.
# 

# # Handling Missing Values

# In[8]:


features_nan=[feature for feature in Train_df.columns if Train_df[feature].isnull().sum()>1]
for feature in features_nan:
    print("{}: {}% missing values".format(feature,np.round(Train_df[feature].isnull().mean(),4)))


# In[9]:


mean_A2 = round(Train_df["A2"].mean())
mean_A2


# In[10]:


Train_df['A2'].fillna(mean_A2, inplace = True)


# In[11]:


Train_df.isnull().sum()


# In[12]:


sn.heatmap(Train_df.isnull())


# In[13]:


max_thresold = Train_df["A21"].quantile(0.99)
max_thresold


# In[14]:


Train_df[ Train_df['A21']>max_thresold]


# In[15]:


min_thresold = Train_df['A21'].quantile(0.03)
min_thresold


# In[16]:


Train_df[Train_df['A21']<min_thresold]


# In[17]:


Train_df=Train_df[(Train_df['A21']<max_thresold) & (Train_df['A21']>min_thresold)]
Train_df


# In[18]:


Train_df.shape


# In[19]:


max_thresold = Train_df["A16"].quantile(0.90)
max_thresold


# In[20]:


Train_df[ Train_df['A16']>max_thresold]


# In[21]:


Train_df.shape


# In[22]:


sn.pairplot(Train_df[["A1","A2"]])


# In[23]:


sn.pairplot(Train_df[["A3","A4"]])


# In[24]:


Train_df.Target.value_counts()


# In[25]:


sn.countplot(x='Target',data=Train_df)


# In[26]:


Train_df.describe()


# In[27]:


import sklearn
new_features=Train_df[['ID','IV', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8',
       'A9', 'A11', 'A10', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18',
       'A19', 'A20', 'A21', 'A22', 'Target']]
x_train=new_features.iloc[:,:-1]
y_train=new_features.iloc[:,-1]


# In[28]:


x_train


# In[29]:


y_train


# In[30]:


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(random_state = 0)
logreg.fit(x_train,y_train)


# In[31]:


Test_df=pd.read_csv("C:/Users/vamshi/Downloads/csv files heart/Problem1_test.csv")
Test_df


# In[32]:


Test_df.isnull().sum()


# In[33]:


sn.heatmap(Test_df.isnull())


# In[34]:


count=0
for i in Test_df.isnull().sum(axis=1):
    if i>0:
        count=count+1
print('Total number of rows with missing values is ', count)
print('since it is only',round((count/len(Test_df.index))*100), 'percent of the entire dataset the rows with missing values are excluded.')


# In[35]:


mean_A2 = round(Test_df["A2"].mean())
mean_A2


# In[36]:


Test_df['A2'].fillna(mean_A2, inplace = True)


# In[37]:


sn.heatmap(Test_df.isnull())


# In[38]:


max_thresold = Test_df["A21"].quantile(0.99)
max_thresold


# In[39]:


Test_df[ Test_df['A21']>max_thresold]


# In[40]:


min_thresold = Test_df['A21'].quantile(0.03)
min_thresold


# In[41]:


Test_df[Test_df['A21']<min_thresold]


# In[42]:


Test_df=Test_df[(Test_df['A21']<max_thresold) & (Test_df['A21']>min_thresold)]
Test_df


# In[43]:


Test_df.groupby('Target').mean()


# In[44]:


Test_df.Target.value_counts()


# In[45]:


sn.countplot(x='Target',data=Test_df)


# In[46]:


import sklearn
new_features=Test_df[['ID','IV', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8',
       'A9', 'A11', 'A10', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18',
       'A19', 'A20', 'A21', 'A22', 'Target']]
x_test=new_features.iloc[:,:-1]
y_test=new_features.iloc[:,-1]


# In[47]:


x_test


# In[48]:


y_test


# In[49]:


y_pred=logreg.predict(x_test)


# In[50]:


y_pred


# # Model Evaluation

# In[51]:


from sklearn.metrics import accuracy_score
print('Accuracy: ',accuracy_score(y_test, y_pred))
plt.show()


# # Confusion matrix
# 
# A confusion matrix is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known.

# In[52]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Actual:1','Actual:0'],index=['predicted:yes','predicted:no'])
plt.figure(figsize = (8,5))
sn.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")


# In[53]:


cm=confusion_matrix(y_test,y_pred)
print ("Confusion Matrix : \n", cm)


# In[54]:


TN=cm[0,0]
TP=cm[1,1]
FN=cm[1,0]
FP=cm[0,1]
sensitivity=TP/float(TP+FN)
specificity=TN/float(TN+FP)


# In[55]:


sensitivity


# In[56]:


specificity


# # Model Evaluation - Statistics¶

# In[57]:


print('The acuuracy of the model = TP+TN/(TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n',

'The Missclassification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n',

'Sensitivity or True Positive Rate = TP/(TP+FN) = ',TP/float(TP+FN),'\n',

'Specificity or True Negative Rate = TN/(TN+FP) = ',TN/float(TN+FP),'\n',

'Positive Predictive value = TP/(TP+FP) = ',TP/float(TP+FP),'\n',

'Negative predictive Value = TN/(TN+FN) = ',TN/float(TN+FN),'\n',

'Positive Likelihood Ratio = Sensitivity/(1-Specificity) = ',sensitivity/(1-specificity),'\n',

'Negative likelihood Ratio = (1-Sensitivity)/Specificity = ',(1-sensitivity)/specificity)


# From the above statistics it is clear that the model is highly specific than sensitive. The positive values are predicted more accurately than the nagatives.

# # Predicted probabilities of 0 (No Coronary Heart Disease) and 1 ( Coronary Heart Disease: Yes) for the test data with a default classification threshold of 0.5

# In[58]:


y_pred_prob=logreg.predict_proba(x_test)[:,:]
y_pred_prob_df=pd.DataFrame(data=y_pred_prob, columns=['Prob of no heart disease (0)','Prob of Heart Disease (1)'])
y_pred_prob_df.head(10)


# # Lower the threshold

# Since the model is predicting Heart disease too many type II errors is not advisable. A False Negative ( ignoring the probability of disease when there actualy is one) is more dangerous than a False Positive in this case. Hence inorder to increase the sensitivity, threshold can be lowered.

# In[59]:


from sklearn.preprocessing import binarize
for i in range(1,5):
    cm2=0
    y_pred_prob_yes=logreg.predict_proba(x_test)
    y_pred2=binarize(y_pred_prob_yes,i/10)[:,1]
    cm2=confusion_matrix(y_test,y_pred2)
    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',
            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',
          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')
    


# In[61]:


print('Positive Likelihood Ratio = Sensitivity/(1-Specificity) = ',sensitivity/(1-specificity),'\n',

'Negative likelihood Ratio = (1-Sensitivity)/Specificity = ',(1-sensitivity)/specificity)


# # ROC curve¶

# In[62]:


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_yes[:,1])
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Heart disease classifier')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)


# A common way to visualize the trade-offs of different thresholds is by using an ROC curve, a plot of the true positive rate (# true positives/ total # positives) versus the false positive rate (# false positives / total # negatives) for all possible choices of thresholds. A model with good classification accuracy should have significantly more true positives than false positives at all thresholds.
# 
# The optimum position for roc curve is towards the top left corner where the specificity and sensitivity are at optimum levels

# # Area Under The Curve (AUC)
# The area under the ROC curve quantifies model classification accuracy; the higher the area, the greater the disparity between true and false positives, and the stronger the model in classifying members of the training dataset. An area of 0.5 corresponds to a model that performs no better than random classification and a good classifier stays as far away from that as possible. An area of 1 is ideal. The closer the AUC to 1 the better.

# In[63]:


sklearn.metrics.roc_auc_score(y_test,y_pred_prob_yes[:,1])


# # CONCLUSION
# The model predicted with 0.69 accuracy. The model is more specific than sensitive.
# 
# 
# 
# **The Area under the ROC curve is 79.7 which is somewhat satisfactory. **
# 
# 
# 
# ** Overall model could be improved with more data.**
# 
# ​
# 
