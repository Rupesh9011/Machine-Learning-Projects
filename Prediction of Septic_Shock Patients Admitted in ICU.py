#!/usr/bin/env python
# coding: utf-8

# # Prediction of Septic_Shock Patients Admitted in ICU(Intensive critical care unit.)

# Sepsis is a serious condition in which the body responds improperly to an infection. 
# The infection-fighting processes turn on the body, causing the organs to work poorly. 
# Sepsis may progress to septic shock. This is a dramatic drop in blood pressure that 
# can damage the lungs, kidneys, liver and other organs. When the damage is severe, 
# it can lead to mortality.

# Sepsis can progress to septic shock when certain changes in the circulatory system, 
# the body's cells and how the body uses energy become more abnormal. 
# Septic shock is more likely to cause death than sepsis is. 
# 

# Sepsis is a clinical syndrome of life-threatening organ dysfunction caused by 
# a dysregulated response to infection. In septic shock, there is critical reduction 
# in tissue perfusion; acute failure of multiple organs, including the lungs, kidneys, 
# and liver, can occur.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data=pd.read_csv("sepsis_status.csv")

dictionary = pd.read_csv("Parameter _detail_Dictionary.csv")


# In[3]:


icu_df=data.copy()
icu_df.head()


# In[4]:


icu_df.info()


# In[5]:


icu_df["septic_shock"].value_counts()


# In[6]:


dictionary


# In[7]:


dictionary.Category.value_counts()


# In[8]:


def selectCategory(df,category):
    return df[df.Category==category]


# GOSSIS score refers to the Global Open Source Severity of Illness Score 
# which gives an idea about how severe a patient is. We can try 
# comparing this particular feature with the hospital_death column to 
# if there is any correlation between the 2 columns.

# In[9]:


cols=selectCategory(dictionary,"GOSSIS example prediction")


# In[10]:


cols


# In[11]:


cols=selectCategory(dictionary,"APACHE grouping")


# In[12]:


cols


# In[13]:


icu_df.head()


# In[14]:


icu_df.isnull().sum()


# In[15]:


icu_df.isnull().sum().sort_values(ascending=False)


# In[16]:


high_null=icu_df.isnull().sum()/len(icu_df)


# In[17]:


high_null.sort_values(ascending=False)


# We have mainly two categories of missing data- one having more than 50% missing
# other having less than 25% missing

# In[18]:


high_missing=high_null[high_null>=0.50].index


# In[19]:


high_missing


# In[20]:


len(high_missing)


# In[21]:


# dropping the high missing (more than 50% missing values)


# In[22]:


icu=icu_df.drop(high_missing, axis=1)


# In[23]:


icu_df.shape


# In[24]:


icu.shape


# In[25]:


icu.info()


# In[26]:


icu.isnull().sum()


# In[27]:


icu.isnull().any().sum()


# In[28]:


icu.isnull().any(axis=1).sum()


# In[29]:


icu_df.shape


# In[30]:


df1=icu.dropna()


# In[31]:


df1.shape


# In[32]:


df1.isnull().sum().sum()


# In[33]:


df1.info()


# In[34]:


df1.head()


# In[35]:


# lets drop some columns which is having no significance as a basic understanding


# In[36]:


cols_to_drop=["encounter_id","patient_id","hospital_admit_source","icu_admit_source","icu_id","icu_stay_type"]
df1=df1.drop(cols_to_drop,axis=1)
df1.shape


# In[37]:


df1.info()


# In[38]:


object_columns=df1.select_dtypes(include="object").columns


# In[39]:


object_columns


# In[40]:


df1_final=pd.get_dummies(df1,columns=object_columns,drop_first=True,dtype=int)


# In[41]:


df1_final.head()


# In[42]:


df1_final.info()


# In[43]:


# split the dependent and independent variables


# In[44]:


X=df1_final.drop("septic_shock",axis=1)
Y=df1_final["septic_shock"]


# In[45]:


Y.value_counts(normalize=True)


# In[46]:


# we must check if any column having only one variable / or no variance column


# In[47]:


# finding the unique variable columns
one_unique=X.apply(pd.Series.nunique)
one_unique


# In[48]:


const_cols=one_unique[one_unique==1].index
const_cols


# In[49]:


X["readmission_status"].value_counts()


# In[50]:


X["gcs_unable_apache"].value_counts()


# In[51]:


# we can drop these columns having no variance
X.drop(['readmission_status','gcs_unable_apache'],axis=1,inplace=True)


# In[52]:


corr_matrix=X.corr(method="spearman").abs()


# In[53]:


corr_matrix


# In[54]:


import seaborn as sns
sns.heatmap(corr_matrix)


# In[55]:


upper=corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))


# In[56]:


upper


# In[57]:


#high_cor_threshold=0.9


# In[58]:


high_cor_features=[col for col in upper.columns if any(upper[col]>=0.9)]


# In[59]:


high_cor_features


# In[60]:


len(high_cor_features)


# In[61]:


X1=X.drop(high_cor_features,axis=1)


# In[62]:


X1.shape


# In[63]:


X.shape


# In[64]:


from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler


# In[65]:


scale=MinMaxScaler()

X_final=scale.fit_transform(X1)


# In[66]:


X_final=pd.DataFrame(X_final,columns=X1.columns)


# In[67]:


X_final.head()


# In[68]:


X_final.corrwith(Y)


# In[69]:


# Model Building process


# In[70]:


from sklearn.model_selection import train_test_split


# In[71]:


X_train,X_test,Y_train,Y_test=train_test_split(X_final,Y,test_size=0.2,stratify=Y,random_state=100)


# In[72]:


#import algorithm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.svm import SVC


# In[73]:


rf=RandomForestClassifier()
bg=BaggingClassifier()
svc=SVC()
dt=DecisionTreeClassifier()


# # Modeling

# In[74]:


from sklearn.model_selection import StratifiedShuffleSplit,cross_val_score,cross_val_predict


# In[75]:


from sklearn.model_selection import cross_validate


# In[76]:


for model in[rf,bg,dt,svc]:
    print("======="*5)
    print("Performance of",model)
    print("======="*5)
    cv=StratifiedShuffleSplit(n_splits=10,test_size=0.25,random_state=50)
    scoring=["roc_auc","f1_macro","precision_macro","recall_macro"]
    cross_val_scores=cross_validate(model,X_final,Y,scoring=scoring,cv=cv)
    roc_auc_test_cv=round(cross_val_scores["test_roc_auc"].mean(),4)
    f1_test_cv=round(cross_val_scores["test_f1_macro"].mean(),4)
    precisio_test_cv=round(cross_val_scores["test_precision_macro"].mean(),4)
    Recall_test_cv=round(cross_val_scores["test_recall_macro"].mean(),4)
    
    # print of metrics
    print("ROC AUC",roc_auc_test_cv) 
    print("F1 MACRO",f1_test_cv) 
    print("PRECISION MACRO",precisio_test_cv) 
    print("RECALL MACRO",Recall_test_cv) 
    


# In[77]:


from sklearn.metrics import accuracy_score, classification_report, precision_score, f1_score, roc_auc_score, recall_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, roc_curve, precision_recall_curve


# In[78]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
for model in[rf,bg,dt,svc]:
    print("======="*6)
    print("Performance of",model)
    print("======="*6)
    abc=model.fit(X_train,Y_train)
    y_pred=abc.predict(X_test)
    cm=confusion_matrix(Y_test,y_pred)
    AS=accuracy_score(Y_test,y_pred)
    CR=classification_report(Y_test,y_pred)
    ROC=roc_auc_score(Y_test,y_pred)
    PRFS=precision_recall_fscore_support(Y_test,y_pred, average='macro')
        
    # print of metrics
    print("confusion matrix \n",cm) 
    print("======="*3)
    print("Accuracy \n",AS) 
    print("======="*3)
    print("Classification Report \n",CR) 
    print("======="*3)
    print("ROC_AUC \n",ROC) 
    print("======="*3)
    print("MACRO Precision_recall_F1 \n",PRFS) 
    #print("PRECISION MACRO",precisio_test_cv) 
    #print("RECALL MACRO",Recall_test_cv) 


# In[79]:


# balancing the data


# In[80]:


from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from imblearn.under_sampling import TomekLinks, RandomUnderSampler, NearMiss, ClusterCentroids


# In[81]:


smote = SMOTE()
adasyn=ADASYN()
over=RandomOverSampler()
border=BorderlineSMOTE()
under=RandomUnderSampler()


# In[82]:


for model in[rf,bg,dt,svc]:
    print("======="*6)
    print("Performance of",model)
    print("======="*6)
    for balancer in[smote,border,under]:
        X_train_1,Y_train_1=balancer.fit_resample(X_train,Y_train)
        abc=model.fit(X_train_1,Y_train_1)
        y_pred=abc.predict(X_test)
        cm=confusion_matrix(Y_test,y_pred)
        AS=accuracy_score(Y_test,y_pred)
        CR=classification_report(Y_test,y_pred)
        ROC=roc_auc_score(Y_test,y_pred)
        PRFS=precision_recall_fscore_support(Y_test,y_pred, average='macro')
        
        # print of metrics
        print("confusion matrix \n",cm) 
        print("======="*3)
        print("Accuracy \n",AS) 
        print("======="*3)
        print("Classification Report \n",CR) 
        print("======="*3)
        print("ROC_AUC \n",ROC) 
        print("======="*3)
        print("MACRO Precision_recall_F1 \n",PRFS) 
    #print("PRECISION MACRO",precisio_test_cv) 
    #print("RECALL MACRO",Recall_test_cv) 

