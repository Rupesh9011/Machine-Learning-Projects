#!/usr/bin/env python
# coding: utf-8

# # cab cancellation prediction

# The cab bookings data are made available through a collaboration between 
# Prof. Galit Shmueli at the Indian School of Business and YourCabs co-founder 
# Mr. Rajath Kedilaya and IDRC managing partner, Mr. Amit Batra.
# 
# This competition is part of the MBA elective Business Analytics Using Data Mining, 
# offered at the Indian School of Business.
# 
# 
# 
# YourCabs is a platform to efficiently connect urban consumers in need of local transport, 
# with vendors in need of increased occupancy.
# 
# 
# 
# Industrial Data Research Corp. (IDRC) is a data sciences consultancy focused on 
# Quantitative Modeling, Data Analytics, Scientific Computing, and Data Visualization/Infographics.
# 
# 

# id - booking ID
# user_id - the ID of the customer (based on mobile number)
# vehicle_model_id - vehicle model type.
# package_id - type of package (1=4hrs & 40kms, 2=8hrs & 80kms, 3=6hrs & 60kms, 4= 10hrs & 100kms, 5=5hrs & 50kms, 6=3hrs & 30kms, 7=12hrs & 120kms)
# travel_type_id - type of travel (1=long distance, 2= point to point, 3= hourly rental).
# from_area_id - unique identifier of area. Applicable only for point-to-point travel and packages
# to_area_id - unique identifier of area. Applicable only for point-to-point travel
# from_city_id - unique identifier of city
# to_city_id - unique identifier of city (only for intercity)
# from_date - time stamp of requested trip start
# to_date - time stamp of trip end
# online_booking - if booking was done on desktop website
# mobile_site_booking - if booking was done on mobile website
# booking_created - time stamp of booking
# from_lat - latitude of from area
# from_long - longitude of from area
# to_lat - latitude of to area
# to_long - longitude of to area
# Car_Cancellation  - whether the booking was cancelled (1) or not (0) due to unavailability of a car.
# Cost_of_error - the cost incurred if the booking is misclassified. 
# 
# Target Variable--car cancellation
# 
# For  No cancellation---0
# 
# For a cancelled booking, ---1 
# 

# In[ ]:





# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.options.display.max_colwidth = 170


# In[2]:


data = pd.read_csv('Cab_cancellation.csv')


# In[3]:


df=data.copy()


# In[4]:


df.tail()


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.isnull().sum()


# In[8]:


(df.isnull().sum()/len(df))*100


# In[9]:


df1=df.drop(["id","package_id","from_city_id","to_city_id"],axis=1)


# In[10]:


(df1.isnull().sum()/len(df1))*100


# In[11]:


df1.shape


# In[12]:


df1["to_area_id"]=df1["to_area_id"].fillna(df1["to_area_id"].mode()[0])


# In[13]:


df1["to_lat"]=df1["to_lat"].fillna(df1["to_lat"].mean())
df1["to_long"]=df1["to_long"].fillna(df1["to_long"].mean())


# In[14]:


(df1.isnull().sum()/len(df1))*100


# In[15]:


df2=df1.copy()


# In[16]:


df2=df2.dropna(axis=0,subset=["from_area_id","from_lat"])


# In[17]:


(df2.isnull().sum()/len(df2))*100


# In[18]:


df2.shape


# In[19]:


df2.head()


# In[20]:


df3=df2.drop("to_date",axis=1)


# In[21]:


df3.isnull().sum().sum()


# In[22]:


df3.shape


# In[23]:


df3.info()


# In[24]:


# convert the required format datetime


# In[25]:


df3["from_date"]=pd.to_datetime(df3["from_date"])
df3["booking_created"]=pd.to_datetime(df3["booking_created"])


# In[26]:


df3.info()


# In[27]:


#calculate the time lag  difference between booking time and journey start time


# In[28]:


df3.head()


# In[29]:


df3["time_diff"]=df3["from_date"]-df3["booking_created"]


# In[30]:


df3.head(1)


# In[31]:


df3.info()


# In[32]:


# extracting information from timedelta datatype--- column time_diff


# In[33]:


#td=pd.to_timedelta(df3.time_diff).dt.components


# In[34]:


#print(td)


# In[35]:


td=df3["time_diff"].dt.components
td


# In[36]:


df3["time_diff_day"]=td['days']+(td['hours']/24)+(td['minutes']/(24*60))


# In[37]:


df3.head()


# In[38]:


df3=df3.drop(["time_diff"],axis=1)


# In[39]:


df3.head(2)


# In[40]:


# calculate the distance from latitute and longitude


# In[ ]:





# In[41]:


#pip install geopy


# In[42]:


from geopy.distance import geodesic
from geopy.distance import distance


# In[43]:


from geopy import Point


# In[44]:


df10=df3.copy()


# In[45]:


df10["point_1"]=df10.apply(lambda row: Point(latitude=row["from_lat"],longitude=row["from_long"]),axis=1)


# In[46]:


df10.head()


# In[47]:


# add a new shifted point_next 


# In[48]:


df10["point_next"]=df10["point_1"].shift(1)


# In[49]:


df10.head()


# In[50]:


df10["point_2"]=df10.apply(lambda row: Point(latitude=row["to_lat"],longitude=row["to_long"]),axis=1)


# In[51]:


df10.head()


# In[52]:


# calculation of distance


# In[53]:


df10["Distance"]=df10.apply(lambda  row:distance(row["point_1"], row["point_2"]).km,axis=1)


# In[54]:


df10.head()


# In[55]:


df11=df10.copy()


# In[56]:


df11=df11.drop(["from_date","from_lat","from_long","to_lat","to_long","point_1","point_next","point_2"],axis=1)


# In[57]:


df11.head(1)


# # assignment :-please do some visualization using matplotlib seaborn,crosstab

# ANALYSING ONLINE_BOOKING

# In[58]:


ax=pd.crosstab(df11.online_booking,df11.Car_Cancellation,normalize='index')
ax.columns=['not_cancelled','cancelled']
ax.plot(kind='bar',stacked=True,figsize=(5,4))

plt.show()
print(ax)


# ANALYSING MOBILE_SITE_BOOKING

# In[59]:


ax=pd.crosstab(df11.mobile_site_booking,df11.Car_Cancellation,normalize='index')
ax.columns=['not_cancelled','cancelled']
ax.plot(kind='bar',stacked=True,figsize=(5,4))

plt.show()
print(ax)


# We observe that mobile_site_booking have double possibility to become canceled

# ANALYSING VEHICLE_MODEL_ID

# In[60]:


#Analysing vehicle_model_id
all_bookings=df11['vehicle_model_id'].value_counts()
cancelled=df11[df11['Car_Cancellation']==1]['vehicle_model_id'].value_counts()
not_cancelled=df11[df11['Car_Cancellation']==0]['vehicle_model_id'].value_counts()
qf=pd.DataFrame([all_bookings,cancelled,not_cancelled])
#qf.fillna(0,inplace=True)
qf.index=['booked','cancelled','not_cancelled']
qf.plot(kind='bar',stacked=True,figsize=(15,8))
plt.show()


# ANALYSING travel_type_id

# In[61]:


#Analysing travel_type_id
all_bookings=df11['travel_type_id'].value_counts()
cancelled=df11[df11['Car_Cancellation']==1]['travel_type_id'].value_counts()
not_cancelled=df11[df11['Car_Cancellation']==0]['travel_type_id'].value_counts()
qf=pd.DataFrame([all_bookings,cancelled,not_cancelled])
#qf.fillna(0,inplace=True)
qf.index=['booked','cancelled','not_cancelled']
qf.plot(kind='bar',stacked=True,figsize=(15,8))
plt.show()


# In[ ]:





# ### FIND HOLIDAYS IN DATASET AND SET THEM AS AN INDEPENDENT VARIABLE

# In[62]:


df12=df11.copy()


# In[63]:


dates=[]
date =[]
for i in df12['booking_created']:
    date = [i.month,i.day]
    dates.append(date)

holidays=[[1,26],[8,15],[10,2],[4,22],[1,14],[4,24],[3,29],[5,22],[9,9],[10,14],[10,13],[11,24],[11,17],[12,25],[1,1]]
#_,_,_,maharashi,id,_,holi,maharana pratap,janmastami,dp,dp,chatt,deepawali,_,_
on_holidays=[]
for i in dates:
    if i in holidays:
        on_holidays.append(1)
    else:
        on_holidays.append(0)
on_holidays = pd.DataFrame({'On_holidays':on_holidays})
df12.insert(8, 'On_holidays', on_holidays)
df12.head()


# In[64]:


on_holidays.head()


# ANALYSING ON_HOLIDAYS VARIABLE

# In[65]:


ax=pd.crosstab(df12.On_holidays,df12.Car_Cancellation,normalize='index')
ax.columns=['not_cancelled','cancelled']
ax.plot(kind='bar',stacked=True,figsize=(5,4))

plt.show()
print(ax)


# In[66]:


#big posibility to canceled on holidays


# Double posibility to canceled a reservation on holidays

# In[67]:


holiday_cancellation=df12.groupby(['On_holidays','Car_Cancellation']).size()
holiday_cancellation


# In[ ]:





# In[ ]:





# In[ ]:





# In[68]:


# extracting month , date,hour from booking_created column


# In[ ]:





# In[69]:


import datetime as dt


# In[70]:


df11["Month"]=df11["booking_created"].dt.month
df11["date"]=df11["booking_created"].dt.day
df11["hour"]=df11["booking_created"].dt.hour


# In[71]:


df11.head()


# In[ ]:





# In[72]:


df11.isnull().sum()


# In[73]:


# create a function for the national holiday and festivals from booking_created coulmn


# In[ ]:





# In[74]:


df_final=df11.copy()


# In[75]:


df_final=df_final.drop(["booking_created"],axis=1)


# In[76]:


X=df_final.drop(["Car_Cancellation","Cost_of_error"],axis=1)


# In[77]:


Y=df_final["Car_Cancellation"]


# In[78]:


# multi collinearity of X


# In[79]:


corr_matrix=X.corr(method="spearman").abs()


# In[80]:


import seaborn as sns
plt.figure(figsize=(20,20))
sns.heatmap(corr_matrix,annot=True)


# In[81]:


# scaling of X


# In[82]:


# JUST GIVE THE VARIABLE NAME AS X2_final


# In[83]:


from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
scale = MinMaxScaler()
X_scaled = scale.fit_transform(X)
X2_final = pd.DataFrame(X_scaled,columns=X.columns)
X2_final.head()


# In[84]:


# ASSIGNMENT -- PLEASE CHECK XGB,ADB,AND OTHER ALGO WITH IMBALANCE DATA ALSO.


# In[85]:


#hello


# # Model building

# In[86]:


from sklearn.model_selection import train_test_split


# In[87]:


X_train,X_test,Y_train,Y_test=train_test_split(X2_final,Y,test_size=0.2,stratify=Y,random_state=100)


# In[88]:


#import algorithm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.svm import SVC


# In[89]:


rf=RandomForestClassifier()
bg=BaggingClassifier()
svc=SVC()
dt=DecisionTreeClassifier()


# In[90]:


from sklearn.model_selection import StratifiedShuffleSplit,cross_val_score,cross_val_predict


# In[91]:


from sklearn.model_selection import cross_validate


# for model in[rf,bg,dt,svc]:
#     print("======="*5)
#     print("Performance of",model)
#     print("======="*5)
#     cv=StratifiedShuffleSplit(n_splits=10,test_size=0.25,random_state=50)
#     scoring=["roc_auc","f1_macro","precision_macro","recall_macro"]
#     cross_val_scores=cross_validate(model,X2_final,Y,scoring=scoring,cv=cv)
#     roc_auc_test_cv=round(cross_val_scores["test_roc_auc"].mean(),4)
#     f1_test_cv=round(cross_val_scores["test_f1_macro"].mean(),4)
#     precisio_test_cv=round(cross_val_scores["test_precision_macro"].mean(),4)
#     Recall_test_cv=round(cross_val_scores["test_recall_macro"].mean(),4)
#     
#     # print of metrics
#     print("ROC AUC",roc_auc_test_cv) 
#     print("F1 MACRO",f1_test_cv) 
#     print("PRECISION MACRO",precisio_test_cv) 
#     print("RECALL MACRO",Recall_test_cv) 
#     

# In[92]:


from sklearn.metrics import accuracy_score, classification_report, precision_score, f1_score, roc_auc_score, recall_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, roc_curve, precision_recall_curve


# In[93]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
for model in[rf,bg,dt]:
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


# In[94]:


# Balancing the data


# In[95]:


from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from imblearn.under_sampling import TomekLinks, RandomUnderSampler, NearMiss, ClusterCentroids


# In[96]:


smote = SMOTE()
adasyn=ADASYN()
over=RandomOverSampler()
border=BorderlineSMOTE()
under=RandomUnderSampler()


# In[97]:


for model in[rf]:
    print("======="*6)
    print("Performance of",model)
    print("======="*6)
    for balancer in[border,under]:
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


# In[ ]:





# In[98]:


from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# In[99]:


ad=AdaBoostClassifier()
gb=GradientBoostingClassifier()
xb=XGBClassifier()
cat=CatBoostClassifier()


# In[100]:


for model in[ad,gb,xb,cat]:
    print("======="*8)
    print("AFTER data-Balancing Performance:-",model)
    print("======="*8)
    for balancer in[border,under]:
        X_train_1, Y_train_1 = balancer.fit_resample(X_train, Y_train)
        abc=model.fit(X_train_1,Y_train_1)
        y_pred=abc.predict(X_test)
        cm=confusion_matrix(Y_test,y_pred)
        AS=accuracy_score(Y_test,y_pred)
        CR=classification_report(Y_test,y_pred)
        ROC=roc_auc_score(Y_test,y_pred)
        PRFS=precision_recall_fscore_support(Y_test,y_pred, average='macro')
        
        print(model,"using technique:- ",balancer)
        #print("======="*6) 
        print("======="*8) 
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
        print("======="*8)
        print("Conclude one balaning method")
        print("======="*8)
        
        print("======="*8)


# In[101]:


from sklearn.ensemble import VotingClassifier


# In[102]:


estimators=[
    ("gb",GradientBoostingClassifier()),
    ("xb",XGBClassifier()),
    ("rf",RandomForestClassifier()),
    ("ad",AdaBoostClassifier(base_estimator=DecisionTreeClassifier())),
    ("bg",BaggingClassifier())
]


# In[103]:


vc=VotingClassifier(estimators=estimators,voting="hard")


# In[104]:


for model in[vc]:
    print("======="*8)
    print("AFTER data-Balancing Performance:-",model)
    print("======="*8)
    for balancer in[smote,under]:
        X_train_1, Y_train_1 = balancer.fit_resample(X_train, Y_train)
        abc=model.fit(X_train_1,Y_train_1)
        y_pred=abc.predict(X_test)
        cm=confusion_matrix(Y_test,y_pred)
        AS=accuracy_score(Y_test,y_pred)
        CR=classification_report(Y_test,y_pred)
        ROC=roc_auc_score(Y_test,y_pred)
        PRFS=precision_recall_fscore_support(Y_test,y_pred, average='macro')
        
        print(model,"using technique:- ",balancer)
        #print("======="*6) 
        print("======="*8) 
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
        print("======="*8)
        print("Conclude one balaning method")
        print("======="*8)
        
        print("======="*8)
        
        


# In[105]:


#StackingClassifier


# In[106]:


from sklearn.ensemble import StackingClassifier


# In[107]:


estimators=[
    ("gb",GradientBoostingClassifier()),
    ("xb",XGBClassifier()),
    ("rf",RandomForestClassifier()),
    ("ad",AdaBoostClassifier(base_estimator=DecisionTreeClassifier())),
    ("bg",BaggingClassifier())
]


# In[108]:


sclf = StackingClassifier(estimators=estimators,
                            final_estimator=rf,
                            cv=10)


# In[ ]:


for model in[sclf]:
    print("======="*8)
    print("AFTER data-Balancing Performance:-",model)
    print("======="*8)
    for balancer in[smote,under]:
        X_train_1, Y_train_1 = balancer.fit_resample(X_train, Y_train)
        abc=model.fit(X_train_1,Y_train_1)
        y_pred=abc.predict(X_test)
        cm=confusion_matrix(Y_test,y_pred)
        AS=accuracy_score(Y_test,y_pred)
        CR=classification_report(Y_test,y_pred)
        ROC=roc_auc_score(Y_test,y_pred)
        PRFS=precision_recall_fscore_support(Y_test,y_pred, average='macro')
        
        print(model,"using technique:- ",balancer)
        #print("======="*6) 
        print("======="*8) 
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
        print("======="*8)
        print("Conclude one balaning method")
        print("======="*8)
        
        print("======="*8)
        


# In[ ]:





# In[ ]:


vc1=VotingClassifier(estimators=estimators,voting="soft",weights=[2,2,3,0.2,1])


# In[ ]:


for model in[vc1]:
    print("======="*8)
    print("AFTER data-Balancing Performance:-",model)
    print("======="*8)
    for balancer in[smote,under]:
        X_train_1, Y_train_1 = balancer.fit_resample(X_train, Y_train)
        abc=model.fit(X_train_1,Y_train_1)
        y_pred=abc.predict(X_test)
        cm=confusion_matrix(Y_test,y_pred)
        AS=accuracy_score(Y_test,y_pred)
        CR=classification_report(Y_test,y_pred)
        ROC=roc_auc_score(Y_test,y_pred)
        PRFS=precision_recall_fscore_support(Y_test,y_pred, average='macro')
        
        print(model,"using technique:- ",balancer)
        #print("======="*6) 
        print("======="*8) 
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
        print("======="*8)
        print("Conclude one balaning method")
        print("======="*8)
        
        print("======="*8)
        


# In[ ]:





# In[ ]:


#conclusion:-


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




