#!/usr/bin/env python
# coding: utf-8

# # Online_shopping & Recommendation Engine

# Online shopping Customer Segmentation

# **Problem statement**
# -  # Customer Segmentation and Recommendation
# -  1.Segment (Group/Cluster) customer on basis of RFM (Recency, Frequency, Monetary) 
# -  2.Identify trends for Day, Month, Season, Time by Invoice count
# -  3.Create a recommendation of different product as per specific client. 
# -  4 Movie Recommendation as per Movie Tags / Review.(MOVIE DATASET)
# -  

# Recency, frequency, monetary value (RFM) is a model used in marketing 
# analysis that segments a company’s consumer base by their purchasing 
# patterns or habits. In particular, it evaluates customers’ recency 
# (how long ago they made a purchase), frequency (how often they make purchases), 
# and monetary value (how much money they spend).

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[2]:


original_data=pd.read_excel("Online_shopping.xlsx")


# # Problem statement 1. Segment (Group/Cluster) customer on basis of RFM (Recency, Frequency, Monetary) 

# In[4]:


df=original_data.copy()
df.head(100)


# In[5]:


# Recency ---- Latest date-Last date of invoice of a customer
# Frequency ---- Total no. of Invoice generated a particular client
# Monetary ----- Sum of all invoice of a particular client


# In[6]:


df["Country"].value_counts(normalize=True)


# In[7]:


df.info()


# In[8]:


# Segmentaion - customer clustering on basis of Recency , Frequency,Monetary


# In[9]:


# we dont need the following items.


# In[10]:


df1=df.drop(["StockCode","Description","lower","Country"],axis=1)
df1.head(2)


# In[11]:


df1.isnull().sum()


# In[12]:


df1.dropna(axis=0,inplace=True)


# In[13]:


df1.isnull().sum()


# In[14]:


df1.describe()


# In[15]:


# Quantity is negative because some customer has returned the item.
# unit price can be zero because its a free item


# In[16]:


df1[df1["UnitPrice"]==0]["UnitPrice"].count()


# In[17]:


df1[df1["UnitPrice"]==0].head(10)


# In[18]:


# we are considering return as - No busines but we are not dropping those transations and customers


# In[19]:


df1=df1[df1["Quantity"]>0]


# In[20]:


df1.describe()


# In[21]:


df1.head()


# In[22]:


# create a column total =Qty* unit price


# In[23]:


df1["Total"]=df1["Quantity"]*df1["UnitPrice"]
df1.head()


# In[24]:


df2=df1.drop(["Quantity","UnitPrice"],axis=1)
df2.head()


# In[25]:


df_final100=df2.groupby(["InvoiceNo"]).agg({"Total":"sum"}).reset_index()
df_final100.head()


# In[26]:


df_final=df2.groupby(["InvoiceNo","InvoiceDate","CustomerID"]).agg({"Total":"sum"}).reset_index()


# In[27]:


df_final.head(100)


# In[28]:


# Goal 1:- PERFORMING Cust Segment on R F M


# In[29]:


max(df_final["InvoiceDate"])


# In[30]:


min(df_final["InvoiceDate"])


# In[31]:


# the given dataset belongs to Dec 1st 2010 to Dec 9th 2011


# Recency= Latest Date - last Invoice date of customer

# In[32]:


# to create a latest date 


# In[33]:


import datetime as dt
latest_date=dt.datetime(2011,12,10)
latest_date


# In[34]:


# Recency--Latest Date - last Invoice date of customer
# frequency - count of invoice no.(how many times the customer id appear)
# Monetary-Sum of total-


# In[35]:


df_final.head()


# In[36]:


RFM_Score=df_final.groupby("CustomerID").agg({"InvoiceDate":lambda x:(latest_date-x.max()).days,
                                             "InvoiceNo":lambda x:x.count(),
                                             "Total":lambda x:x.sum()}).reset_index()


# In[37]:


RFM_Score.head()


# In[38]:


# Rename the column Name
RFM_Score=RFM_Score.rename(columns={"InvoiceDate":"Recency","InvoiceNo":"Frequency","Total":"Monetary"})


# In[39]:


RFM_Score.head()


# In[40]:


RFM1=RFM_Score.copy()


# In[41]:


RFM1["R_Rank"]=RFM1["Recency"].rank(ascending=False)


# In[42]:


RFM1["R_Rank"]


# In[43]:


RFM1["F_Rank"]=RFM1["Frequency"].rank(ascending=True)


# In[44]:


RFM1["M_Rank"]=RFM1["Monetary"].rank(ascending=True)


# In[45]:


RFM1.head()


# In[46]:


RFM1["R_final"]=(RFM1["R_Rank"]/RFM1["R_Rank"].max())*100


# In[47]:


RFM1["R_final"]


# In[48]:


RFM1["F_final"]=(RFM1["F_Rank"]/RFM1["F_Rank"].max())*100


# In[49]:


RFM1["M_final"]=(RFM1["M_Rank"]/RFM1["M_Rank"].max())*100


# In[50]:


RFM1.head()


# In[51]:


RFM1.drop(columns=["R_Rank","F_Rank","M_Rank"],inplace=True)


# In[52]:


RFM1.head()


# In[53]:


# we can assign the weightage on R F M


# In[54]:


RFM1["RFM_SUM"]=0.15*RFM1["R_final"]+0.30*RFM1["F_final"]+0.60*RFM1["M_final"]


# In[55]:


RFM1.head()


# In[56]:


RFM1=RFM1.round(2)
RFM1.head()


# In[57]:


max(RFM1["RFM_SUM"])


# In[58]:


min(RFM1["RFM_SUM"])


# In[59]:


loyality_label=["Silver","Gold","Diamond","Platinum"]


# In[60]:


quantile100=RFM1["RFM_SUM"].quantile(q=[0.25,0.50,0.75])


# In[61]:


quantile100


# In[62]:


# create a function
def loyality(x):
    if x<=28:
        return "Silver"
    elif x<=51:
        return "Gold"
    elif x<=76:
        return "Diamond"
    else:
        return "Platinum"
        


# In[63]:


RFM1["Loyality"]=RFM1["RFM_SUM"].apply(loyality)


# In[64]:


RFM1.head(100)


# In[65]:


RFM1["Loyality"].value_counts()


# In[66]:


# all the above segregation is mannual approach


# # K means ALGORITHM segrigation

# In[67]:


New_data=RFM1["RFM_SUM"]
New_data.head()
New_data=pd.DataFrame(New_data)


# In[68]:


from sklearn.cluster import KMeans


# In[69]:


kmeans1=KMeans(n_clusters=4,max_iter=100,random_state=10)


# In[70]:


y_means=kmeans1.fit_predict(New_data)
y_means


# In[71]:


RFM1["Cluster"]=kmeans1.labels_
RFM1.head(100)


# In[72]:


RFM1["Cluster"].value_counts()


# In[73]:


# Evaluation of cluster


# In[74]:


from sklearn.metrics import silhouette_score


# In[75]:


labels=kmeans1.labels_
print(silhouette_score(New_data,labels))


# In[76]:


# Goal no. 2 ------ Analysing the sales Trend as per day, timing, season, month,weekend


# In[77]:


df_final.head()


# In[78]:


df_final.info()


# In[79]:


df_final100=df_final.copy()
df_final100.head()


# In[80]:


# how to extract info from Datetime datatypes


# In[81]:


import datetime as dt


# In[82]:


df_final100["Day"]=df_final100["InvoiceDate"].dt.day_name()


# In[83]:


df_final100.head()


# In[84]:


df_final100["Day"].value_counts()


# In[85]:


df_final100["Month"]=df_final100["InvoiceDate"].dt.month


# In[86]:


df_final100["Hour"]=df_final100["InvoiceDate"].dt.hour


# In[87]:


df_final100.head()


# In[88]:


df_final100["week_info"]=np.where((df_final100.Day=="Saturday")|(df_final100.Day=="Sunday"),"Weekend","WeekDay")


# In[89]:


df_final100.head()


# In[90]:


# Analysing the business as per day


# In[91]:


day_count=df_final100.groupby("Day")["InvoiceNo"].count().to_frame("count_of_invoices").reset_index()


# In[92]:


day_count


# In[93]:


plt.pie(day_count['count_of_invoices'],autopct='%0.01f%%',labels=day_count['Day'])
plt.show()


# #### assignment 
# (1)plot the no. of invoice by season...winter, Monsoon,Summer,spring
# (2)plot the no. of invoice by morning, afternoon, evening 
# (3)plot the no. of invoice by Month 

# In[94]:


# function to create season and timing


# In[95]:


def cal_season(x):
    if x in [10,11,12,1]:
        return "Winter"
    if x in [2,3]:
        return "spring"
    if x in [4,5,6]:
        return "Summer"
    else:
        return "Monsoon"


# In[96]:


def cal_timing(x):
    if x in range(4,7):
        return "early morning"
    elif x in range(7,11):
        return "Morning"
    elif x in range(11,16):
        return "afternoon"
    else:
        return "evening"


# In[97]:


df_final100["season"]=df_final100["Month"].apply(cal_season)


# In[98]:


df_final100["Time"]=df_final100["Hour"].apply(cal_timing)


# In[99]:


df_final100.head()


# In[100]:


df_final100["Time"].value_counts()


# In[101]:


season_count = df_final100.groupby('season')['InvoiceNo'].count().to_frame('count_of_invoices')


# In[102]:


season_count.reset_index(inplace=True)


# In[103]:


season_count


# In[104]:


plt.pie(season_count.count_of_invoices,labels=season_count['season'],autopct='%.1f%%')
plt.show()


# In[105]:


# plot number of customers by season graph
season=df_final100["season"].value_counts()
plt.figure(figsize=(10,5))
sns.barplot(season.index,season.values)
#plot the average 
plt.axhline(y=season.mean())
plt.title("No. of customers by season")
plt.ylabel("no. of customers",fontsize=12)
plt.xlabel("season of year",fontsize=12)
plt.show()


# In[ ]:





# # Recommendation Engine

# The recommendations that companies give you sometimes use data analysis 
# techniques to identify items that match your taste and preferences. 

# Recommendation Engine:-
# A recommendation engine is a data filtering system that operates on different 
# machine learning algorithms to recommend products, services, and information to 
# users based on data analysis. It works on the principle of finding patterns in 
# customer behavior data employing a variety of factors such as customer preferences, 
# past transaction history, attributes, or situational context.

# Different Techniques of Recommendation Engines
# There are three different types of recommender engines known in 
# machine learning, and they are:
# 
# 1. Collaborative Filtering
# 
# The collaborative filtering method collects and analyzes data on user 
# behavior, online activities, and preferences to predict what they will 
# like based on the similarity with other users. It uses a matrix-style 
# formula to plot and calculates these similarities.
# 
# Advantage
# 
# One significant advantage of collaborative filtering is that 
# it doesn’t need to analyze or understand the object (products, films, books) 
# to recommend complex items precisely. There is no dependence on analyzable 
# machine content, which means it chooses recommendations based on what it knows about the user.
# 
# Example
# 
# If user X likes Book A, Book B, and Book C while user Y likes Book A, 
# Book B, and Book D, they have similar interests. So, it is favorably 
# possible that user X would select Book D and user Y would enjoy reading Bood C. 
# This is how collaborative filtering happens.
# 
# 2. Content-Based Filtering
# 
# Content-based filtering works on the principle of describing a product 
# and a profile of the user’s desired choices. It assumes that you will 
# also like this other item if you like a particular item. Products are 
# defined using keywords (genre, product type, color, word length) to make 
# recommendations. A user profile is created to describe the kind of item 
# this user enjoys. Then the algorithm evaluates the similarity of items 
# using cosine and Euclidean distances.
# 
# Advantage
# 
# One significant advantage of this recommender engine technique is that it 
# does not need any additional data about other users since the recommendations 
# are specific to this user. Also, this model can capture the particular 
# interests of a user and suggest niche objects that very few other users are interested in.
# 
# Example
# 
# Suppose a user X likes to watch action movies like Spider-man. In that case, 
# this recommender engine technique only recommends movies of the action genre 
# or films describing Tom Holland.
# 
# 3. Hybrid Model
# 
# In hybrid recommendation systems, both the meta (collaborative) data and 
# the transactional (content-based) data are used simultaneously to suggest 
# a broader range of items to the users. In this technique, natural language 
# processing tags can be allocated for each object (movie, song), and vector 
# equations calculate the similarity. A collaborative filtering matrix can then 
# suggest things to users, depending on their behaviors, actions, and intentions.
# 
# Advantages
# 
# This recommendation system is up-and-coming and is said to outperform both of 
# the above methods in terms of accuracy.
# 
# Example
# 
# Netflix uses a hybrid recommendation engine. It makes recommendations by 
# analyzing the user’s interests (collaborative) and recommending such 
# shows/movies that share similar attributes with those rated highly by the user(content-based).

# In[106]:


df50=original_data.copy()
df50.head(100)


# In[107]:


# drop the column "lower"


# In[108]:


df50=df50.drop(["lower"],axis=1)


# In[109]:


df50.isnull().sum()


# In[110]:


df50.dropna(axis=0,inplace=True)


# In[111]:


df50.isnull().sum()


# In[112]:


df50.shape


# In[113]:


# step 1-- create a pivot_table
# step 2-- apply cosine_similarity function on pivot_table


# In[114]:


custI_matrix=df50.pivot_table(index="CustomerID",columns="StockCode",values="Quantity",aggfunc="sum",fill_value=0)


# In[115]:


custI_matrix


# In[116]:


# we want to see the similarity or filteration of item code as per customer ID 


# In[117]:


from sklearn.metrics.pairwise import cosine_similarity


# In[118]:


similarity=cosine_similarity(custI_matrix)


# In[119]:


similarity


# In[120]:


similarity.shape


# In[121]:


similarity[90]


# In[122]:


sorted(similarity[90],reverse=True)


# In[123]:


final=pd.DataFrame(similarity)


# In[124]:


final


# In[125]:


# we need to set the customerid at rows and columns


# In[126]:


final.columns=custI_matrix.index


# In[127]:


final.head()


# In[128]:


final["customerID"]=custI_matrix.index


# In[129]:


final=final.set_index("customerID")


# In[130]:


final.head(100)


# In[131]:


# above matrix is the degree of similarity among the customers. we can use it for Recommendation


# In[132]:


# for example reference cust id 12350.0


# In[133]:


final.loc[12350.0].sort_values(ascending=False)


# In[134]:


# CUSTOMER A--12350.0
item_cust_A=set(custI_matrix.loc[12350.0].iloc[custI_matrix.loc[12350.0].to_numpy().nonzero()].index)


# In[135]:


item_cust_A


# In[136]:


# CUSTOMER B--15180.0
item_cust_B=set(custI_matrix.loc[15180.0].iloc[custI_matrix.loc[15180.0].to_numpy().nonzero()].index)


# In[137]:


item_cust_B


# In[138]:


Recommended_item_list_cust_A=item_cust_B-item_cust_A


# In[139]:


Recommended_item_list_cust_A


# In[140]:


df50.loc[df50["StockCode"].isin(item_cust_B),["StockCode","Description"]].drop_duplicates().set_index("StockCode")


# In[ ]:




