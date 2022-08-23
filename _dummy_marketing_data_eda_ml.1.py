#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[5]:


import pandas as pd


# In[7]:


import matplotlib.pyplot as plt


# In[9]:


import seaborn as sns


# In[13]:


df = pd.read_csv(r"C:\Users\korum\Downloads\Dummy Data HSS(1).xls" )


# In[18]:


df = df.dropna()
df


# In[20]:


df.describe()


# In[22]:


df.info()


# In[24]:


df.Influencer.value_counts()


# In[30]:


df.Influencer.unique()


# In[32]:


df["Influencer_encoded"] = df["Influencer"]
influencer_mapping = {'Mega':3, 'Micro':1, 'Nano':0, 'Macro':2}
df["Influencer_encoded"] = df["Influencer_encoded"].map(influencer_mapping)
df.head()


# In[34]:


col_names = ['Macro', 'Mega', 'Micro', 'Nano']


# In[37]:


df.groupby("Influencer")["Sales"].mean()


# In[40]:


p = np.array(df.groupby("Influencer")["Sales"].mean())
p


# In[42]:


influencer_df = pd.DataFrame()
influencer_df["Type"] = col_names
influencer_df["values"] = p
influencer_df


# In[49]:


sns.scatterplot(x=influencer_df["Type"],y=influencer_df["values"])


# In[48]:


sns.scatterplot(x=df["TV"],y=df["Sales"],cmap="curve")


# In[56]:


sns.scatterplot(x=df["Radio"],y=df["Sales"],cmap="rainbow")


# In[58]:


sns.scatterplot(x=df["Social Media"],y=df["Sales"])


# In[61]:


sns.scatterplot(x=df["Social Media"],y=df["Sales"],cmap="rainbow",c=df["Influencer_encoded"])


# In[63]:


sns.scatterplot(x=df["Influencer"],y=df["Social Media"],cmap="rainbow",c=df["Influencer_encoded"])


# In[65]:




df.groupby("Influencer")["Social Media"].mean()


# In[67]:


df


# In[70]:


inf_one_hot = pd.get_dummies(df["Influencer"],drop_first=True)
inf_one_hot


# In[72]:


df_encoded = pd.concat([df,inf_one_hot],axis=1)
df_encoded = df_encoded.drop(["Influencer",'Influencer_encoded'],axis=1)
df_encoded


# In[74]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded)
df_scaled = pd.DataFrame(df_scaled,columns=df_encoded.columns)
df_scaled


# In[76]:


from sklearn.model_selection import train_test_split


# In[78]:




X = df_scaled.drop("Sales",axis=1)
y = df_scaled["Sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[79]:




from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
lin_reg.score(X_test,y_test)

