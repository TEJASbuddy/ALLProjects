#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import string


# In[2]:


df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")


# In[4]:


df_fake.head(10)


# In[5]:


df_true.head(10)


# In[6]:


df_fake["class"] = 0
df_true["class"] = 1


# In[7]:


df_fake.shape, df_true.shape


# In[8]:


df_fake_manual_testing = df_fake.tail(10)
for i in range(23480,23470,-1):
    df_fake.drop([i],axis=0, inplace=True)
df_true_manual_testing = df_true.tail(10)
for i in range(21416,21406,-1):
    df_true.drop([i],axis=0, inplace=True)


# In[9]:


df_manual_testing = pd.concat([df_fake_manual_testing], axis=0)
df_manual_testing.to_csv("manual_testing.csv")


# In[10]:


df_merge = pd.concat([df_fake, df_true], axis=0)
df_merge.head(10)


# In[11]:


df= df_merge.drop(["title","subject","date"], axis=1)
df.head(10)


# In[12]:


df = df.sample(frac=1)


# In[13]:


df.head(10)


# In[14]:


df.isnull().sum()


# In[15]:


def word_drop(text):
    text = text.lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+','',text)
    text = re.sub('<.*?>+','',text)
    text = re.sub('[%s]'% re.escape(string.punctuation),'',text)
    text = re.sub('\n','',text)
    text = re.sub('\w*\d\w*','',text)
    return text


# In[16]:


df["text"] = df["text"].apply(word_drop)


# In[17]:


df.head(10)


# In[18]:


x = df["text"]
y = df["class"]


# In[19]:


x_train , x_test, y_train ,y_test = train_test_split(x,y, test_size = .25)


# In[20]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[21]:


vectrorization = TfidfVectorizer()
xv_train = vectrorization.fit_transform(x_train)
xv_test = vectrorization.transform(x_test)


# Logistic Regression

# In[22]:


from sklearn.linear_model import LogisticRegression


# In[23]:


LR = LogisticRegression()
LR.fit(xv_train, y_train)


# In[24]:


LR.score(xv_test , y_test)


# In[25]:


pred_LR = LR.predict(xv_test)


# In[26]:


print(classification_report(y_test, pred_LR))


# Decision tree classification

# In[27]:


from sklearn.tree import DecisionTreeClassifier


# In[28]:


DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)


# In[29]:


DT.score(xv_test, y_test)


# In[30]:


pred_DT = DT.predict(xv_test)


# In[31]:


print(classification_report(y_test, pred_DT))


# Gradient Boosting Classifier

# In[32]:


from sklearn.ensemble import GradientBoostingClassifier


# In[34]:


GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train, y_train)


# In[35]:


GBC.score(xv_test, y_test)


# In[36]:


pred_GBC = GBC.predict(xv_test)


# In[38]:


print(classification_report(y_test, pred_GBC))


# Random Forest Classifier

# In[39]:


from sklearn.ensemble import RandomForestClassifier


# In[40]:


RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train,y_train)


# In[41]:


RFC.score(xv_test, y_test)


# In[42]:


pred_RFC = RFC.predict(xv_test)


# In[43]:


print(classification_report(y_test, pred_RFC))


# Manual Testing

# In[53]:


def output_label(n):
    if n==0:
        return "Fake News"
    elif n==1:
        return "True News"
def manaul_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(word_drop)
    new_x_test = new_def_test["text"]
    new_xv_test = vectrorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)
    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(output_label(pred_LR[0]), 
                                                                                                              output_label(pred_DT[0]), 
                                                                                                              output_label(pred_GBC[0]), 
                                                                                                              output_label(pred_RFC[0])))


# In[54]:


news = str(input())
manaul_testing(news)

