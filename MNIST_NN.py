#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[4]:


def load_data(filePath):
    df=pd.read_csv(filePath)
    x=df['label']
    Y =df.drop('label',axis=1)
    return x,Y


# In[5]:


x,y=load_data('mnist_train.csv')


# In[6]:


index=y.columns


# In[7]:


t_d=pd.DataFrame(x)
t_d


# In[8]:


def normalize_data(x):
    r= x/(255*0.99) +.01
    return r
    


# In[9]:


df=y[index].apply(normalize_data)


# In[10]:


#def hot_encodings(df):
output_label=pd.get_dummies(t_d['label'])


# In[11]:


type(output_label)
y_label=np.array(output_label)


# In[12]:


y_label


# In[30]:


inpu=np.array(df[index])
inpu


# In[18]:


y_label[1]


# In[21]:


from keras.models import Sequential


# In[22]:


from keras.layers import Dense


# In[23]:


import numpy


# In[24]:


model=Sequential()
model.add(Dense(784,input_dim=784,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(10,activation='softmax'))
#model.compile()


# In[26]:


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[33]:


model.fit(inpu,y_label,epochs=15,batch_size=10)


# In[32]:


scores = model.evaluate(inpu, y_label)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

