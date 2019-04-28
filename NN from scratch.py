#!/usr/bin/env python
# coding: utf-8

# importing library

# In[1]:


import numpy as np 
import pandas as pd


# In[2]:


def load_data(filePath):
    df=pd.read_csv(filePath)
    x=df['label']
    Y =df.drop('label',axis=1)
    return x,Y


# In[3]:


y,x=load_data('mnist_train.csv')


# In[5]:


index=x.columns
data=pd.DataFrame(y)


# Normalizing features variable

# In[6]:


def normalize_data(x):
    r= x/(255*0.99) +.01
    return r
df=x[index].apply(normalize_data)
output_label=pd.get_dummies(data['label'])


# In[7]:


type(output_label)
y_label=np.array(output_label)


# In[8]:


input_data=np.array(df[index])


# In[9]:


input_data.shape


# Neural Network Without Backpropagation

# In[10]:


def activation(m):
    s=np.exp(-m)
    sig=1/(1+s)
    return sig
def fetch_input(each):
    return input_data[each]
def node(n,data):
    if n==0:
        weight=np.random.random((100,784))
        bias=np.random.random(100)
        data1=np.dot(weight,data)+bias
        a=activation(data1)
    if n==1:
        weight=np.random.random((10,100))
        bias=np.random.random((10))
        data1=np.dot(weight,data)+bias
        a=activation(data)
    return a   
def make_layer(l):
    layer_node=[]
    for each in range(input_data.shape[0]):
        data=fetch_input(each)
        for eachs in range(l):
            data=node(eachs,data)
        layer_node.append(data)
    return layer_node 


pp=make_layer(2)


# Neural network with backpropagation

# In[11]:


class two_layer_nn:
#     L is layer in nn
    def __init__(self,L):
        self.error=1
        self.L=L
        #self.x=x
        #self.y=y
        #self.lr=lr
        self.n=len(L)
        self.parameter={}
        #for i in range(self.n):
        self.input_node=np.array(L[0])
        for i in range(1,self.n-1,1):
            
            self.parameter['weight'+str(i)]=np.random.rand(self.L[i+1],self.L[i])
        for i in range(self.n-1,0,-1):
            self.parameter['bias'+str(i)]=np.random.rand(self.L[i],1)
    def initialization(self,x):
        self.input_node=x
        self.parameter['a0']=self.input_node
    def forward_prop(self):
        for i in range(1,self.n,1):
            self.parameter['z'+str(i)]=np.dot(self.parameter['weight'+str(i)],self.parameter['a'+str(i-1)])+ self.parameter['bias'+str(i)]
            if(i!=self.n-1):
                self.parameter['a'+str(i)]=np.sigm(self.parameter['z'+str(i)])
            else:
                exp=np.exp(self.parameter['z'+str(i)])
                exprs=np.sum(exp)                                                                                                                 
                self.parameter['a'+str(i)]=np.divide(exp,exps)                                                                                                                 
    def compute_error(self,y):
         self.error= -np.sum(np.dot(y,np.log(self.parameter['a'+str(self.n-1)])))
    def backward_prop(self,lr):
        for i in range(self.n-1,0,-1):
            self.parameter['bias'+str(i)]=self.parameter['bias'+str(i)]-lr*self.error
            self.parameter['weight'+str(i)]=self.parameter['weight'+str(i)]-lr*(self.parameter['a'+str(i-1)]*self.error)
    def predict(self,x):
        self.initialization(x)
        self.forward_prop(self)
        return self.parameter['a'+str(self.n-1)]
    def fit(self,x,y,lr):
        nc=0
        for i in range(x.shape[0]):
            self.initialization(x[i])
            self.forward_prop()
            self.compute_error(y)
            self.backward_prop(lr)
            y1=self.predict(x[i].reshape(len(x[i],1)))
            ys=list(y[i])
            for i in y1:
                if i>0.5 and ys[i]==1:
                    nc=+1
        
        print('accuracy:'(nc/x.shape[0])*100)
    
    
        


# In[ ]:


nnk=[784,10,10]
l=np.array([784,10,10])
nn=two_layer_nn(nnk)
nn.fit(input_data,y_label,0.07)

