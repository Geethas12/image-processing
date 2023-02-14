#!/usr/bin/env python
# coding: utf-8

# In[7]:


#global thresholding using image
import numpy as np
import cv2
from matplotlib import pyplot as plt
img=cv2.imread('fig.png',0)
epsilon=0.001
diff_threshold=100
mean_data=[np.mean(img)]
print(mean_data)


# In[8]:


print(len(img))


# In[3]:


[r, c]=img.shape


# In[4]:


while diff_threshold > epsilon:
    data_one=[]
    data_two=[]
    for i in range(r):
        for j in range(c):
            if img[i,j]<mean_data :
                data_one.append(img[i,j])
            else:
                data_two.append(img[i,j])
    print(data_one)
    print(data_two)
    mu_one=np.mean(data_one)
    mu_two=np.mean(data_two)
    avg_mean=(mu_one+mu_two)/2
    diff_threshold=abs(mean_data-avg_mean)
    mean_data=avg_mean


# In[5]:


print(mean_data)


# In[6]:


plt.imshow(img)


# In[ ]:




