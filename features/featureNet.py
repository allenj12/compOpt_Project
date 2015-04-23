
# coding: utf-8

# In[1]:

from IPython.core import display
from io import BytesIO
import Image
import numpy as np
import random
import os

os.chdir("..")


# In[13]:

def getData(path):
    f = open(path)
    lines = f.readlines()
    f.close
    
    images = []
    for line in lines:
        nums = line.split()
        images.append([((float (val))) for val in nums])
    print images[0]


# In[14]:

getData("ZipDigits.train.txt")


# In[9]:




# In[ ]:



