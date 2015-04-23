
# coding: utf-8

# In[1]:

from IPython.core import display
from io import BytesIO
import Image
import numpy as np
import random
import os
import neurolab as nl


# In[2]:

os.chdir("..")
get_ipython().magic(u'pwd')


# In[3]:

def getData(path):
    f = open(path)
    lines = f.readlines()
    f.close
    
    images = []
    classification = []
    for line in lines:
        nums = line.split()
        classification.append(nums[0])
        images.append([((float (val))) for val in nums[1:]])
    return (classification,images)


# In[4]:

classification,trainData = getData("ZipDigits.train.txt")


# In[5]:

# Create network with 256 inputs, 2 neurons in hidden layer
# And 256 in output layer
testData = trainData[:101]
inputParams = [[-1, 1]] * len(trainData[0])
ann = nl.net.newff(inputParams, [2, 256])

# Train process
err = ann.train(testData, testData,epochs=20,show=5)

test = ann.sim(testData)

print test



# In[12]:

print trainData[0][7]
print test[0][7]


# In[ ]:



