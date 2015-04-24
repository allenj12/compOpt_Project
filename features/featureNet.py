
# coding: utf-8

# In[54]:

from IPython.core import display
from io import BytesIO
import Image
import numpy as np
import random
import os
import neurolab as nl
from matplotlib import pyplot as plt
import matplotlib.cm as cm

get_ipython().magic(u'matplotlib inline')


# In[2]:

os.chdir("..")
get_ipython().magic(u'pwd')


# In[74]:

def getData(path):
    "Splits the data into X and y both being numpy arrays (data already normalized)"
    f = open(path)
    lines = f.readlines()
    f.close
    
    images = []
    classification = []
    for line in lines:
        nums = line.split()
        classification.append(nums[0])
        images.append([((float (val))) for val in nums[1:]])
    return (np.array(classification),np.array(images))

def display_grayscale(arr):
    "SIDE EFFECTS: INTENDED TO BE USED IN IPYTHON NOTEBOOK"
    img = arr.astype('uint8').reshape((16,16))
    plt.imshow(img, cmap = cm.Greys_r)
    return Image.fromarray(img)


# In[75]:

classification,trainData = getData("ZipDigits.train.txt")


# In[13]:

# Create network with 256 inputs, 2 neurons in hidden layer
# And 256 in output layer
testData = trainData[:101]
inputParams = [[-1, 1]] * len(trainData[0])
ann = nl.net.newff(inputParams, [2, 256])

# Train process
err = ann.train(testData, testData,epochs=500,show=1)

test = ann.sim(testData)


# In[64]:

print trainData[0][8]
print test[0][8]


# In[76]:

example = display_grayscale(trainData[0])

example.save('example.png')


# In[ ]:



