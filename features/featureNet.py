
# coding: utf-8

# In[2]:

from IPython.core import display
from io import BytesIO
from PIL import Image
import numpy as np
import random
import os
import neurolab as nl
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import time

get_ipython().magic(u'matplotlib inline')


# In[3]:

os.chdir("..")
get_ipython().magic(u'pwd')


# In[6]:

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
    return (np.array(classification, dtype=np.float),np.array(images))

def display_grayscale(arr):
    "SIDE EFFECTS: INTENDED TO BE USED IN IPYTHON NOTEBOOK"
    img = np.array([round(val,2) for val in arr]).astype('uint8').reshape((16,16))
    plt.imshow(img, cmap = cm.Greys_r)
    return Image.fromarray(img)


# In[7]:

#get the data in two arrays classifications and the actual data

#classificationTrain,trainData = getData("ZipDigits.train.txt")
#classificationTest,testData = getData("ZipDigits.test.txt")
#combined set since we will use or on seperation technique
classification,data = getData('combinedData.txt')


# In[8]:

# Create network with 256 inputs, 2 neurons in hidden layer
# And 256 in output layer

ann = nl.load('features/ann80.net')
size = 500
subData = data[:size]
#inputParams = [[-1, 1]] * len(subData[0])
#ann = nl.net.newff(inputParams, [80,256])
#ann.trainf = nl.train.train_rprop


# In[8]:

#cuts the last layer off of ann making the net that makes the features
inputParams = [[-1, 1]] * len(subData[0])
featureNet = nl.net.newff(inputParams, [80])
featureNet.layers[0].np['w'][:] = ann.layers[0].np['w']
featureNet.layers[0].np['b'][:] = ann.layers[0].np['b']


# In[8]:

#trainNetFeatures = featureNet.sim(trainData)
#testNetFeatures = featureNet.sim(testData)
netFeatures = featureNet.sim(data)


# In[14]:

#np.savetxt('features/trainNetFeatures.txt', trainNetFeatures)
#np.savetxt('features/testNetFeatures.txt', testNetFeatures)
#np.savetxt('features/classificationTrain.txt',classificationTrain)
#np.savetxt('features/classificationTest.txt',classificationTest)

#for combined data since we will probably use K-folds anyway
np.savetxt('features/netFeatures.txt', netFeatures)
np.savetxt('features/classification.txt', classification)


# In[9]:

# Train process
#last condition epochs=100000000,show=100000,
err = ann.train(subData, subData,epochs=1,show=1, goal=25600.0)


# In[11]:

#this block just used for comparing random pictures within the training set
ind = random.randrange(0,size)
sub = ann.sim(subData[ind:ind+1])
print "test"
exIn = display_grayscale(sub[0])


# In[12]:

exOut = display_grayscale(data[ind])


# In[6]:

#outside of the data set
ind = random.randrange(size,size+200)
exampleInput = display_grayscale(data[ind])


# In[7]:

exampleOutput = ann.sim(data[ind:ind+1])
#exampleTest = 
exampleOut = display_grayscale(exampleOutput[0])

#example.save('example1OUTPUT.png')


# In[10]:

ann.save('features/ann80.net')


# In[10]:

out = featureNet.sim(data)


# In[14]:

np.savetxt('features/data80NetFeatures', out)

