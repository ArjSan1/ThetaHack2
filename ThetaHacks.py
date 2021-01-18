#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[1]:


# imports
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflowjs as tfjs
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import cv2
import os
import glob
import h5py
import os.path
import streamlit as st
if os.path.isfile(r'C:\Users\Arjun\Desktop\ThetaHacks/machineLearningModel2.h5') is False:
    model.save('machineLearningModel2.h5')
from tensorflow.keras.models import load_model
model = load_model(r'C:\Users\Arjun\Desktop\ThetaHacks/machineLearningModel2.h5')
# Deep learning stuff
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import ReduceLROnPlateau


# ## Splitting testing and training data and prepocessing images

# In[2]:



DIRECTORY = r'C:\Users\Arjun\Desktop\ThetaHacks\COVID RADIOLOGY DATASET'
os.listdir(DIRECTORY)
labels = ['COVID', 'NORMAL']
img_size = 128
def datafunc(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_arr = cv2.cvtColor(img_arr,cv2.COLOR_GRAY2RGB)               
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) 
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

train_data = datafunc(r'C:\Users\Arjun\Desktop\ThetaHacks\COVID RADIOLOGY DATASET\train')
test_data = datafunc(r'C:\Users\Arjun\Desktop\ThetaHacks\COVID RADIOLOGY DATASET\test')
val = datafunc(r'C:\Users\Arjun\Desktop\ThetaHacks\COVID RADIOLOGY DATASET\val')



# ## Displaying split between COVID and Normal in the training data

# In[3]:


trainlabels = []
for img in train_data:
    if (img[1] == 0):
        trainlabels.append("Covid")
    else:
        trainlabels.append("Normal")
sns.countplot(trainlabels)


# ## Appending labels and necessary features

# In[4]:


x_train = []
y_train = []

x_test = []
y_test = []

x_val = []
y_val = []

for feature, label in train_data:
    x_train.append(feature)
    y_train.append(label)

for feature, label in test_data:
    x_test.append(feature)
    y_test.append(label)
    
for feature, label in val:
    x_val.append(feature)
    y_val.append(label)
    


# ## Creating array that ranges from 0-1

# In[5]:


x_train = np.array(x_train)/255.0
x_test = np.array(x_test)/255.0
x_val = np.array(x_val)/255.0


# ## Reshapes arrray to be used in model

# In[6]:


x_train = (x_train.reshape(-1, img_size, img_size, 3))
x_test = (x_test.reshape(-1, img_size, img_size, 3))
x_val = (x_val.reshape(-1, img_size, img_size, 3))
x_train.shape


# In[7]:


x_train = np.array(x_train)
x_test = np.array(x_test)
x_val = np.array(x_val)

y_train = np.array(y_train)
y_test = np.array(y_test)
y_val = np.array(y_val)


# ## Import CNN model to be used to classify data

# In[8]:


from keras.models import Sequential
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Concatenate
from keras import applications
input_shape = (128,128, 3)

base_model = keras.applications.InceptionResNetV2(weights='imagenet', 
                                include_top=False, 
                                input_shape=(128,128,3))
base_model.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation='relu')(x)
x= BatchNormalization()(x)
x=Dropout(0.3)(x)
x= Dense(64,activation='relu')(x)
x= BatchNormalization()(x)
x=Dropout(0.3)(x)
predictions = Dense(2, activation='softmax')(x)
for layer in base_model.layers:
    layer.trainable = False

model=Sequential()
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='RMSprop',
              metrics=['accuracy'])


# 

# In[9]:



datagenerator = ImageDataGenerator(zoom_range = .2)
datagenerator.fit(x_train)


# ## The two following cells help us determine the constraints and inputs for our model

# In[10]:


length=len(x_train)
b_max= 80 # set this based on  how much your  memory can hold
batch_size=sorted([int(length/n) for n in range(1,length+1) if length % n ==0 and 
                  length/n<=b_max],reverse=True)[0] 
steps=int(length/batch_size)


# In[11]:


length=len(x_test)
b_max= 80 # set this based on  how much your  memory can hold
validation_batch_size=sorted([int(length/n) for n in range(1,length+1) if length % n ==0 and 
                  length/n<=b_max],reverse=True)[0] 
validation_steps=int(length/batch_size)


# ## Sets the amount of layers in the CNN

# In[12]:


for layer in model.layers[:630]:
    layer.trainable = False
for layer in model.layers[630:]:
    layer.trainable = True


# ## Shows ten COVID and ten Normal images

# In[29]:


x = range(10)

for i in x:
    plt.figure(figsize = (5,5))
    plt.imshow(x_test[i], cmap='gray')
    plt.title(labels[train_data[0][1]] )


b = 240
for i in x:
    plt.figure(figsize = (5,5))
    plt.imshow(x_test[i+b], cmap='gray')
    plt.title(labels[train_data[-1][1]])


# ## Implementing the model and implementing user input

# In[31]:



model.compile(loss='sparse_categorical_crossentropy',optimizer='RMSprop',metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("inceptionresnet.h5", monitor='val_acc', verbose=1, save_best_only=True, 
                             save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='auto')
#history = model.fit(datagenerator.flow(x_train,y_train, batch_size) ,epochs = 5, 
 #                   verbose=1 , steps_per_epoch=30, validation_steps=int(length/batch_size), 
  #                  validation_batch_size=sorted([int(length/n) for n in range(1,length+1) if length % n ==0 and 
   #               length/n<=b_max],reverse=True)[0])
#print(x_test[1])

        

print("This is a machine learning model where you are able to choose a Covid or normal image, and then it outputs the probability of the x-ray being covid or not")

choice = input("Would you like to choose a Covid or normal image: \n ")
if choice == "Covid":
    index = input("Which image number would you like to select? Enter a number between 0-9 \n")
    index = int(index)
    x = x_test[index]
    x = (x.reshape(-1, img_size, img_size, 3))
    img_class = model.predict(x)
    print(img_class[0])
else:
    index = input("Which image number would you like to select? Enter a number between 0-9 \n")
    index = int(index) +239
    x = x_test[index]
    x = (x.reshape(-1, img_size, img_size, 3))
    img_class = model.predict(x)
    print("[probability of covid   probability of non-covid]: ", img_class[0])




# In[ ]:




