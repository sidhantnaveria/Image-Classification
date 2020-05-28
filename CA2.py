# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 17:41:36 2019

@author: sidhant
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.initializers import he_normal
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization

dataDir="C:\sidhant\CA2\data"
category=["healthyfood","junkfood"]

def save(X,y,x_test,y_test):
    pickle_out = open("X.pickle","wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()
    
    pickle_out = open("x_test.pickle","wb")
    pickle.dump(x_test, pickle_out)
    pickle_out.close()
    
    pickle_out = open("y.pickle","wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()
    
    pickle_out = open("y_test.pickle","wb")
    pickle.dump(y_test, pickle_out)
    pickle_out.close()

def load():
    pickle_in = open("X.pickle","rb")
    X = pickle.load(pickle_in)
    
    pickle_in = open("x_test.pickle","rb")
    x_test = pickle.load(pickle_in)
    
    pickle_in = open("y.pickle","rb")
    y = pickle.load(pickle_in)
    
    pickle_in = open("y_test.pickle","rb")
    y_test = pickle.load(pickle_in)
    return X,x_test,y,y_test
def data_set():
    
    IMG_SIZE=50
    data=[]
    
    for cat in category:  # do dogs and cats
        path = os.path.join(dataDir,cat)
        class_num=category.index(cat)
        for img in os.listdir(path):
            
        # iterate over each image per dogs and cats
            try:
                
                img_array = cv2.imread(os.path.join(path,img) )  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                data.append([new_array, class_num])
            except Exception as e:  # in the interest in keeping the output clean...
                    pass
                
    
    
    random.shuffle(data)
    
    X=[]
    y=[] 
    for features,label in data:
        X.append(features)
        y.append(label)         
    X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)
    
    
    X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    
    X_train = X_train/255.0
    X_test=X_test/255.0
    
    save(X_train,y_train,X_test,y_test)
    
    
    


X_train,X_test,y_train,y_test=load()
model = Sequential()
model.add(Conv2D(256, (3, 3), input_shape=X_train.shape[1:]))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())
model.add(Dense(64,activation='relu' ,kernel_regularizer=regularizers.l2(0.05)))

model.add(Dropout(0.3))
model.add(Dense(32,activation='sigmoid' ,kernel_regularizer=regularizers.l2(0.05)))
model.add(Dropout(0.5))
model.add(Dense(32, activation='sigmoid' ,kernel_regularizer=regularizers.l2(0.05)))
model.add(Dropout(0.3))



model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=60, validation_split=0.3)

result=model.evaluate(X_test,y_test)
predicts    = model.predict(X_test)

#predout = np.argmax(predicts)
#testout = np.argmax(y_test)
#
#
#testScores = metrics.accuracy_score(y_test,predicts)
#confusion = metrics.confusion_matrix(testout,predout)
print(result)
#print(confusion)
       