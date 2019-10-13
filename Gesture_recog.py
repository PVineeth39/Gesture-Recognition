import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
import tensorflow as tf


model=Sequential()
model.add(Conv2D(128,3,3,input_shape=(128,128,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(output_dim=128,activation='relu',init='random_uniform'))
model.add(Dense(output_dim=4,activation='softmax',init='random_uniform'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])




from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)


x_train=train_datagen.flow_from_directory("D:/MLAI/Hackathon/Gesture recog/last/train_set",target_size=(128,128),batch_size=10,class_mode='categorical')
x_test=test_datagen.flow_from_directory("D:/MLAI/Hackathon/Gesture recog/last/test_set",target_size=(128,128),batch_size=10,class_mode='categorical')



print(x_train.class_indices)


model.fit_generator(x_train, samples_per_epoch = 3100, epochs = 5, validation_data = x_test, nb_val_samples = 900)



model.save('gesture.h5')
