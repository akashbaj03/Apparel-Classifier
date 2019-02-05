## Import all the necessary packages ##

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from keras.datasets import mnist
import os
import numpy as np
from keras.preprocessing import image

## This code will build a deep convolutional neural network which will classify the images in categories ##

classifier = Sequential()

classifier.add(Conv2D(16, (3, 3), input_shape = (28, 28, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())

classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 10, activation = 'softmax'))

classifier.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('D:/Apparel/training_set',
target_size = (28, 28),
batch_size = 32,
class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('D:/Apparel/validation_set',
target_size = (28, 28),
batch_size = 32,
class_mode = 'categorical')

classifier.fit_generator(training_set,
steps_per_epoch = 8000,
epochs = 5,
validation_data = test_set,
validation_steps = 2000)

#Epoch 1/5
#8000/8000 [==============================] - 2688s 336ms/step - loss: 0.2515 - acc: 0.9089 - val_loss: 0.2615 - val_acc: 0.9125
#Epoch 2/5
#8000/8000 [==============================] - 1565s 196ms/step - loss: 0.1046 - acc: 0.9609 - val_loss: 0.3500 - val_acc: 0.9098
#Epoch 3/5
#8000/8000 [==============================] - 1596s 200ms/step - loss: 0.0471 - acc: 0.9831 - val_loss: 0.4701 - val_acc: 0.9071
#Epoch 4/5
#8000/8000 [==============================] - 1514s 189ms/step - loss: 0.0266 - acc: 0.9907 - val_loss: 0.5618 - val_acc: 0.9075
#Epoch 5/5
#8000/8000 [==============================] - 1311s 164ms/step - loss: 0.0187 - acc: 0.9936 - val_loss: 0.6324 - val_acc: 0.9022
