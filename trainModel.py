
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

#Loading model
data = np.load('savedData/data.npy')
target = np.load('savedData/target.npy')

#Partitioning data
trainData, testData, trainLabel, testLabel = train_test_split(data, target, test_size = 0.1)

#Building model
print('[Status] Building model...')
model = Sequential()

#First Convolutional Layer
model.add(Conv2D(200, (3, 3), input_shape = data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

#First Convolutional Layer
model.add(Conv2D(100, (3, 3), input_shape = data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

#Flattening to One-Dimensional Value
model.add(Flatten())
#Dropout to avoid overfitting
model.add(Dropout(0.5))

#Dense to 50 layers
model.add(Dense(50, activation = 'relu'))
#Dense to output layer (2 Categories)
model.add(Dense(2, activation = 'softmax'))

#Compiling model
print('[Status] Compiling model...')
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', 
	metrics = ['accuracy'])

#Adding checkpoint and saving best model
checkpoint = ModelCheckpoint('model/m-{epoch:03d}.model', monitor = 'val_loss',
	verbose = 0, save_best_only = True)

#Training model
history = model.fit(trainData, trainLabel, epochs = 10, 
	callbacks = [checkpoint], validation_split = 0.2)

