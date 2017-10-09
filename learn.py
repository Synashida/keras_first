import data
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import SGD

X = data.x_train
Y = data.y_train

model = Sequential()
model.add(Dense(5120, input_dim=7))
model.add(Activation('relu'))
#model.add(Dense(2048, input_dim=5120))
#model.add(Activation('relu'))
#model.add(Dense(1024, input_dim=2048))
model.add(Activation('softmax')) 
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


from keras.callbacks import ModelCheckpoint
check = ModelCheckpoint("model.hdf5")
model.fit(X, Y, epochs=8, batch_size=1, verbose=1)

# 保存
model.save("model.h5")

model.metrics_names
print(model.evaluate(X, Y, batch_size=10))
