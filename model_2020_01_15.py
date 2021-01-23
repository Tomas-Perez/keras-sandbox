# Model for exam of 15/01/2020

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Dropout

patchsize = 128
pool_size = (2,2)
kernel_size = (3,3)
nb_filters = 8
num_channels = 3

model = Sequential()
model.add(Conv2D(nb_filters, (3,3) , input_shape=(patchsize, patchsize, num_channels), padding = "same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Conv2D(nb_filters*2, (3,3), padding = "same"))
model.add(Activation('relu'))
model.add(Conv2D(nb_filters*2, (3,3), padding = "same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(Conv2D(nb_filters*3, (5,5), padding = "same"))
model.add(Activation('relu'))
model.add(Conv2D(nb_filters*3, (5,5), padding = "same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = pool_size))
model.add(GlobalAveragePooling2D())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation('softmax'))


"""
---------------------------------------------------------------------------
Layer (type)                        Output Shape                    Param #
===========================================================================
conv2d_1 (Conv2D)                   (None, 64, 64, 8)               8 * ((3 * 3 * 3) + 1) = 224
---------------------------------------------------------------------------
activation_1 (Activation)           (None, 64, 64, 8)               0
---------------------------------------------------------------------------
max_pooling2d_1 (MaxPooling2D)      (None, 32, 32, 8)               0
---------------------------------------------------------------------------
conv2d_2 (Conv2D)                   (None, 32, 32, 16)              16 * ((3 * 3 * 8) + 1) = 1168
---------------------------------------------------------------------------
activation_2 (Activation)           (None, 32, 32, 16)              0
---------------------------------------------------------------------------
conv2d_3 (Conv2D)                   (None, 32, 32, 16)              16 * ((3 * 3 * 16) + 1) = 2320
---------------------------------------------------------------------------
activation_3 (Activation)           (None, 32, 32, 16)              0
---------------------------------------------------------------------------
max_pooling2d_2 (MaxPooling2D)      (None, 16, 16, 16)              0
---------------------------------------------------------------------------
conv2d_4 (Conv2D)                   (None, 16, 16, 24)              24 * ((5 * 5 * 16) + 1) = 9624
---------------------------------------------------------------------------
activation_4 (Activation)           (None, 16, 16, 24)              0
---------------------------------------------------------------------------
conv2d_5 (Conv2D)                   (None, 16, 16, 24)              24 * ((5 * 5 * 24) + 1) = 14424
---------------------------------------------------------------------------
activation_5 (Activation)           (None, 16, 16, 24)              0
---------------------------------------------------------------------------
max_pooling2d_3 (MaxPooling2D)      (None, 8, 8, 24)                0
---------------------------------------------------------------------------
global_average_pooling2d_1          (None, 24)                      0
---------------------------------------------------------------------------
dense_1 (Dense)                     (None, 64)                      64 * (24 + 1) = 1600
---------------------------------------------------------------------------
activation_6 (Activation)           (None, 64)                      0
---------------------------------------------------------------------------
dropout_1 (Dropout)                 (None, 64)                      0
---------------------------------------------------------------------------
dense_2 (Dense)                     (None, 16)                      16 * (64 + 1) = 1040
---------------------------------------------------------------------------
activation_7 (Activation)           (None, 16)                      0
---------------------------------------------------------------------------
dropout_2 (Dropout)                 (None, 16)                      0
---------------------------------------------------------------------------
dense_3 (Dense)                     (None, 5)                       5 * (16 + 1) = 85
---------------------------------------------------------------------------
activation_8 (Activation)           (None, 5)                       0
===========================================================================
Total params: 30485
---------------------------------------------------------------------------
"""

model.summary()