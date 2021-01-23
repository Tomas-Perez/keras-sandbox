# Model for exam of 19/06/2020

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Dropout, Concatenate, Flatten

inputs = Input((32, 32, 1))
x = Conv2D(32, (5, 5), padding='same')(inputs)          # (32, 32, 32)  |   32 * ( 5 * 5 * 1 + 1) = 832
max_pool_1 = MaxPooling2D(pool_size=(2, 2))(x)          # (16, 16, 32)  |   0
x = Conv2D(64, (5, 5), padding='same')(max_pool_1)      # (16, 16, 64)  |   64 * ( 5 * 5 * 32 + 1) = 51264
max_pool_2 = MaxPooling2D(pool_size=(2, 2))(x)          # (8, 8, 64)    |   0
x = Conv2D(128, (5, 5), padding='same')(max_pool_2)     # (8, 8, 128)   |   128 * ( 5 * 5 * 64 + 1) = 204928
x = MaxPooling2D(pool_size=(2, 2))(x)                   # (4, 4, 128)   |   0
max_pool_1 = MaxPooling2D(pool_size=(4,4))(max_pool_1)  # (4, 4, 32)    |   0
max_pool_2 = MaxPooling2D(pool_size=(2,2))(max_pool_2)  # (4, 4, 64)    |   0
x = Concatenate()([x, max_pool_1, max_pool_2])          # (4, 4, 224)   |   0
x = Flatten()(x)                                        # (3584)        |   0
x = Dense(1024)(x)                                      # (1024)        |   1024 * (3584 + 1) = 3671040
output = Dense(43)(x)                                   # (43)          |   43 * (1024 + 1) = 44075

# total = 832 + 51264 + 204928 + 3671040 + 44075 = 3972139
model = Model(inputs=inputs, outputs=output)

model.summary()