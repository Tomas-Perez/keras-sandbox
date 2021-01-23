# Model for exam of 15/07/2020

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Dropout, Concatenate, Flatten, Conv2DTranspose

blue_filter = (3, 3)

conv_params = {
    'padding': 'same',
    'activation': 'relu',
}

inputs = Input((256, 256, 3))                               #   (256, 256, 3)   |   0
x = Conv2D(32, blue_filter, **conv_params)(inputs)          #   (256, 256, 32)  |   32 * (3 * 3 * 3 + 1) = 896
skip_1 = Conv2D(32, blue_filter, padding='same')(x)         #   (256, 256, 32)  |   32 * (3 * 3 * 32 + 1) = 9248
x = MaxPooling2D(pool_size=(2,2))(skip_1)                   #   (128, 128, 32)  |   0
x = Conv2D(64, blue_filter, **conv_params)(x)               #   (128, 128, 64)  |   64 * (3 * 3 * 32 + 1) = 18496
skip_2 = Conv2D(64, blue_filter, **conv_params)(x)          #   (128, 128, 64)  |   64 * (3 * 3 * 64 + 1) = 36928
x = MaxPooling2D(pool_size=(2,2))(skip_2)                   #   (64, 64, 64)    |   0
x = Conv2D(128, blue_filter, **conv_params)(x)              #   (64, 64, 128)   |   128 * (3 * 3 * 64 + 1) = 73856
skip_3 = Conv2D(128, blue_filter, **conv_params)(x)         #   (64, 64, 128)   |   128 * (3 * 3 * 128 + 1) = 147584
x = MaxPooling2D(pool_size=(2,2))(skip_3)                   #   (32, 32, 128)   |   0
x = Conv2D(256, blue_filter, **conv_params)(x)              #   (32, 32, 256)   |   256 * (3 * 3 * 128 + 1) = 295168
skip_4 = Conv2D(256, blue_filter, **conv_params)(x)         #   (32, 32, 256)   |   256 * (3 * 3 * 256 + 1) = 590080
x = MaxPooling2D(pool_size=(2,2))(skip_4)                   #   (16, 16, 256)   |   0
x = Conv2D(512, blue_filter, **conv_params)(x)              #   (16, 16, 512)   |   512 * (3 * 3 * 256 + 1) = 1180160
skip_5 = Conv2D(512, blue_filter, **conv_params)(x)         #   (16, 16, 512)   |   512 * (3 * 3 * 512 + 1) = 2359808
x = MaxPooling2D(pool_size=(2,2))(skip_5)                   #   (8, 8, 512)     |   0
x = Conv2D(1024, blue_filter, **conv_params)(x)             #   (8, 8, 1024)    |   1024 * (3 * 3 * 512 + 1) = 4719616
x = Conv2D(1024, blue_filter, **conv_params)(x)             #   (8, 8, 1024)    |   1024 * (3 * 3 * 1024 + 1) = 9438208

x = Conv2DTranspose(512, blue_filter, strides=(2, 2), padding='same')(x)    #   (16, 16, 512)   |   512 * (3 * 3 * 1024 + 1) = 4719104
x = Concatenate()([skip_5, x])                                              #   (16, 16, 1024)  |   0
x = Conv2D(512, blue_filter, **conv_params)(x)                              #   (16, 16, 512)   |   512 * (3 * 3 * 1024 + 1) = 4719104
x = Conv2D(512, blue_filter, **conv_params)(x)                              #   (16, 16, 512)   |   512 * (3 * 3 * 512 + 1) = 2359808
x = Conv2DTranspose(256, blue_filter, strides=(2, 2), padding='same')(x)    #   (32, 32, 256)   |   256 * (3 * 3 * 512 + 1) = 1179904
x = Concatenate()([skip_4, x])                                              #   (32, 32, 512)   |   0
x = Conv2D(256, blue_filter, **conv_params)(x)                              #   (32, 32, 256)   |   256 * (3 * 3 * 512 + 1) = 1179904
x = Conv2D(256, blue_filter, **conv_params)(x)                              #   (32, 32, 256)   |   256 * (3 * 3 * 256 + 1) = 590080
x = Conv2DTranspose(128, blue_filter, strides=(2, 2), padding='same')(x)    #   (64, 64, 128)   |   128 * (3 * 3 * 256 + 1) = 295040
x = Concatenate()([skip_3, x])                                              #   (64, 64, 256)   |   0
x = Conv2D(128, blue_filter, **conv_params)(x)                              #   (64, 64, 128)   |   128 * (3 * 3 * 256 + 1) = 295040
x = Conv2D(128, blue_filter, **conv_params)(x)                              #   (64, 64, 128)   |   128 * (3 * 3 * 128 + 1) = 147584
x = Conv2DTranspose(64, blue_filter, strides=(2, 2), padding='same')(x)     #   (128, 128, 64)  |   64 * (3 * 3 * 128 + 1) = 73792
x = Concatenate()([skip_2, x])                                              #   (128, 128, 128) |   0
x = Conv2D(64, blue_filter, **conv_params)(x)                               #   (128, 128, 64)  |   64 * (3 * 3 * 128 + 1) = 73792
x = Conv2D(64, blue_filter, **conv_params)(x)                               #   (128, 128, 64)  |   64 * (3 * 3 * 64 + 1) = 36928
x = Conv2DTranspose(32, blue_filter, strides=(2, 2), padding='same')(x)     #   (256, 256, 32)  |   32 * (3 * 3 * 64 + 1) = 18464
x = Concatenate()([skip_1, x])                                              #   (256, 256, 64)  |   0
x = Conv2D(32, blue_filter, **conv_params)(x)                               #   (256, 256, 32)  |   32 * (3 * 3 * 64 + 1) = 18464
x = Conv2D(32, blue_filter, **conv_params)(x)                               #   (256, 256, 32)  |   32 * (3 * 3 * 32 + 1) = 9248
output = Conv2D(2, (1, 1), activation='softmax', padding='same')(x)         #   (256, 256, 2)   |   2 * (1 * 1 * 32 + 1) = 66

# total = 896 + 9248 + 18496 + 36928 + 73856 + 147584 + 295168 + 590080 + 1180160 + 2359808 + 4719616 + 9438208 + 4719104 + 4719104 + 2359808 + 1179904 + 1179904 + 590080 + 295040 + 295040 + 147584 + 73792 + 73792 + 36928 + 18464 + 18464 + 9248 + 66 = 34,586,370

model = Model(inputs=inputs, outputs=output)

model.summary()