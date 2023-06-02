import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, AveragePooling2D, UpSampling2D, Concatenate, Activation
from efficientnet.keras import EfficientNetB0
from tensorflow.keras.applications import ResNet50, VGG19
from keras.applications.resnet import ResNet101
from tensorflow.keras import backend as K
import numpy as np
from keras.callbacks import EarlyStopping, TensorBoard

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

# Input images that converted to numpy array with (n, 512, 512, 3) shape, n - number of images, 3 - number of channels
X = np.load('path/to/numpy/array.npy')
# Output images that converted to numpy array with (n, 512, 512, 1) shape using alpha channel, n - number of images
Y = np.load('path/to/numpy/array.npy')

input_shape = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
num_classes = 1


def pspnet(backbone, num_classes=1):
    # Encoder
    encoder = backbone.output

    # Pyramid Pooling Module
    pool_sizes = [1, 2, 4, 8]
    pyramid_features = []
    for pool_size in pool_sizes:
        x = AveragePooling2D(pool_size=(pool_size, pool_size))(encoder)
        x = Conv2D(512, (1, 1), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((pool_size, pool_size))(x)
        pyramid_features.append(x)
    psp = Concatenate()(pyramid_features)
    psp = Concatenate()([psp, encoder])

    # Decoder
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(psp)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_classes, (1, 1), activation='sigmoid', padding='same')(x)

    # Create PSPNet model
    model = Model(inputs=backbone.input, outputs=x, name='PSPNet-EfficientNet')
    return model


def train_model(model, model_name):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    print(model_name.upper() + ' starts learning')
    print('*' * 50)
    callbacks = [EarlyStopping(patience=5, monitor='val_loss'), TensorBoard(log_dir='logs/' + model_name)]
    model.fit(X, Y, validation_split=0.2, batch_size=8, epochs=100, callbacks=callbacks)
    model.save('models/pspnet/' + model_name + '.h5')
    print('*' * 50)
    print(model_name.upper() + ' model saved successfully!')
    K.clear_session()


inputs = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)

effnet = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=None, input_shape=inputs)
resnet101 = ResNet101(include_top=False, weights='imagenet', input_tensor=None, input_shape=inputs)
resnet50 = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=inputs)
vgg19 = VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=inputs)

backbones = [(effnet, 'effnet'), (resnet101, 'resnet101'), (resnet50, 'resnet50'), (vgg19, 'vgg19')]

for backbone, backbone_name in backbones:
    train_model(pspnet(backbone), 'pspnet_' + backbone_name)