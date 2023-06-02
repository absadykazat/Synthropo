import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
from keras.models import Model
import keras
from efficientnet.keras import EfficientNetB0
from tensorflow.keras.applications import ResNet50, VGG19
from keras.applications.resnet import ResNet101
from tensorflow.keras import backend as K
from keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, Activation
from keras.callbacks import EarlyStopping, TensorBoard

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

# Input images that converted to numpy array with (n, 512, 512, 3) shape, n - number of images, 3 - number of channels
X = np.load('path/to/numpy/array.npy')
# Output images that converted to numpy array with (n, 512, 512, 1) shape using alpha channel, n - number of images
Y = np.load('path/to/numpy/array.npy')


def get_last_layer_with_shape(model, shape):
    last_layer = None

    for layer in model.layers:
        try:
            if layer.output.shape[1] == shape:
                last_layer = layer
        except:
            print("none")

    return last_layer


def iou_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection

    iou = intersection / (union + tf.keras.backend.epsilon())
    loss = 1.0 - iou

    return loss


def unet(backbone, num_classes=1):

    # Encoder
    encoder = backbone.output
    encoder = BatchNormalization()(encoder)

    # Decoder
    x = UpSampling2D(size=(2, 2))(encoder)
    x = keras.layers.concatenate([get_last_layer_with_shape(backbone, 32).output, x], axis=-1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = keras.layers.concatenate([get_last_layer_with_shape(backbone, 64).output, x], axis=-1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = keras.layers.concatenate([get_last_layer_with_shape(backbone, 128).output, x], axis=-1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = keras.layers.concatenate([get_last_layer_with_shape(backbone, 256).output, x], axis=-1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(16, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = keras.layers.concatenate([get_last_layer_with_shape(backbone, 512).output, x], axis=-1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Output layer
    x = Conv2D(num_classes, (1, 1), activation='sigmoid')(x)

    # Create model
    model = Model(inputs=backbone.input, outputs=x)

    return model


def train_model(model, model_name):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    print(model_name.upper() + ' starts learning')
    print('*' * 50)
    callbacks = [EarlyStopping(patience=5, monitor='val_loss'), TensorBoard(log_dir='logs_unet/' + model_name)]
    model.fit(X, Y, validation_split=0.2, batch_size=8, epochs=100, callbacks=callbacks)
    model.save('models/unet/' + model_name + '.h5')
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
    train_model(unet(backbone), 'unet_' + backbone_name)
