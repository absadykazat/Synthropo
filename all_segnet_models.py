import cv2
from efficientnet.keras import EfficientNetB0
from tensorflow.keras.layers import Conv2D, UpSampling2D, Conv2DTranspose, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.applications import ResNet50, VGG19
from keras.applications.resnet import ResNet101
import tensorflow as tf
from tensorflow.keras import backend as K

from keras.callbacks import EarlyStopping, TensorBoard


IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

# Input images that converted to numpy array with (n, 512, 512, 3) shape, n - number of images, 3 - number of channels
X = np.load('path/to/numpy/array.npy')
# Output images that converted to numpy array with (n, 512, 512, 1) shape using alpha channel, n - number of images
Y = np.load('path/to/numpy/array.npy')


input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)

effnet = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape)
resnet101 = ResNet101(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape)
resnet50 = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape)
vgg19 = VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape)

backbones = [(effnet, 'effnet'), (resnet101, 'resnet101'), (resnet50, 'resnet50'), (vgg19, 'vgg19')]


def segnet(backbone, backbone_name, num_classes=1):

    # Encoder
    encoder = backbone.output
    encoder = BatchNormalization()(encoder)

    # Decoder
    x = UpSampling2D(size=(2, 2))(encoder)
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

    # Output layer
    x = Conv2D(num_classes, (1, 1), activation='sigmoid')(x)

    # Create model
    model = Model(inputs=backbone.input, outputs=x, name='segnet_'+backbone_name)

    return model


def iou_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection

    iou = intersection / (union + tf.keras.backend.epsilon())
    loss = 1.0 - iou

    return loss


def train_model(model, model_name):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    print(model.name+' starts learning')
    callbacks = [EarlyStopping(patience=100, monitor='val_loss'), TensorBoard(log_dir='logs_pspnet/'+model_name)]
    model.fit(X, Y, validation_split=0.2, batch_size=8, epochs=100, callbacks=callbacks)
    model.save('models/segnet/'+model_name+'.h5')
    print(model_name+' model saved successfully!')
    K.clear_session()


for backbone, backbone_name in backbones:
    train_model(segnet(backbone, backbone_name), segnet(backbone, backbone_name).name)



