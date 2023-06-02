import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
from keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Concatenate, AveragePooling2D, UpSampling2D, \
	Activation, ReLU, SeparableConv2D, Add, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from efficientnet.keras import EfficientNetB0
from tensorflow.keras.applications import ResNet50, VGG19
from keras.applications.resnet import ResNet101
from tensorflow.keras import backend as K

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

# Input images that converted to numpy array with (n, 512, 512, 3) shape, n - number of images, 3 - number of channels
X = np.load('path/to/numpy/array.npy')
# Output images that converted to numpy array with (n, 512, 512, 1) shape using alpha channel, n - number of images
Y = np.load('path/to/numpy/array.npy')


def get_last_layer_with_shape(model, shape):
	model.summary()
	last_layer = None

	# Loop through the layers of the model
	for layer in model.layers:
		try:
			if layer.output.shape[1] == shape:
				last_layer = layer
		except:
			print("none")

	print(last_layer.output.shape)
	return last_layer


def aspp(inputs):
	shape = inputs.shape
	y_pool = AveragePooling2D(pool_size=(shape[1], shape[2]))(inputs)
	y_pool = Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(y_pool)
	y_pool = BatchNormalization()(y_pool)
	y_pool = Activation('relu')(y_pool)
	y_pool = UpSampling2D((shape[1], shape[2]), interpolation='bilinear')(y_pool)

	y_1 = Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(inputs)
	y_1 = BatchNormalization()(y_1)
	y_1 = Activation('relu')(y_1)

	y_6 = Conv2D(filters=256, kernel_size=1, dilation_rate=6, padding='same', use_bias=False)(inputs)
	y_6 = BatchNormalization()(y_6)
	y_6 = Activation('relu')(y_6)

	y_12 = Conv2D(filters=256, kernel_size=1, dilation_rate=12, padding='same', use_bias=False)(inputs)
	y_12 = BatchNormalization()(y_12)
	y_12 = Activation('relu')(y_12)

	y_18 = Conv2D(filters=256, kernel_size=1, dilation_rate=18, padding='same', use_bias=False)(inputs)
	y_18 = BatchNormalization()(y_18)
	y_18 = Activation('relu')(y_18)

	y = Concatenate()([y_pool, y_1, y_6, y_12, y_18])

	y = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same', use_bias=False)(y)
	y = BatchNormalization()(y)
	y = Activation('relu')(y)

	return y


def deeplab(base, shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)):
	# Input
	image_features = base.output
	x_a = aspp(image_features)
	x_a = UpSampling2D((4, 4), interpolation='bilinear')(x_a)

	x_b = get_last_layer_with_shape(base, 64)
	x_b = Conv2D(filters=48, kernel_size=1, dilation_rate=18, padding='same', use_bias=False)(x_b.output)
	x_b = BatchNormalization()(x_b)
	x_b = Activation('relu')(x_b)

	x = Concatenate()([x_a, x_b])

	x = Conv2D(filters=128, kernel_size=3, dilation_rate=18, padding='same', use_bias=False)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = Conv2D(filters=64, kernel_size=3, dilation_rate=18, padding='same', use_bias=False)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = UpSampling2D((4, 4), interpolation='bilinear')(x)
	x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	x = UpSampling2D(size=(2, 2))(x)
	x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	""" Outputs """
	x = Conv2D(1, (1, 1), name='output_layer')(x)
	x = Activation('sigmoid')(x)

	""" Model """
	model = Model(inputs=base.input, outputs=x)
	return model


def train_model(model, model_name):
	optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
	model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
	print(model_name.upper() + ' starts learning')
	print('*' * 50)
	callbacks = [EarlyStopping(patience=5, monitor='val_loss'), TensorBoard(log_dir='logs/' + model_name)]
	model.fit(X, Y, validation_split=0.2, batch_size=8, epochs=50, callbacks=callbacks)
	model.save('models/deeplabv3/' + model_name + '.h5')
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
	train_model(deeplab(backbone), 'deeplab_' + backbone_name)
