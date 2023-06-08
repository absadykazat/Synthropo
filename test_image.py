import tensorflow as tf
import numpy as np
from tensorflow.keras import backend, layers
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt


class FixedDropout(layers.Dropout):
        def _get_noise_shape(self, inputs):
            if self.noise_shape is None:
                return self.noise_shape

            symbolic_shape = backend.shape(inputs)
            noise_shape = [symbolic_shape[axis] if shape is None else shape
                           for axis, shape in enumerate(self.noise_shape)]
            return tuple(noise_shape)


print("Loading model...")
model = tf.keras.models.load_model('path/to/model.h5', custom_objects={'FixedDropout': FixedDropout(rate=0.2)})
print("Loaded!")

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

filepath = 'path/to/image.png'

X_train = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
img = imread(filepath)[:, :, :IMG_CHANNELS]
img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
X_train[0] = img

preds_train = model.predict(X_train[:], verbose=1)
preds_train_t = (preds_train > 0.5).astype(np.uint8)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(X_train[0])
axes[0].set_title('Input Image')
axes[1].imshow(np.squeeze(preds_train_t))
axes[1].set_title('Output Image')
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[1].set_xticks([])
axes[1].set_yticks([])

plt.show()