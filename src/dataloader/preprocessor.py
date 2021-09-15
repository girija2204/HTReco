import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from src import constants


class Preprocessor:
    def __init__(self):
        self.random_invert = self.random_invert()

    @tf.function
    def random_invert_img(self, x, p=0.5):
        if tf.random.uniform([]) < p:
            x = (255 - x)
        else:
            x = x
        return x

    def random_invert(self, factor=0.5):
        return layers.Lambda(lambda x: self.random_invert_img(x, factor))

    def augmentation(self, dataset,
                     rotation_range=constants.ROTATION_RANGE,
                     scale_range=constants.SCALE_RANGE,
                     height_shift_range=constants.HEIGHT_SHIFT_RANGE,
                     width_shift_range=constants.WIDTH_SHIFT_RANGE,
                     dilate_range=constants.DILATION_RANGE,
                     erode_range=constants.ERODE_RANGE):
        images = dataset['input']
        images = tf.keras.preprocessing.image.apply_affine_transform(
            images, theta=0, tx=0, ty=0, shear=0, row_axis=1, col_axis=0,
            channel_axis=2, fill_mode='nearest', cval=0.0, order=1
        )

        # new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
        # images = tf.image.stateless_random_brightness(
        #     images, max_delta=0.5, seed=new_seed)
        # images = tf.image.stateless_random_contrast(
        #     images, lower=0.2, upper=0.6, seed=new_seed)
        # images = self.random_invert(images)
        images = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)(images)
        return (images, dataset['target'])

    def preprocess(self, image):
        image = tf.squeeze(image)
        unique_vals, indices = tf.unique(tf.reshape(image, [-1]))
        background = int(unique_vals[np.argmax(np.bincount(indices))])
        wt, ht, _ = constants.INPUT_SIZE
        h, w = np.asarray(image).shape
        f = max((w / wt), (h / ht))

        new_size = [max(min(ht, int(h / f)), 1), max(min(wt, int(w / f)), 1)]
        image = image[tf.newaxis, ..., tf.newaxis]
        image = tf.image.resize(image, new_size)[0, ..., 0].numpy()

        target = np.ones([ht, wt], dtype=np.uint8) * background
        target[0:new_size[0], 0:new_size[1]] = image
        target = target[..., tf.newaxis]
        img = tf.image.transpose(target)
        return img
