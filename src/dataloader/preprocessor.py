import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src import configurations as cfg


class Preprocessor:
    def __init__(self):
        pass

    def augmentation_cv2(self, dataset,
                     rotation_range=cfg.ROTATION_RANGE,
                     scale_range=cfg.SCALE_RANGE,
                     height_shift_range=cfg.HEIGHT_SHIFT_RANGE,
                     width_shift_range=cfg.WIDTH_SHIFT_RANGE,
                     dilate_range=cfg.DILATION_RANGE,
                     erode_range=cfg.ERODE_RANGE):
        print(type(dataset))
        print(dataset.keys())
        # images = dataset['image']
        images = dataset['image'].numpy()
        images = images.astype(np.float32)
        w, h, _ = images.shape

        dilate_kernel = np.ones((int(np.random.uniform(1, dilate_range)),), np.uint8)
        erode_kernel = np.ones((int(np.random.uniform(1, erode_range)),), np.uint8)
        height_shift = np.random.uniform(-height_shift_range, height_shift_range)
        rotation = np.random.uniform(-rotation_range, rotation_range)
        scale = np.random.uniform(1 - scale_range, 1)
        width_shift = np.random.uniform(-width_shift_range, width_shift_range)

        trans_map = np.float32([[1, 0, width_shift * w], [0, 1, height_shift * h]])
        rot_map = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)

        trans_map_aff = np.r_[trans_map, [[0, 0, 1]]]
        rot_map_aff = np.r_[rot_map, [[0, 0, 1]]]
        affine_mat = rot_map_aff.dot(trans_map_aff)[:2, :]

        for i in range(images.shape[2]):
            images[:, :, i] = cv2.warpAffine(images[:, :, i], affine_mat, (h, w), flags=cv2.INTER_NEAREST, borderValue=255)
            images[:, :, i] = cv2.erode(images[:, :, i], erode_kernel, iterations=1)
            images[:, :, i] = cv2.dilate(images[:, :, i], dilate_kernel, iterations=1)

        dataset['images']=images
        return dataset

    def augmentation(self, dataset,
                     rotation_range=cfg.ROTATION_RANGE,
                     scale_range=cfg.SCALE_RANGE,
                     height_shift_range=cfg.HEIGHT_SHIFT_RANGE,
                     width_shift_range=cfg.WIDTH_SHIFT_RANGE,
                     dilate_range=cfg.DILATION_RANGE,
                     erode_range=cfg.ERODE_RANGE):
        images = dataset['input']
        # images = dataset[0]
        images=tf.keras.preprocessing.image.apply_affine_transform(
            images, theta=0, tx=0, ty=0, shear=0, row_axis=1, col_axis=0,
            channel_axis=2, fill_mode='nearest', cval=0.0, order=1
        )
        images = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)(images)
        return (images, dataset['target'])
