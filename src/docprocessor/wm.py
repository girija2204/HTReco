import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf


def process_image(path):
    img = cv2.imread(path)
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, (540, 420))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = np.reshape(img, (420, 540, 1))

    return img

model_location = "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\HTR Backups\\output\\Denoising Documents\\"
reconstructed_model = keras.models.load_model(os.path.join(model_location, "my_model.h5"))
files_path = "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\MyCode\\data files\\datasets\\ocr_sample_files\\"
images_folder = os.path.join(files_path, "ocr_sample_images")
for image_filename in os.listdir(images_folder):
# image_filename = "invoice2.jpg"
    if not image_filename.endswith("png") and not image_filename.endswith("jpg"):
        continue
    image = process_image(os.path.join(images_folder, image_filename))
    image = image[tf.newaxis, ...]
    Y_test = reconstructed_model.predict(image, batch_size=16)

    plt.figure(figsize=(7, 7))
    # for i in range(0, 2):
    plt.subplot(1, 2, 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(tf.reshape(image, [420, 540]), cmap='gray')
    # plt.title('Noise image: {}'.format(tf.reshape(image, [420, 540])))

    plt.subplot(1, 2, 2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(tf.reshape(Y_test, [420, 540]), cmap='gray')
    # plt.title('Denoised image: {}'.format(tf.reshape(Y_test, [420, 540])))
    plt.show()
