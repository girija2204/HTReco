import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from src.dataloader.preprocessor import Preprocessor
from src.dataloader.tokenizer import Tokenizer
from src.network.model import HTRModel
from src import constants
import tensorflow as tf
import string

image_file = r'C:\Users\g.shankar.behera\My Files\Project\Code\MyCode\data files\datasets\ocr_sample_files\ocr_sample_images\image0.png'
img = cv2.imread(image_file)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(gray, 0, 255,
                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 1))
detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(img, [c], -1, (255, 255, 255), 2)

repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))
result = 255 - cv2.morphologyEx(255 - img,
                                cv2.MORPH_CLOSE,
                                repair_kernel, iterations=1)

gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,
                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 99))
detected_vlines = cv2.morphologyEx(thresh,
                                   cv2.MORPH_OPEN,
                                   vertical_kernel, iterations=2)

cnts = cv2.findContours(detected_vlines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(img, [c], -1, (255, 255, 255), 2)

repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
result = 255 - cv2.morphologyEx(255 - img,
                                cv2.MORPH_CLOSE,
                                repair_kernel, iterations=1)

gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,
                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

rect_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 1))
dilation = cv2.dilate(thresh, rect_kernel, iterations=1)

contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

im2 = cv2.imread(image_file)
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

model = HTRModel("flor", constants.INPUT_SIZE, constants.VOCABULARY_SIZE)
model.compile()
model.load_checkpoint(r'C:\Users\g.shankar.behera\My Files\Project\Code\HTR\output\word\flor\checkpoint_weights.hdf5')
preprocessor = Preprocessor()
tokenizer = Tokenizer(string.printable[:95])
minContourSize = 250


# def doContours():
#     # create a copy of the image (needed for use with trackbar)
#     res = im2.copy()
#     # find contours - external only
#     countours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # create an empty mask
#     mask = np.zeros(im2.shape[:2], dtype=np.uint8)
#     # draw filled boundingrects if the contour is large enough
#     for c in countours:
#         if cv2.contourArea(c) > minContourSize:
#             x, y, w, h = cv2.boundingRect(c)
#             cv2.rectangle(mask, (x, y), (x + w, y + h), (255,0,0), -1)
#     cv2.imshow("Mask", mask)
#
#     # find the contours on the mask (with solid drawn shapes) and draw outline on input image
#     countours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for c in countours:
#         cv2.drawContours(res, [c], 0, (0, 255, 0), 2)
#     # show image
#     cv2.imshow("Contour", res)
# doContours()
def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]


contours.sort(key=lambda x: get_contour_precedence(x, img.shape[1]))
for cnt in contours:
    if cv2.contourArea(cnt) > minContourSize:
        x, y, w, h = cv2.boundingRect(cnt)
        im2 = cv2.rectangle(im2, (x, y), (x + w, y + h), (255, 0, 0), 4)
        # im2=cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
        cropped = im2[y:y + h, x:x + w]
        # plt.imshow(im2)
        # plt.show()
        cropped = cropped[..., tf.newaxis]
        cropped = preprocessor.preprocess(cropped)
        cropped = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)(cropped)
        cropped = cropped[tf.newaxis, ...]
        preds, _ = model.predict(cropped,
                                 ctc_decode=True)
        predicts = [tokenizer.decode(predict[0]) for predict in preds]
        print("predicts")
#
# cv2.imshow('final', im2)
plt.figure(figsize=(11, 11))
plt.imshow(im2)
plt.show()
