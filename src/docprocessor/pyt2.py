import os
import numpy as np

import pytesseract
from PIL import Image
import os
import string

import cv2
import matplotlib.pyplot as plt
import pytesseract
from pdf2image import convert_from_path
import pandas as pd
from src.dataloader.preprocessor import Preprocessor
from src.dataloader.tokenizer import Tokenizer
from src.network.model import HTRModel
from src import constants
import tensorflow as tf


def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


model = HTRModel("flor", constants.INPUT_SIZE, constants.VOCABULARY_SIZE)
model.compile()
model.load_checkpoint(r'C:\Users\g.shankar.behera\My Files\Project\Code\HTR\output\word\flor\checkpoint_weights.hdf5')
preprocessor = Preprocessor()
tokenizer = Tokenizer(string.printable[:95])


def preprocess(image_path):
    image = Image.open(image_path)
    image = image.convert("RGB")

    width, height = image.size
    w_scale = 1000 / width
    h_scale = 1000 / height
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\g.shankar.behera\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

    ocr_df = pytesseract.image_to_data(image, output_type='data.frame')
    ocr_df = ocr_df.dropna().assign(left_scaled=ocr_df.left * w_scale,
                                    width_scaled=ocr_df.width * w_scale,
                                    top_scaled=ocr_df.top * h_scale,
                                    height_scaled=ocr_df.height * h_scale,
                                    right_scaled=lambda x: x.left_scaled + x.width_scaled,
                                    bottom_scaled=lambda x: x.top_scaled + x.height_scaled)
    float_cols = ocr_df.select_dtypes('float').columns
    ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
    ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
    ocr_df = ocr_df.dropna().reset_index(drop=True)

    words = list(ocr_df.text)
    coordinates = ocr_df[['left', 'top', 'width', 'height']]
    actual_boxes = []
    img = np.array(image)
    img_copy = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new_img = img_copy.copy()

    results_df = pd.DataFrame(
        columns=['left', 'top', 'width', 'height', 'HTR Prob', 'HTR Text', 'Combo Prob', 'Combo Text'])

    for idx, row in coordinates.iterrows():
        x, y, w, h = tuple(row)  # the row comes in (left, top, width, height) format
        actual_box = [x, y, x + w, y + h]
        cropped = img_gray[y:y + h, x:x + w]
        cropped = cropped[..., tf.newaxis]
        try:
            # new_img = img_copy.copy()
            cropped = preprocessor.preprocess(cropped)
            cropped = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)(cropped)
            cropped = cropped[tf.newaxis, ...]
            preds, probabilities = model.predict(cropped,
                                                 ctc_decode=True)
            predicts = [tokenizer.decode(predict[0]) for predict in preds]
            text_to_write = predicts[0] if int(probabilities[0][0] * 100) > int(ocr_df.iloc[idx]['conf']) else \
            ocr_df.iloc[idx]['text']
            color = (0,0,255) if int(probabilities[0][0] * 100) > int(ocr_df.iloc[idx]['conf']) else \
                (0,255,0)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
            img = cv2.putText(img, text_to_write, (x, y), cv2.FONT_HERSHEY_PLAIN, 0.7, color, 1)
            new_img = cv2.rectangle(new_img, (x, y), (x + w, y + h), (255, 0, 0), 1)
            new_img = cv2.putText(new_img, predicts[0], (x, y), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 255), 1)
            print(f"Predicted: {idx}/{coordinates.shape[0]} - max: {text_to_write} - htr: {predicts[0]}")
            results_df.loc[idx] = (
            x, y, w, h, int(probabilities[0][0] * 100), predicts[0], int(ocr_df.iloc[idx]['conf']), ocr_df.iloc[idx]['text'])
            # cv2.imshow("new_img",new_img)
            # cv2.waitKey(0)
        except ValueError as ve:
            print(ve)
        actual_boxes.append(actual_box)
    boxes = []
    for box in actual_boxes:
        boxes.append(normalize_box(box, width, height))
    return image, words, boxes, actual_boxes, img, new_img, results_df


files_path = "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\MyCode\\data files\\datasets\\ocr_sample_files\\"
images_folder = os.path.join(files_path, "ocr_sample_images")
output_images = os.path.join(images_folder, "output_images")
output_csvs = os.path.join(images_folder, "output_new_csvs")
for image_filename in os.listdir(images_folder):
    if not image_filename.endswith(".jpg") and not image_filename.endswith(".png"):
        continue
    print(f"Processing {image_filename}")
    # image_filename = "invoice.jpg"
    image_path = os.path.join(images_folder, image_filename)
    image, words, boxes, actual_boxes, img, new_img, results_df = preprocess(image_path)
    # plt.imshow(img)
    # plt.show()
    cv2.imwrite(os.path.join(output_images, f"{image_filename.split('.')[0]}_max.{image_filename.split('.')[1]}"), img)
    cv2.imwrite(os.path.join(output_images, f"{image_filename.split('.')[0]}_htr.{image_filename.split('.')[1]}"),
                new_img)
    csv_filename = os.path.join(output_csvs, f"{image_filename.split('.')[0]}.csv")
    results_df.to_csv(csv_filename, index=False)
    print(f"done")
