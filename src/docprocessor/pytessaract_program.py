import os
import string

import cv2
import pytesseract
from src.dataloader.preprocessor import Preprocessor
from src.dataloader.tokenizer import Tokenizer
from src.network.model import HTRModel
from src import constants
import tensorflow as tf

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\g.shankar.behera\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

files_path = "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\MyCode\\data files\\datasets\\ocr_sample_files\\"
images_folder = os.path.join(files_path, "ocr_sample_images")
image_filename = "image0_0.png"
output_images = os.path.join(images_folder, "output_images")
os.makedirs(output_images, exist_ok=True)

image = cv2.imread(os.path.join(images_folder, image_filename))

details = pytesseract.image_to_data(image, lang='eng', output_type=pytesseract.Output.DATAFRAME)
# remove_indices = details[details['conf'] == -1].index
# for i in range(details.shape[0]):
#     if pd.isna(details.iloc[i]['text']):
#         remove_indices = remove_indices.append(pd.Index([i]))
#         continue
#     if len(details.iloc[i]['text'].split()) == 0:
#         remove_indices = remove_indices.append(pd.Index([i]))
# details.drop(remove_indices, inplace=True)
# details.reset_index(inplace=True)
total_boxes = len(details['text'])
# remove_indices = []
details.sort_values(['left', 'top'], ascending=[True, True], inplace=True)
model = HTRModel("flor", constants.INPUT_SIZE, constants.VOCABULARY_SIZE)
model.compile()
model.load_checkpoint(r'C:\Users\g.shankar.behera\My Files\Project\Code\HTR\output\word\flor\checkpoint_weights.hdf5')
preprocessor = Preprocessor()
tokenizer = Tokenizer(string.printable[:95])
image_bg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
for sequence_number in range(total_boxes):
    (x, y, w, h) = (
        details['left'][sequence_number], details['top'][sequence_number], details['width'][sequence_number],
        details['height'][sequence_number])
    # if w > 1500 or h > 1000 or w < 50 or h < 50:
        # if w < 50 or h < 50:
        # continue

    cropped = image_bg[y:y + h, x:x + w]
    cropped = cropped[..., tf.newaxis]
    try:
        cropped = preprocessor.preprocess(cropped)
        cropped = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)(cropped)
        cropped = cropped[tf.newaxis, ...]
        preds, _ = model.predict(cropped,
                             ctc_decode=True)
        predicts = [tokenizer.decode(predict[0]) for predict in preds]
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        image = cv2.putText(image, predicts[0], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        print(f"Predicted: {sequence_number}/{total_boxes} - {predicts[0]}")
    except ValueError as ve:
        print(ve)
        continue

    # plt.imshow(image)
    # plt.show()

cv2.imwrite(os.path.join(output_images, image_filename), image)
