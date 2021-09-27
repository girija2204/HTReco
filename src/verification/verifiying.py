import pandas as pd
# from pdf2image import convert_from_path
import cv2
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# base_path = "..//docreader-models//test_files//"
# images_folder = "analysis//target//"
# csv_filename = "prediction//invoice//MicrosoftTeams-image (2)//final_output_20210909145456.csv"
# new_csv_filename = csv_filename.split(".")[0] + "_updated." + csv_filename.split(".")[1]

base_path = "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\MyCode\\data files\\datasets\\ocr_sample_files\\ocr_sample_images\\"
images_folder = "output_images\\target\\"
csv_folder = "output_csvs\\target\\"
csv_filename = "image2_0.csv"
csv_filepath = os.path.join(base_path, csv_folder, csv_filename)
new_csv_filename = csv_filename.split(".")[0] + "_updated." + csv_filename.split(".")[1]


def read_ocroutput(file):
    dataset = pd.read_csv(file)
    dataset.sort_values(['top', 'left'], ascending=[True, True],
                        inplace=True, ignore_index=True)
    duplicate_indices = [v for k, v in dataset.groupby(['top', 'left', 'width', 'height']).indices.items() if
                         len(v) > 1]
    delete_indices = []
    for di in duplicate_indices:
        text_to_replace = dataset.loc[di].text.str.cat()
        dataset.loc[di[0], 'text'] = text_to_replace
        delete_indices.extend(di[1:].tolist())
    dataset.drop(delete_indices, axis=0, inplace=True)
    if 'Actual' not in dataset.columns: dataset['Actual'] = np.nan
    if 'Expected' not in dataset.columns: dataset['Expected'] = np.nan
    if 'Correctness' not in dataset.columns: dataset['Correctness'] = np.nan
    dataset.to_csv(base_path + csv_folder + new_csv_filename, index=False)
    return dataset


# def convert_pdf_to_image():
#     images = convert_from_path(base_path + pdf_filename, 500, poppler_path=r'C:\Program Files\poppler-0.68.0_x86'
#                                                                             r'\poppler-0.68.0\bin')
#     for i, image in enumerate(images):
#         fname = 'image' + str(i) + '.png'
#         image.save(base_path + images_folder + fname, "PNG")


images = os.listdir(base_path + images_folder)
if not Path(base_path + csv_folder + new_csv_filename).exists():
    _ = read_ocroutput(base_path + csv_folder + csv_filename)

csv_filename = new_csv_filename
dataset = pd.read_csv(base_path + csv_folder + csv_filename)
record_index = 0
for i in range(dataset.shape[0]):
    if dataset.loc[i, 'Actual'] is np.nan:
        record_index = i
        break
current_image = cv2.imread(base_path + images_folder + images[0])
# factor = 3508 / current_image.shape[0]
# rec_image = np.zeros_like(current_image)
copy_image = current_image.copy()
for ri in range(dataset.shape[0]):
    left, top, width, height = dataset.loc[ri, 'left'], dataset.loc[ri, 'top'], dataset.loc[
        ri, 'width'], dataset.loc[ri, 'height']
    scale = 1
    rec_image = current_image.copy()
    rec_image = cv2.rectangle(rec_image, (left, top), (left + width, top + height), (255, 0, 0), 1)
    rec_image = cv2.putText(rec_image, f"{ri}_{dataset.loc[ri, 'Combo Text']}",
                            (int(left), int(top)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0, 0, 255),
                            4)
    copy_image = cv2.putText(copy_image, f"{ri}_{dataset.loc[ri, 'Combo Text']}",
                             (int(left), int(top)),
                             cv2.FONT_HERSHEY_SIMPLEX,
                             1,
                             (0, 0, 200),
                             1)
    rec_image = cv2.resize(rec_image, (0, 0), fx=0.17, fy=0.17)
    # if ri > 372:
    # cv2.imshow('sgfsg', rec_image)
    # cv2.waitKey(0)
    del rec_image
copy_image = cv2.resize(copy_image, (0, 0), fx=0.4, fy=0.4)
cv2.imshow('sgfsg', copy_image)
cv2.waitKey(0)
plt.imshow(copy_image)
plt.axis("off")
plt.show()
cv2.imwrite(os.path.join(base_path, images_folder, f"{images[0].split('.')[0]}_inter.{images[0].split('.')[1]}"),
            copy_image)


def update_dataset(dataset, identifiers, correctness):
    print('updating dataset')
    if identifiers is None and correctness == 'fully':
        for record_index in range(dataset.shape[0]):
            if pd.isna(dataset.loc[record_index, 'Correctness']):
                dataset.loc[record_index, 'Expected'] = dataset.loc[record_index, 'Combo Text']
                dataset.loc[record_index, 'Actual'] = dataset.loc[record_index, 'Combo Text']
                dataset.loc[record_index, 'Correctness'] = 'fully'
        return dataset
    for identifier in identifiers.split():
        if identifier.startswith("w"):
            dataset.loc[int(identifier.split("-")[0][1:]), 'Expected'] = identifier.split("-")[1]
            dataset.loc[int(identifier.split("-")[0][1:]), 'Actual'] = dataset.loc[
                int(identifier.split("-")[0][1:]), 'Combo Text']
        else:
            break
        dataset.loc[int(identifier.split("-")[0][1:]), 'Correctness'] = correctness
    return dataset


identifiers = input("List all the Partial Incorrect words...")
dataset = update_dataset(dataset, identifiers, 'partial')
identifiers = input("List all the Fully Incorrect words...")
dataset = update_dataset(dataset, identifiers, 'incorrect')
dataset = update_dataset(dataset, None, 'fully')
missing_words = input("List all the missing words...")
for word in missing_words.split():
    new_row_index = dataset.shape[0]
    dataset.loc[new_row_index, 'text'] = word
    dataset.loc[new_row_index, 'Correctness'] = 'missing'
print(f"dataset updated. Saving into {base_path + csv_folder + csv_filename}")
dataset.to_csv(base_path + csv_folder + csv_filename, index=False)
