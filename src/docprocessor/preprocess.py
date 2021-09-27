from PIL import Image
import cv2
import os
import math

base_path = "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\MyCode\\data files\\datasets\\ocr_sample_files\\ocr_sample_images\\ocr tool comparison\\original images\\"
image_filename = "image0_0.png"
new_image_filename = f"{image_filename.split('.')[0]}_resized.png"


def resize(image):
    new_height = 1400
    new_width = math.ceil((new_height / image.size[1]) * image.size[0])
    image_resized = image.resize((new_width, new_height), Image.LANCZOS)
    image_resized.save(os.path.join(base_path, new_image_filename))
    return image_resized


image = Image.open(os.path.join(base_path, image_filename))
image_resized = resize(image)
resize(image)
print(image.size)
print(image_resized.size)
