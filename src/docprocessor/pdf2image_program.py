import os

import pytesseract
from pdf2image import convert_from_path

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\g.shankar.behera\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

files_path = "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\MyCode\\data files\\datasets\\ocr_sample_files\\"
pdf_files = "ocr_sample_files"
filename = "4c70cdfb-f126-46c1-8175-e039966bc3af.pdf"
images_folder = os.path.join(files_path, "ocr_sample_images")


def convert_pdf_to_image():
    for pdf_index, filename in enumerate(os.listdir(os.path.join(files_path, pdf_files))):
        images = convert_from_path(os.path.join(files_path, pdf_files, filename),
                                   500,
                                   poppler_path=r'C:\Users\g.shankar.behera\Downloads\poppler-0.68.0_x86\poppler-0.68.0\bin')
        for i, image in enumerate(images):
            fname = 'image' + str(pdf_index) + "_" + str(i) + '.png'
            image.save(os.path.join(images_folder, fname), "PNG")


convert_pdf_to_image()
