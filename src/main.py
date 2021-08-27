import os
import numpy as np
import pandas as pd

arr = np.array([1, 2, 3])
print(arr)

image_dataset = "iam"
images_path = "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\MyCode\\data files\\datasets\\iam\\lines\\lines\\"
ground_truth_path = "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\MyCode\\data files\\datasets\\iam\\ascii\\lines.txt"

gt_list = []
img_paths = []
with open(ground_truth_path,'r') as gtp:
    for gt in gtp.readlines():
        if not gt.startswith("#"):
            gt_splitted=gt.split(" ")
            line_id=gt_splitted[0]
            folder=line_id.split("-")[0]
            subfolder=line_id.split("-")[1]
            file=line_id.split("-")[2]
            file_path=os.path.join(images_path, folder, f"{folder}-{subfolder}", f"{folder}-{subfolder}-{file}.png")
            word_segmentation=gt_splitted[1]
            gray_level=gt_splitted[2]
            num_components=gt_splitted[3]
            bb_x,bb_y,bb_w,bb_h=gt_splitted[4],gt_splitted[5],gt_splitted[6],gt_splitted[7]
            # if there's a space between two letters, then join them first, then split
            sentence=" ".join(" ".join(gt_splitted[8:]).split("|"))
            gt_list.append(sentence)
            img_paths.append(file_path)

