import os

from data.datastore import Datastore, DataGenerator

image_dataset = "iam"
images_path = "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\HTR\\data\\raw\\iam\\lines\\"
ground_truth_path = "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\HTR\\data\\raw\\iam\\ascii\\lines.txt "
partition_criteria = "C:\\Users\\g.shankar.behera\\My " \
                     "Files\\Project\\Code\\HTR\\data\\raw\\iam\\largeWriterIndependentTextLineRecognitionTask\\"
partitions = {'train': "trainset_new.txt", 'validation': "validationset_new.txt", 'test': "testset_new.txt"}

gt_dict = {}
img_paths = {}
with open(ground_truth_path, 'r') as gtp:
    for gt in gtp.read().splitlines():
        if not gt.startswith("#"):
            gt_splitted = gt.split(" ")
            line_id = gt_splitted[0]
            # if there's a space between two letters, then join them first, then split
            sentence = " ".join(" ".join(gt_splitted[8:]).split("|"))
            gt_dict[line_id] = sentence

datastore = Datastore(partitions=partitions)
for partition_key, partition_value in partitions.items():
    with open(os.path.join(partition_criteria, partition_value), 'r') as pc:
        for line_id in pc.read().splitlines():
            folder = line_id.split("-")[0]
            subfolder = line_id.split("-")[1]
            file = line_id.split("-")[2]
            file_path = os.path.join(images_path, folder, f"{folder}-{subfolder}", f"{folder}-{subfolder}-{file}.png")
            datastore.dataset[partition_key]['ground_truth'].append(gt_dict[line_id])
            datastore.dataset[partition_key]['file_path'].append(file_path)

datastore.save()

# dtgen=DataGenerator(partitions)
# tr_batch=dtgen.generate_train_batch()
# import tensorflow as tf
#
# raw_example=next(iter(tr_batch))
# parsed=raw_example.numpy()
# print("hello")

# import os
#
# a_list = [] path_a = "C:\\Users\\g.shankar.behera\\My
# Files\\Project\\Code\\HTR\\data\\raw\\iam\\largeWriterIndependentTextLineRecognitionTask\\" part_files =
# os.listdir(path_a) for part_file in part_files: a_list = [] with open(os.path.join(path_a, part_file),
# 'r') as file: lines = file.read().splitlines() for line in lines: if line.startswith("a"): a_list.append(line+"\n")
# if len(a_list) > 0: with open( os.path.join(path_a, f"{part_file.split('.')[0]}_new.{part_file.split('.')[1]}"),
# 'w') as writer: writer.writelines(a_list)

print("hello hi")
