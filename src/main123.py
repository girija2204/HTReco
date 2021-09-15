import os
import random
import numpy as np

# from src.dataloader.datastore import Datastore

#
# images_path = "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\MyCode\\data files\\datasets\\iam\\words\\"
# gt_path = "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\MyCode\\data files\\datasets\\iam\\ascii\\words.txt"
# partition_criteria_path = "C:\\Users\\g.shankar.behera\\My " \
#                           "Files\\Project\\Code\\MyCode\\data files\\datasets\\iam\\largeWriterIndependentTextLineRecognitionTask\\"
# partitions = {'train': "trainset.txt", 'validation': "validationset1.txt", 'test': "testset.txt"}
# datastore=Datastore(partitions)
# gt=datastore.read_ground_truth(gt_path)
# print(gt)

# output_partition_path="C:\\Users\\g.shankar.behera\\My " \
#                           "Files\\Project\\Code\\MyCode\\data files\\datasets\\iam\\largeWriterIndependentTextLineRecognitionTask\\words\\"

# for partition_key, partition_value in partitions.items():
#     with open(os.path.join(partition_criteria_path, partition_value), 'r') as pc:
#         f = open(output_partition_path + partition_value,'w')
#         for line_id in pc.read().splitlines():
#             folder = line_id.split("-")[0]
#             subfolder = line_id.split("-")[1]
#             file = line_id.split("-")[2]
#             file_path = os.path.join(images_path, folder, f"{folder}-{subfolder}",
#                                      f"{folder}-{subfolder}-{file}.png")
#             list_images=os.listdir(os.path.join(images_path, folder, f"{folder}-{subfolder}"))
#             for img in list_images:
#                 if img.startswith(line_id):
#                     f.write(img.split(".")[0]+"\n")
#             print("hello")

# images_path = "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\MyCode\\data files\\datasets\\breta\\processed\\breta\\words_gaplines\\"
# partition_location = "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\MyCode\\data files\\datasets\\breta\\processed\\breta\\partitions\\"
# os.makedirs(partition_location, exist_ok=True)
# images=[file for file in os.listdir(images_path) if file.endswith(".png")]
# val_size=int(len(images)*0.1)
# test_size=int(len(images)*0.2)
# val_data=list(np.random.choice(np.arange(0,len(images)), val_size, replace=False))
# for index, data in enumerate(val_data):
#     val_data[index] = images[data]
# for data in val_data:
#     images.remove(data)
# test_data=list(np.random.choice(np.arange(0,len(images)), test_size, replace=False))
# for index, data in enumerate(test_data):
#     test_data[index] = images[data]
# for data in test_data:
#     images.remove(data)
# for image in images:
#     label=image.split("_")[0]
# with open(os.path.join(partition_location, "val.txt"), "w") as val_writer:
#     val_writer.writelines(line + '\n' for line in val_data)
# with open(os.path.join(partition_location, "test.txt"), "w") as test_writer:
#     test_writer.writelines(line + '\n' for line in test_data)
# with open(os.path.join(partition_location, "train.txt"), "w") as train_writer:
#     train_writer.writelines(line + '\n' for line in images)
# print("hello")

images_path = "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\MyCode\\data files\\datasets\\cvl\\trainset\\words\\"
test_images_path = "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\MyCode\\data files\\datasets\\cvl\\testset\\words\\"
partition_location = "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\MyCode\\data files\\datasets\\cvl\\trainset\\partitions\\"
os.makedirs(partition_location, exist_ok=True)
folders = os.listdir(images_path)
test_folders = os.listdir(test_images_path)
all_images = []
for folder in folders:
    images = os.listdir(os.path.join(images_path, folder))
    for image in images:
        all_images.append(os.path.join(os.path.join(os.path.join(images_path, folder)), image))
val_size = int(len(all_images) * 0.1)
val_data = list(np.random.choice(np.arange(0, len(all_images)), val_size, replace=False))
for index, data in enumerate(val_data):
    val_data[index] = all_images[data]
for data in val_data:
    all_images.remove(data)
all_test_images = []
for folder in test_folders:
    images = os.listdir(os.path.join(test_images_path, folder))
    for image in images:
        all_test_images.append(os.path.join(os.path.join(os.path.join(test_images_path, folder)), image))
with open(os.path.join(partition_location, "val.txt"), "w") as val_writer:
    val_writer.writelines(line + '\n' for line in val_data)
with open(os.path.join(partition_location, "test.txt"), "w") as test_writer:
    test_writer.writelines(line + '\n' for line in all_test_images)
with open(os.path.join(partition_location, "train.txt"), "w") as train_writer:
    train_writer.writelines(line + '\n' for line in all_images)
print("hello")
