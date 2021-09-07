import os

from src.dataloader.datastore import Datastore


images_path = "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\MyCode\\data files\\datasets\\iam\\words\\"
gt_path = "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\MyCode\\data files\\datasets\\iam\\ascii\\words.txt"
partition_criteria_path = "C:\\Users\\g.shankar.behera\\My " \
                          "Files\\Project\\Code\\MyCode\\data files\\datasets\\iam\\largeWriterIndependentTextLineRecognitionTask\\"
partitions = {'train': "trainset.txt", 'validation': "validationset1.txt", 'test': "testset.txt"}
# datastore=Datastore(partitions)
# gt=datastore.read_ground_truth(gt_path)
# print(gt)

output_partition_path="C:\\Users\\g.shankar.behera\\My " \
                          "Files\\Project\\Code\\MyCode\\data files\\datasets\\iam\\largeWriterIndependentTextLineRecognitionTask\\words\\"

for partition_key, partition_value in partitions.items():
    with open(os.path.join(partition_criteria_path, partition_value), 'r') as pc:
        f = open(output_partition_path + partition_value,'w')
        for line_id in pc.read().splitlines():
            folder = line_id.split("-")[0]
            subfolder = line_id.split("-")[1]
            file = line_id.split("-")[2]
            file_path = os.path.join(images_path, folder, f"{folder}-{subfolder}",
                                     f"{folder}-{subfolder}-{file}.png")
            list_images=os.listdir(os.path.join(images_path, folder, f"{folder}-{subfolder}"))
            for img in list_images:
                if img.startswith(line_id):
                    f.write(img.split(".")[0]+"\n")
            print("hello")