import argparse

from dataloader.datastore import Datastore
from htr_service import HTRService

image_dataset = "iam"
images_path = "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\MyCode\\data files\\datasets\\iam\\lines\\lines\\"
ground_truth_path = "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\MyCode\\data files\\datasets\\iam\\ascii\\lines.txt "
partition_criteria_path = "C:\\Users\\g.shankar.behera\\My " \
                     "Files\\Project\\Code\\MyCode\\data files\\datasets\\iam\\largeWriterIndependentTextLineRecognitionTask\\"
partitions = {'train': "trainset.txt", 'validation': "validationset1.txt", 'test': "testset.txt"}

# image_dataset = "iam"
# images_path = "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\MyCode\\data files\\datasets\\iam\\words\\"
# ground_truth_path = "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\MyCode\\data files\\datasets\\iam\\ascii\\words.txt "
# partition_criteria_path = "C:\\Users\\g.shankar.behera\\My " \
#                           "Files\\Project\\Code\\MyCode\\data files\\datasets\\iam\\largeWriterIndependentTextLineRecognitionTask\\words\\"
# partitions = {'train': "trainset.txt", 'validation': "validationset1.txt", 'test': "testset.txt"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--transform", action="store_true", default=False)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--architecture", type=str, required=False, default="flor")
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)

    args = parser.parse_args()

    if args.transform:
        print(f"{args.source} dataset will be transformed...")
        datastore = Datastore(partitions, datasource=args.source)
        datastore.read_raw_dataset(ground_truth_path, images_path, partition_criteria_path)
    elif args.train:
        print(f"{args.source} dataset will be trained with architecture {args.architecture}...")
        htrService = HTRService(architecture=args.architecture, datasource=args.source)
        model_history = htrService.start_training(partitions)
    elif args.test:
        print(f"Testing will be done...")
        htrService = HTRService(architecture=args.architecture, datasource=args.source)
        htrService.start_predicting(partitions)
