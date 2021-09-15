import os
import string
from functools import partial

import numpy as np
import tensorflow as tf
import cv2
import tqdm

from src import constants
from src.dataloader.preprocessor import Preprocessor
from src.dataloader.tokenizer import Tokenizer


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def image_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def create_example(image, ground_truth):
    feature = {
        "input": image_feature(image),
        "target": _int64_feature(ground_truth)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_function(example, labeled):
    feature_description = {
        "input": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
    }
    example = tf.io.parse_single_example(example, feature_description)
    example['input'] = tf.io.decode_image(example["input"])
    paddings = [[0, constants.MAX_SEQUENCE_LENGTH - tf.shape(example['target'])[0]]]
    example['target'] = tf.pad(example['target'], paddings, 'CONSTANT', constant_values=0)
    # if not labeled:
    #     example['target'] = None
    return example


class Datastore:
    def __init__(self, images_path=None, ground_truth_path=None, partition_criteria_path=None,
                 dataset_path=None, datasource="iam", format=constants.STORAGE_FORMAT, img_format=constants.IMAGE_FORMAT):
        self.tokenizer = Tokenizer(string.printable[:95])
        self.format = format
        self.dataset = dict()
        self.input_size = constants.INPUT_SIZE
        self.img_format = img_format if img_format is not None else constants.IMAGE_FORMAT
        self.datasource = datasource
        self.preprocessor = Preprocessor()
        self.images_path = images_path
        self.ground_truth_path = ground_truth_path
        self.partition_criteria_path = partition_criteria_path
        self.dataset_path = os.path.join(constants.PROCESSED_DATA_PATH if dataset_path is None else dataset_path.strip('\"'), self.datasource)
        for partition_key, partition_value in constants.DEFAULT_PARTITIONING.items():
            self.dataset[partition_key] = {'ground_truth': [], 'file_path': []}

    def save(self):
        os.makedirs(self.dataset_path, exist_ok=True)
        for partition_key, partition_value in constants.DEFAULT_PARTITIONING.items():
            partition_size = len(self.dataset[partition_key]['ground_truth'])
            samples_per_tfrec = constants.SAMPLES_PER_TFRECORD
            num_tfrecs = partition_size // samples_per_tfrec + 1
            if partition_size % samples_per_tfrec == 0:
                num_tfrecs -= 1
            print(f"Parition: {partition_key} -> Size: {partition_size}")
            print(f"Total number of tfrecords: {num_tfrecs}")
            total_index = 0
            ignored_image_count = 0
            for current_tfrec in tqdm.tqdm(range(num_tfrecs)):
                examples_list = []
                current_index = 0
                # while current_index < samples_per_tfrec:
                while len(examples_list) < samples_per_tfrec:
                    if total_index == partition_size-1:
                        break
                    image_filepath = self.dataset[partition_key]['file_path'][total_index]
                    try:
                        if self.img_format == "tif":
                            image = cv2.imread(image_filepath, cv2.IMREAD_GRAYSCALE)
                            image = tf.convert_to_tensor(image)
                            image = image[..., tf.newaxis]
                        else:
                            image = tf.io.decode_jpeg(tf.io.read_file(image_filepath))
                        if image.shape.as_list()[0] > 10 and image.shape.as_list()[1] > 10:
                            image = self.preprocessor.preprocess(image)
                            image_label = self.dataset[partition_key]['ground_truth'][total_index]
                            image_label = self.tokenizer.encode(image_label)
                            example = create_example(image, image_label)
                            examples_list.append(example)
                        else:
                            ignored_image_count += 1
                            print(f"removing image count: {ignored_image_count}")
                        total_index += 1
                        current_index += 1
                    except BaseException as e:
                        print(e)
                        print(image_filepath)
                        total_index += 1
                        current_index += 1
                        ignored_image_count += 1
                        continue
                tfrec_path = os.path.join(self.dataset_path, f"{partition_key}_{current_tfrec}-{len(examples_list)}.tfrec")
                print(
                    f"{current_tfrec}/{num_tfrecs} - From {total_index-samples_per_tfrec-ignored_image_count} To {total_index} - Writing at {tfrec_path}")
                with tf.io.TFRecordWriter(tfrec_path) as writer:
                    for example in examples_list:
                        writer.write(example.SerializeToString())
                if total_index == partition_size - 1:
                    break

    def read_raw_dataset(self):
        getattr(self, self.datasource)()
        self.save()

    def load(self, tfrec_filenames=None, labeled=True):
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        if tfrec_filenames:
            dataset = tf.data.TFRecordDataset(filenames=tfrec_filenames)
        else:
            dataset = tf.data.TFRecordDataset(filenames=os.listdir(self.dataset_path))
        if dataset is None:
            print("No Existing Dataset found")
        else:
            dataset = dataset.with_options(ignore_order)
            dataset = dataset.map(partial(parse_function, labeled=labeled),
                                  num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    def breta(self):
        for partition_key, partition_value in constants.DEFAULT_PARTITIONING.items():
            with open(os.path.join(self.partition_criteria_path, partition_value), 'r') as pc:
                for filename in pc.read().splitlines():
                    file_path = os.path.join(self.images_path, f"{filename}")
                    self.dataset[partition_key]['ground_truth'].append(filename.split("_")[0])
                    self.dataset[partition_key]['file_path'].append(file_path)

    def cvl(self):
        for partition_key, partition_value in constants.DEFAULT_PARTITIONING.items():
            with open(os.path.join(self.partition_criteria_path, partition_value), 'r') as pc:
                for file_path in pc.read().splitlines():
                    self.dataset[partition_key]['ground_truth'].append(os.path.basename(file_path).split(".")[0].split("-")[-1])
                    self.dataset[partition_key]['file_path'].append(file_path)

    def iam(self):
        gt_dict = {}
        with open(self.ground_truth_path, 'r') as gtp:
            for gt in gtp.read().splitlines():
                if not gt.startswith("#"):
                    gt_splitted = gt.split(" ")
                    line_id = gt_splitted[0]
                    # if there's a space between two letters, then join them first, then split
                    sentence = " ".join(" ".join(gt_splitted[8:]).split("|"))
                    gt_dict[line_id] = sentence
        for partition_key, partition_value in constants.DEFAULT_PARTITIONING.items():
            with open(os.path.join(self.partition_criteria_path, partition_value), 'r') as pc:
                for line_id in pc.read().splitlines():
                    folder = line_id.split("-")[0]
                    subfolder = line_id.split("-")[1]
                    file = line_id.split("-")[2]
                    if self.datasource == "iam_words":
                        file = '-'.join(line_id.split("-")[2:])
                    file_path = os.path.join(self.images_path, folder, f"{folder}-{subfolder}",
                                             f"{folder}-{subfolder}-{file}.png")
                    self.dataset[partition_key]['ground_truth'].append(gt_dict[line_id])
                    self.dataset[partition_key]['file_path'].append(file_path)
