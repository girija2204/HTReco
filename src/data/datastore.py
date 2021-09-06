import os
from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
import tqdm

from src import configurations as cfg


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def image_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]))


def create_example(image, ground_truth):
    feature = {
        "input": image_feature(image),
        "target": bytes_feature(str.encode(ground_truth))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_function(example, labeled):
    feature_description = {
        "input": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, feature_description)
    example['input'] = tf.io.decode_image(example["input"])
    tokenizer = tf_text.UnicodeCharTokenizer()
    if labeled:
        target = tokenizer.tokenize(example['target'])
        paddings = [[0, cfg.MAX_SEQUENCE_LENGTH - tf.shape(target)[0]]]
        example['target'] = tf.pad(target, paddings, 'CONSTANT', constant_values=0)
        return example
    return example


class Datastore:
    def __init__(self, partitions, format=cfg.STORAGE_FORMAT,
                 raw_data_path=os.path.join(cfg.ROOT_DIR, "data", "raw"),
                 dataset_path=os.path.join(cfg.ROOT_DIR, "data", "processed"),
                 datasource="iam"):
        # self.tokenizer = Tokenizer(string.printable[:95])
        self.partitions = partitions
        self.format = format
        self.dataset = dict()
        self.input_size = cfg.INPUT_SIZE
        self.raw_data_path = os.path.join(raw_data_path, datasource)
        self.dataset_path = os.path.join(dataset_path, datasource)
        for partition_key, partition_value in self.partitions.items():
            self.dataset[partition_key] = {'ground_truth': [], 'file_path': []}

    def preprocess(self, image):
        image = tf.squeeze(image)
        unique_vals, indices = tf.unique(tf.reshape(image, [-1]))
        background = int(unique_vals[np.argmax(np.bincount(indices))])
        wt, ht, _ = self.input_size
        h, w = np.asarray(image).shape
        f = max((w / wt), (h / ht))

        new_size = [max(min(ht, int(h / f)), 1), max(min(wt, int(w / f)), 1)]
        image = image[tf.newaxis, ..., tf.newaxis]
        image = tf.image.resize(image, new_size)[0, ..., 0].numpy()

        target = np.ones([ht, wt], dtype=np.uint8) * background
        target[0:new_size[0], 0:new_size[1]] = image
        target = target[..., tf.newaxis]
        img = tf.image.transpose(target)
        return img

    def save(self):
        os.makedirs(self.dataset_path, exist_ok=True)
        for partition_key, partition_value in self.partitions.items():
            partition_size = len(self.dataset[partition_key]['ground_truth'])
            samples_per_tfrec = cfg.SAMPLES_PER_TFRECORD
            num_tfrecs = partition_size // samples_per_tfrec + 1
            if partition_size % samples_per_tfrec == 0:
                num_tfrecs -= 1
            print(f"Parition: {partition_key} -> Size: {partition_size}")
            print(f"Total number of tfrecords: {num_tfrecs}")
            total_index = 0
            for current_tfrec in tqdm.tqdm(range(num_tfrecs)):
                tfrec_path = os.path.join(self.dataset_path, f"{partition_key}_{current_tfrec}-{partition_size if current_tfrec == num_tfrecs-1 else total_index}.tfrec")
                print(
                    f"{current_tfrec}/{num_tfrecs} - From {total_index} To {total_index + samples_per_tfrec if total_index + samples_per_tfrec < partition_size else partition_size} - Writing at {tfrec_path}")
                with tf.io.TFRecordWriter(tfrec_path) as writer:
                    current_index = 0
                    while current_index < samples_per_tfrec:
                        if total_index == len(self.dataset[partition_key]['file_path']):
                            break
                        image_filepath = self.dataset[partition_key]['file_path'][total_index]
                        image = tf.io.decode_jpeg(tf.io.read_file(image_filepath))
                        image = self.preprocess(image)
                        image_label = self.dataset[partition_key]['ground_truth'][total_index]
                        example = create_example(image, image_label)
                        writer.write(example.SerializeToString())
                        current_index += 1
                        total_index += 1

    def read_ground_truth(self, ground_truth_path):
        gt_dict = {}
        with open(ground_truth_path, 'r') as gtp:
            for gt in gtp.read().splitlines():
                if not gt.startswith("#"):
                    gt_splitted = gt.split(" ")
                    line_id = gt_splitted[0]
                    # if there's a space between two letters, then join them first, then split
                    sentence = " ".join(" ".join(gt_splitted[8:]).split("|"))
                    gt_dict[line_id] = sentence
        return gt_dict

    def transform(self, images_path, ground_truth, partition_criteria_path):
        for partition_key, partition_value in self.partitions.items():
            with open(os.path.join(partition_criteria_path, partition_value), 'r') as pc:
                for line_id in pc.read().splitlines():
                    folder = line_id.split("-")[0]
                    subfolder = line_id.split("-")[1]
                    file = line_id.split("-")[2]
                    file_path = os.path.join(images_path, folder, f"{folder}-{subfolder}",
                                             f"{folder}-{subfolder}-{file}.png")
                    self.dataset[partition_key]['ground_truth'].append(ground_truth[line_id])
                    self.dataset[partition_key]['file_path'].append(file_path)

    def read_raw_dataset(self, ground_truth_path, images_path, partition_criteria_path):
        ground_truth = self.read_ground_truth(ground_truth_path)
        self.transform(images_path, ground_truth, partition_criteria_path)
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
