import os
import string
import unicodedata
from functools import partial
from itertools import groupby

import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
import tqdm

from src import configurations as cfg
from src.data.preprocessor import Preprocessor


def bytes_feature(value):
    # return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))
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
        # return (example['input'], example['target'])
    return example
    # return (example['input'])


class Datastore:
    def __init__(self, partitions, format=cfg.STORAGE_FORMAT,
                 raw_data_path=os.path.join(cfg.ROOT_DIR, "data\\raw\\"),
                 dataset_path=os.path.join(cfg.ROOT_DIR, "data\\processed\\")):
        self.tokenizer = Tokenizer(string.printable[:95])
        self.partitions = partitions
        self.format = format
        self.dataset = dict()
        self.input_size = cfg.INPUT_SIZE
        self.raw_data_path = raw_data_path
        self.dataset_path = dataset_path
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
                tfrec_path = os.path.join(self.dataset_path, f"{partition_key}_{current_tfrec}-{total_index}.tfrec")
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

    def read_raw_dataset(self):
        pass

    def load(self, tfrec_filenames=None, labeled=True):
        # tf.compat.v1.disable_eager_execution()
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


class DataGenerator:
    def __init__(self, partitions, data_source=os.path.join(cfg.ROOT_DIR, "data\\processed\\"),
                 batch_size=cfg.BATCH_SIZE):
        self.batch_size = batch_size
        self.data_source = data_source
        self.partitions = partitions
        self.datastore = Datastore(self.partitions)
        self.train_tfrs = [os.path.join(self.data_source, tfrecord) for tfrecord in os.listdir(self.data_source) if
                           tfrecord.startswith("train")]
        self.validation_tfrs = [os.path.join(self.data_source, tfrecord) for tfrecord in os.listdir(self.data_source) if
                                tfrecord.startswith("validation")]
        self.test_tfrs = [tfrecord for tfrecord in os.listdir(self.data_source) if tfrecord.startswith("test")]
        self.steps_per_epoch = {
            "train": 0,
            "validation": 0,
            "test": 0
        }

    def generate_batch(self, filenames, labeled=True):
        dataset = self.datastore.load(filenames, labeled=labeled)
        preprocessor = Preprocessor()
        dataset = dataset.map(preprocessor.augmentation, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(self.batch_size * 10)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def generate_train_batch(self, filenames, labeled=True):
        return self.generate_batch(self.train_tfrs, labeled)

    def generate_valid_batch(self, filenames, labeled=True):
        return self.generate_batch(self.validation_tfrs, labeled)

    def generate_test_batch(self, filenames, labeled=False):
        return self.generate_batch(self.test_tfrs, labeled=False)


class Tokenizer():
    def __init__(self, chars, max_text_length=128):
        self.PAD_TK, self.UNK_TK = "¶", "¤"
        self.chars = (self.PAD_TK + self.UNK_TK + chars)

        self.PAD = self.chars.find(self.PAD_TK)
        self.UNK = self.chars.find(self.UNK_TK)

        self.vocab_size = len(self.chars)
        self.maxlen = max_text_length

    def encode(self, text):
        """Encode text to vector"""

        if isinstance(text, bytes):
            text = text.decode()

        text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")
        text = " ".join(text.split())

        groups = ["".join(group) for _, group in groupby(text)]
        text = "".join([self.UNK_TK.join(list(x)) if len(x) > 1 else x for x in groups])
        encoded = []

        for item in text:
            index = self.chars.find(item)
            index = self.UNK if index == -1 else index
            encoded.append(index)

        return np.asarray(encoded)

    def decode(self, text):
        """Decode vector to text"""

        decoded = "".join([self.chars[int(x)] for x in text if x > -1])
        decoded = self.remove_tokens(decoded)

        return decoded

    def remove_tokens(self, text):
        """Remove tokens (PAD) from text"""

        return text.replace(self.PAD_TK, "").replace(self.UNK_TK, "")
