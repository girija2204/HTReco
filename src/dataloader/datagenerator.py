import os
import math

import tensorflow as tf

from src import configurations as cfg
from src.dataloader.datastore import Datastore
from src.dataloader.preprocessor import Preprocessor


def get_dataset_size(tfrs):
    size = 0
    for tfr in tfrs:
        records_count = int(os.path.basename(tfr).split(".")[0].split("-")[1])
        if records_count > size:
            size = records_count
    return size


class DataGenerator:
    def __init__(self, partitions, data_source="iam",
                 batch_size=cfg.BATCH_SIZE):
        self.batch_size = batch_size
        self.data_source = os.path.join(cfg.ROOT_DIR, "data", "processed", data_source)
        self.partitions = partitions
        self.datastore = Datastore(self.partitions)
        self.train_tfrs = [os.path.join(self.data_source, tfrecord) for tfrecord in os.listdir(self.data_source) if
                           tfrecord.startswith("train")]
        self.validation_tfrs = [os.path.join(self.data_source, tfrecord) for tfrecord in os.listdir(self.data_source) if
                                tfrecord.startswith("validation")]
        self.test_tfrs = [os.path.join(self.data_source, tfrecord) for tfrecord in os.listdir(self.data_source) if
                          tfrecord.startswith("test")]
        self.dataset_size = {
            "train": get_dataset_size(self.train_tfrs),
            "validation": get_dataset_size(self.validation_tfrs),
            "test": get_dataset_size(self.test_tfrs)
        }
        self.steps_per_epoch = {
            "train": math.ceil(self.dataset_size['train']/cfg.BATCH_SIZE),
            "validation": math.ceil(self.dataset_size['validation']/cfg.BATCH_SIZE),
            "test": math.ceil(self.dataset_size['test']/cfg.BATCH_SIZE)
        }

    def generate_batch(self, filenames, labeled=True, training=True):
        dataset = self.datastore.load(filenames, labeled=labeled)
        preprocessor = Preprocessor()
        dataset = dataset.map(preprocessor.augmentation, num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            dataset = dataset.shuffle(self.batch_size * 10)
        dataset = dataset.batch(self.batch_size).repeat()
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def generate_train_batch(self, filenames=None, labeled=True):
        if filenames is not None and len(filenames) > 0:
            self.train_tfrs = filenames
        return self.generate_batch(self.train_tfrs, labeled)

    def generate_valid_batch(self, filenames=None, labeled=True):
        if filenames is not None and len(filenames) > 0:
            self.validation_tfrs = filenames
        return self.generate_batch(self.validation_tfrs, labeled)

    def generate_test_batch(self, filenames=None, labeled=False):
        if filenames is not None and len(filenames) > 0:
            self.test_tfrs = filenames
        return self.generate_batch(self.test_tfrs, labeled, training=False)

    """
    Mostly required for prediction, as while prediction all the records are required at once to get their ground truth.
    No shuffling, batching or prefetching has been done on this data. By default, labeled is marked as False,
    as this is only for testing data. If needed in case for training data, then call this with labeled True.
    Not recommended to use this method for training.
    """
    def get_all_data(self, filenames=None, labeled=True):
        if filenames is not None and len(filenames) > 0:
            self.test_tfrs = filenames
        dataset = self.datastore.load(self.test_tfrs, labeled=labeled)
        preprocessor = Preprocessor()
        dataset = dataset.map(preprocessor.augmentation, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset


