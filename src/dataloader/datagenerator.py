import os

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
        self.test_tfrs = [tfrecord for tfrecord in os.listdir(self.data_source) if tfrecord.startswith("test")]
        self.dataset_size = {
            "train": get_dataset_size(self.train_tfrs),
            "validation": get_dataset_size(self.validation_tfrs),
            "test": get_dataset_size(self.test_tfrs)
        }
        self.steps_per_epoch = {
            "train": int(self.dataset_size['train']/cfg.BATCH_SIZE),
            "validation": int(self.dataset_size['validation']/cfg.BATCH_SIZE),
            "test": int(self.dataset_size['test']/cfg.BATCH_SIZE)
        }

    def generate_batch(self, filenames, labeled=True):
        dataset = self.datastore.load(filenames, labeled=labeled)
        preprocessor = Preprocessor()
        dataset = dataset.map(preprocessor.augmentation, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(self.batch_size * 10)
        dataset = dataset.batch(self.batch_size)
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
        return self.generate_batch(self.test_tfrs, labeled=False)

# class Tokenizer():
#     def __init__(self, chars, max_text_length=128):
#         self.PAD_TK, self.UNK_TK = "¶", "¤"
#         self.chars = (self.PAD_TK + self.UNK_TK + chars)
#
#         self.PAD = self.chars.find(self.PAD_TK)
#         self.UNK = self.chars.find(self.UNK_TK)
#
#         self.vocab_size = len(self.chars)
#         self.maxlen = max_text_length
#
#     def encode(self, text):
#         """Encode text to vector"""
#
#         if isinstance(text, bytes):
#             text = text.decode()
#
#         text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")
#         text = " ".join(text.split())
#
#         groups = ["".join(group) for _, group in groupby(text)]
#         text = "".join([self.UNK_TK.join(list(x)) if len(x) > 1 else x for x in groups])
#         encoded = []
#
#         for item in text:
#             index = self.chars.find(item)
#             index = self.UNK if index == -1 else index
#             encoded.append(index)
#
#         return np.asarray(encoded)
#
#     def decode(self, text):
#         """Decode vector to text"""
#
#         decoded = "".join([self.chars[int(x)] for x in text if x > -1])
#         decoded = self.remove_tokens(decoded)
#
#         return decoded
#
#     def remove_tokens(self, text):
#         """Remove tokens (PAD) from text"""
#
#         return text.replace(self.PAD_TK, "").replace(self.UNK_TK, "")
