import os
import string
import cv2
from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_text as tf_text

from src.dataloader.datastore import parse_function, Datastore
from src.dataloader.datagenerator import DataGenerator
from src.dataloader.preprocessor import Preprocessor
from src.dataloader.tokenizer import Tokenizer
from src.network.model import HTRModel


class test_HTR:
    def __init__(self):
        pass
        # self.tokenizer = Tokenizer(string.printable[:95])

    def test_parse_function(self):
        print(f"test_parse_function")
        tokenizer = Tokenizer(string.printable[:95])
        tfrec_filenames = ["C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\HTR\\data\\processed\\iam\\train_0-0.tfrec"]
        dataset = tf.data.TFRecordDataset(filenames=tfrec_filenames)
        example = parse_function(next(iter(dataset)), labeled=True)
        label, image = example['ground_truth'], example['image'].numpy()
        tokens = tokenizer.decode(label)
        paddings = [[0, 40 - tf.shape(example['ground_truth'])[0].numpy()]]
        example['ground_truth'] = tf.pad(example['ground_truth'], paddings, 'CONSTANT', constant_values=0)
        return image, label

    # def test_parse_function(self):
    #     print(f"test_parse_function")
    #     # self.tokenizer = Tokenizer(string.printable[:95])
    #     tfrec_filenames = [
    #         "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\HTR\\data\\processed\\train_0-0.tfrec"]
    #     dataset = tf.data.TFRecordDataset(filenames=tfrec_filenames)
    #     tokenizer = tf_text.UnicodeCharTokenizer()
    #     dataset = dataset.map(partial(parse_function, labeled=True),
    #                           num_parallel_calls=tf.data.AUTOTUNE)
    #     records = next(iter(dataset))
    #     label, image = records['ground_truth'], records['image'].numpy()
    #     tokens = tokenizer.detokenize(label)
    #     return image, label

    def test_parse_function_label_false(self):
        print(f"test_parse_function_label_false")
        # self.tokenizer = Tokenizer(string.printable[:95])
        tfrec_filenames = ["C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\HTR\\data\\processed\\test_0-0.tfrec"]
        dataset = tf.data.TFRecordDataset(filenames=tfrec_filenames)
        dataset = dataset.map(partial(parse_function, labeled=False),
                              num_parallel_calls=tf.data.AUTOTUNE)
        records = next(iter(dataset))
        _, image = records['ground_truth'].numpy(), records['image'].numpy()
        return image

    def test_parse_function_label_true(self):
        print(f"test_parse_function_label_true")
        # self.tokenizer = Tokenizer(string.printable[:95])
        tfrec_filenames = [
            "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\HTR\\data\\processed\\train_0-0.tfrec"]
        dataset = tf.data.TFRecordDataset(filenames=tfrec_filenames)
        tokenizer = tf_text.UnicodeCharTokenizer()
        dataset = dataset.map(partial(parse_function, labeled=True),
                              num_parallel_calls=tf.data.AUTOTUNE)
        records = next(iter(dataset))
        label, image = records['ground_truth'].numpy(), records['image'].numpy()
        tokens = tokenizer.detokenize(label)
        return image, label

    def test_datastore_load(self):
        partitions = {'train': "trainset_new.txt", 'validation': "validationset_new.txt", 'test': "testset_new.txt"}
        datastore = Datastore(partitions=partitions)
        print(f"test_parse_function_label_true")
        # self.tokenizer = Tokenizer(string.printable[:95])
        tfrec_filenames = [
            "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\HTR\\data\\processed\\train_9-1152.tfrec"]
        dataset = datastore.load(tfrec_filenames, labeled=False)
        records = next(iter(dataset))
        image = records['image'].numpy()
        return image

    def test_datagenerator_generate_batch(self):
        partitions = {'train': "trainset_new.txt", 'validation': "validationset_new.txt", 'test': "testset_new.txt"}
        datagen = DataGenerator(partitions=partitions)
        print(f"test_parse_function_label_true")
        tfrec_filenames = [
            "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\HTR\\data\\processed\\train_0-0.tfrec"]
        dataset = datagen.generate_batch(tfrec_filenames, labeled=False)
        records = next(iter(dataset))
        image = records['image'].numpy()
        return image

    def test_augmentation_1(self):
        prep = Preprocessor()
        tfrec_filenames = ["C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\HTR\\data\\processed\\test_0-0.tfrec"]
        dataset = tf.data.TFRecordDataset(filenames=tfrec_filenames)
        dataset = dataset.map(partial(parse_function, labeled=False),
                              num_parallel_calls=tf.data.AUTOTUNE)
        records = next(iter(dataset))
        aug_file = prep.augmentation(records)
        return aug_file

    def test_augmentation_2(self):
        prep = Preprocessor()
        tfrec_filenames = ["C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\HTR\\data\\processed\\train_9-1152"
                           ".tfrec"]
        dataset = tf.data.TFRecordDataset(filenames=tfrec_filenames)
        dataset = dataset.map(partial(parse_function, labeled=True),
                              num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(prep.augmentation, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(16 * 10)
        dataset = dataset.batch(16)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def test_datagenerator_generate_batch_1(self):
        partitions = {'train': "trainset_new.txt", 'validation': "validationset_new.txt", 'test': "testset_new.txt"}
        datagen = DataGenerator(partitions=partitions)
        print(f"test_parse_function_label_true")
        tfrec_filenames = [
            "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\HTR\\data\\processed\\train_0-0.tfrec"]
        dataset = datagen.generate_train_batch(tfrec_filenames, labeled=True)
        return dataset

    def test_model(self):
        partitions = {'train': "trainset_new.txt", 'validation': "validationset_new.txt", 'test': "testset_new.txt"}
        datagen = DataGenerator(partitions=partitions)
        print(f"test_parse_function_label_true")
        tfrec_filenames = [
            "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\HTR\\data\\processed\\train_0-0.tfrec"]
        dataset = datagen.generate_train_batch(tfrec_filenames, labeled=True)
        val_tfrec_filenames = [
            "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\HTR\\data\\processed\\validation_0-0.tfrec"]
        val_dataset = datagen.generate_valid_batch(val_tfrec_filenames, labeled=True)

        dataset_name = "iam"
        arch = "flor"
        base_path = "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\HTR\\"
        output_path = f"{base_path}output\\{dataset_name}\\{arch}\\"
        checkpoint_path = f"{output_path}\\checkpoint_weights.hdf5"

        model = HTRModel("flor", (1024, 128, 1), 128)
        model.compile()
        model.load_checkpoint()
        model.summary()
        callbacks = model.get_callbacks(logdir=output_path, checkpoint_path=checkpoint_path, verbose=1)
        model_history = model.fit(x=dataset,
                                  epochs=10,
                                  steps_per_epoch=50,
                                  validation_data=val_dataset,
                                  validation_steps=20,
                                  callbacks=callbacks,
                                  shuffle=True,
                                  verbose=1)

        return model_history

    def test_preprocessing_brightness(self):
        partitions = {'train': "trainset_new.txt", 'validation': "validationset_new.txt", 'test': "testset_new.txt"}
        datagen = DataGenerator(partitions=partitions)
        print(f"test_parse_function_label_true")
        tfrec_filenames = [
            "C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\HTR\\data\\processed\\iam\\train_0-1024.tfrec"]
        dataset_1 = datagen.generate_train_batch(tfrec_filenames, labeled=True)
        records_1 = next(iter(dataset_1))
        print("hello")


# test_htr = test_HTR()
# image = test_htr.test_parse_function_label_false()
# plt.imshow(image)
# plt.show()
# #
# test_htr = test_HTR()
# image, label = test_htr.test_parse_function()
# plt.imshow(image)
# plt.show()
# print(label)
# #
# image, label = test_htr.test_parse_function_label_true()
# plt.imshow(image)
# plt.show()
# print(label)

# test_htr = test_HTR()
# image = test_htr.test_datastore_load()
# plt.imshow(image)
# plt.show()

# test_htr = test_HTR()
# image = test_htr.test_datagenerator_generate_batch()
# plt.imshow(image)
# plt.show()

# test_htr = test_HTR()
# dataset = test_htr.test_datagenerator_generate_batch_1()
# tokenizer = tf_text.UnicodeCharTokenizer()
# records=next(iter(dataset))
# print(tokenizer.detokenize(records['ground_truth'][0]))
# plt.imshow(records['input'][0])
# plt.show()

# htr = test_HTR()
# dataset = htr.test_augmentation_2()
# # records = next(iter(dataset))
# # plt.imshow(records['image'][0])
# # plt.show()
#
# import matplotlib.pyplot as plt
#
# records = next(iter(dataset))
# tokenizer = Tokenizer(string.printable[:95])
# print(tokenizer.decode(records['ground_truth'][0]))
# plt.imshow(records['image'][0])
# plt.show()

test_htr = test_HTR()
test_htr.test_preprocessing_brightness()

# img_path="C:\\Users\\g.shankar.behera\\My Files\\Project\\Code\\MyCode\\data files\\datasets\\iam\\lines\\lines\\a05\\a05-004\\a05-004-00.png"
# img=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
# image_enhanced = cv2.equalizeHist(img)
#
# plt.imshow(image_enhanced, cmap='gray'), plt.axis("off")
# plt.show()
#
# print("hello")
