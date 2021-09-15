import os
import string
import time

import cv2
from matplotlib import pyplot as plt

from src.dataloader.datagenerator import DataGenerator
from src.dataloader.preprocessor import Preprocessor
from src.dataloader.tokenizer import Tokenizer
from src.network import evaluation
from src.network.model import HTRModel
from src import constants

import tensorflow as tf


class HTRService:
    def __init__(self, architecture, datasource, segmentation_level):
        self.architecture = architecture
        self.input_image_size = constants.INPUT_SIZE
        self.output_units = constants.VOCABULARY_SIZE
        self.datasource = datasource
        self.segmentation_level = constants.DEFAULT_SEGMENTATION_LEVEL if segmentation_level is None else segmentation_level
        self.preprocessor = Preprocessor()

        self.temp_dir = os.path.join(constants.ROOT_DIR, "output", self.segmentation_level, "temp")
        os.makedirs(self.temp_dir, exist_ok=True)

        self.output_path = os.path.join(constants.ROOT_DIR, "output", self.segmentation_level, self.architecture)
        os.makedirs(self.output_path, exist_ok=True)

        self.checkpoint_path = os.path.join(self.output_path, "checkpoint_weights.hdf5")
        self.train_output_file = os.path.join(self.output_path, "train.txt")
        self.prediction_output_file = os.path.join(self.output_path, "predict.txt")
        self.evaluation_output_file = os.path.join(self.output_path, "evaluate.txt")

        self.datagen = None

    def benchmark(self, batch_size, num_step):
        start = time.time()
        self.datagen = DataGenerator(data_source=self.datasource, batch_size=batch_size)
        dataset = self.datagen.generate_train_batch(labeled=True)
        num_step = self.datagen.steps_per_epoch['train'] if num_step == -1 else num_step
        print(f"Batch Size: {self.datagen.batch_size}, Number Steps: {num_step}")
        dataset_iterator = iter(dataset)
        for i in range(num_step):
            _, _ = next(dataset_iterator)
        end = time.time()
        return end - start

    def get_data(self, training=True):
        self.datagen = DataGenerator(data_source=self.datasource)
        if not training:
            test_dataset = self.datagen.generate_test_batch(labeled=False)
            test_all_data = self.datagen.get_all_data(labeled=True)
            return test_dataset, test_all_data
        dataset = self.datagen.generate_train_batch(labeled=True)
        val_dataset = self.datagen.generate_valid_batch(labeled=True)
        return dataset, val_dataset

    def train(self):
        dataset, val_dataset = self.get_data()
        model = HTRModel(self.architecture, self.input_image_size, self.output_units)
        model.compile()
        model.load_checkpoint(self.checkpoint_path)
        model.summary()
        callbacks = model.get_callbacks(logdir=self.output_path, checkpoint_path=self.checkpoint_path, verbose=1)
        model_history = model.fit(x=dataset,
                                  epochs=constants.EPOCHS,
                                  steps_per_epoch=self.datagen.steps_per_epoch['train'],
                                  validation_data=val_dataset,
                                  validation_steps=self.datagen.steps_per_epoch['validation'],
                                  callbacks=callbacks,
                                  shuffle=True,
                                  verbose=1)
        loss = model_history.history['loss']
        val_loss = model_history.history['val_loss']

        min_val_loss = min(val_loss)
        min_val_loss_i = val_loss.index(min_val_loss)

        t_corpus = "\n".join([
            f"Total train images:      {self.datagen.dataset_size['train']}",
            f"Total validation images: {self.datagen.dataset_size['validation']}",
            f"Batch:                   {self.datagen.batch_size}\n",
            f"Total epochs:            {len(loss)}",
            f"Best epoch               {min_val_loss_i + 1}\n",
            f"Training loss:           {loss[min_val_loss_i]:.8f}",
            f"Validation loss:         {min_val_loss:.8f}"
        ])

        with open(self.train_output_file, "w") as tof:
            tof.write(t_corpus)
            print(t_corpus)
        return model_history

    def predict_and_evaluate(self):
        test_dataset, test_all_data = self.get_data(training=False)
        model = HTRModel(self.architecture, self.input_image_size, self.output_units)
        model.compile()
        model.load_checkpoint(self.checkpoint_path)
        model.summary()

        count = 0
        preds_list = []
        gt_list = []
        test_size = self.datagen.dataset_size['test']
        for records in iter(test_dataset):
            if count >= test_size:
                break
            for index, record in enumerate(records[0]):
                preds, _ = model.predict(tf.reshape(record, [1, 1024, 128, 1]))
                predicts = [self.datagen.datastore.tokenizer.decode(predict[0]) for predict in preds]
                ground_truth = self.datagen.datastore.tokenizer.decode(records[1][index])
                count += 1
                print(f"Processing {count}/{test_size} test file: gt - {ground_truth}, pt - {predicts[0]}")
                preds_list.append(predicts[0])
                gt_list.append(ground_truth)

        with open(self.prediction_output_file, "w") as pof:
            for pd, gt in zip(preds_list, gt_list):
                pof.write(f"GroundT: {gt}\nPredict: {pd}\n\n")

        evaluate = evaluation.ocr_metrics(predicts=preds_list,
                                          ground_truth=gt_list,
                                          norm_accentuation=True,
                                          norm_punctuation=True)

        e_corpus = "\n".join([
            f"Total test images:    {self.datagen.dataset_size['test']}",
            f"Metrics:",
            f"Character Error Rate: {evaluate[0]:.8f}",
            f"Word Error Rate:      {evaluate[1]:.8f}",
            f"Sequence Error Rate:  {evaluate[2]:.8f}"
        ])

        with open(self.evaluation_output_file, "w") as eof:
            eof.write(e_corpus)
            print(e_corpus)

    def predict(self, data_source, results_folder):
        if not os.path.isdir(data_source) or not os.path.isdir(results_folder): print(
            f"Make sure {data_source} and {results_folder} are present.")
        files = os.listdir(data_source)
        tokenizer = Tokenizer(string.printable[:95])
        model = HTRModel(self.architecture, self.input_image_size, self.output_units)
        model.compile()
        model.load_checkpoint(self.checkpoint_path)
        for file in files:
            # if file.endswith(".png"):
            image = cv2.imread(os.path.join(data_source, file), cv2.IMREAD_GRAYSCALE)
            # elif file.endswith(".tif"):
            #     image = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
            # image = tf.convert_to_tensor(image)
            # else:
            #     continue
            image = image[..., tf.newaxis]
            image = self.preprocessor.preprocess(image)
            image_1 = tf.reshape(image, [1, 1024, 128, 1])
            image_1 = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)(image_1)
            preds, _ = model.predict(image_1,
                                     ctc_decode=True)
            predicts = [tokenizer.decode(predict[0]) for predict in preds]
            # ground_truth = records[1][index]
            tttt = tf.reshape(image_1, [1024, 128])
            ttt = tf.transpose(tttt)
            plt.imshow(ttt.numpy(), cmap="gray")
            plt.axis("off")
            plt.show()
            print(f"Predicted: {predicts[0]}")
            # print(f"Ground Truth: {tokenizer.decode(ground_truth)}")
            # preds_list.append(predicts[0])
            # gt_list.append(htrService.datagen.datastore.tokenizer.decode(ground_truth))
            print("\n")
