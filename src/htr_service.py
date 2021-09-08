import os

from src.dataloader.datagenerator import DataGenerator
from src.network import evaluation
from src.network.model import HTRModel
import src.configurations as cfg


class HTRService:
    def __init__(self, architecture, datasource):
        self.architecture = architecture
        self.input_image_size = cfg.INPUT_SIZE
        self.output_units = cfg.VOCABULARY_SIZE
        self.datasource = datasource
        self.output_path = os.path.join(cfg.ROOT_DIR, "output", self.datasource, self.architecture)
        self.checkpoint_path = os.path.join(self.output_path, "checkpoint_weights.hdf5")
        self.datagen = None

    def get_data(self, partitions, training=True):
        self.datagen = DataGenerator(partitions=partitions, data_source=self.datasource)
        if not training:
            test_dataset = self.datagen.generate_test_batch(labeled=False)
            test_all_data = self.datagen.get_all_data(labeled=True)
            return test_dataset, test_all_data
        dataset = self.datagen.generate_train_batch(labeled=True)
        val_dataset = self.datagen.generate_valid_batch(labeled=True)
        return dataset, val_dataset

    def start_training(self, partitions):
        dataset, val_dataset = self.get_data(partitions)
        model = HTRModel(self.architecture, self.input_image_size, self.output_units)
        model.compile()
        model.load_checkpoint(self.checkpoint_path)
        model.summary()
        callbacks = model.get_callbacks(logdir=self.output_path, checkpoint_path=self.checkpoint_path, verbose=1)
        model_history = model.fit(x=dataset,
                                  epochs=cfg.EPOCHS,
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

        with open(os.path.join(self.output_path, "train.txt"), "w") as lg:
            lg.write(t_corpus)
            print(t_corpus)
        return model_history

    def start_predicting(self, partitions):
        test_dataset, test_all_data = self.get_data(partitions, training=False)
        model = HTRModel(self.architecture, self.input_image_size, self.output_units)
        model.compile()
        model.load_checkpoint(self.checkpoint_path)
        model.summary()
        preds, _ = model.predict(test_dataset,
                                    steps=self.datagen.steps_per_epoch['test'],
                                    verbose=1)
        predicts = [self.datagen.datastore.tokenizer.decode(predict[0]) for predict in preds]
        ground_truth = next(iter(test_all_data.batch(self.datagen.dataset_size['test'])))[1]
        gts = []
        for gt in ground_truth:
            gts.append(self.datagen.datastore.tokenizer.decode(gt))
        with open(os.path.join(self.output_path, "predict.txt"), "w") as lg:
            for pd, gt in zip(predicts, ground_truth):
                lg.write(f"TE_L {gt}\nTE_P {pd}\n")

        evaluate = evaluation.ocr_metrics(predicts=predicts,
                                          ground_truth=ground_truth,
                                          norm_accentuation=True,
                                          norm_punctuation=True)

        e_corpus = "\n".join([
            f"Total test images:    {self.datagen.dataset_size['test']}",
            f"Metrics:",
            f"Character Error Rate: {evaluate[0]:.8f}",
            f"Word Error Rate:      {evaluate[1]:.8f}",
            f"Sequence Error Rate:  {evaluate[2]:.8f}"
        ])

        with open(os.path.join(self.output_path, f"evaluate.txt"), "w") as lg:
            lg.write(e_corpus)
            print(e_corpus)
