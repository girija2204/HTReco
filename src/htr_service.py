import os

from src.dataloader.datagenerator import DataGenerator
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

    def get_data(self, partitions):
        self.datagen = DataGenerator(partitions=partitions, data_source=self.datasource)
        dataset = self.datagen.generate_train_batch(labeled=True)
        val_dataset = self.datagen.generate_valid_batch(labeled=True)
        return dataset, val_dataset

    def start_training(self, partitions):
        dataset, val_dataset = self.get_data(partitions)
        model = HTRModel(self.architecture, self.input_image_size, self.output_units)
        model.compile()
        model.load_checkpoint()
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
            f"Total validation images: {self.datagen.dataset_size['valid']}",
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

    def predict(self):
        pass
