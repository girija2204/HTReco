import os

import numpy as np
import tensorflow as tf

from contextlib import redirect_stdout
from tensorflow.keras.layers import Input, Conv2D, Dense, Activation, Multiply, Dropout, MaxPooling2D, Reshape, \
    Bidirectional, GRU
from tensorflow.keras.layers import PReLU, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras import Model
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from src import configurations as cfg


class HTRModel:
    def __init__(self, architecture, input_size, vocab_size, greedy=False,
                 beam_width=10,
                 top_paths=1):
        self.learning_schedule = False
        self.model = None
        self.architecture = globals()[architecture]
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.stop_tolerance = cfg.STOP_TOLERANCE
        self.reduce_tolerance = cfg.REDUCE_TOLERANCE
        self.cooldown = cfg.COOLDOWN
        self.greedy = greedy
        self.beam_width = beam_width
        self.top_paths = max(1, top_paths)

    def summary(self, target=None):
        self.model.summary()
        if target is not None:
            os.makedirs(target, exist_ok=True)
            with open(os.path.join(target, "summary.txt"), "w") as f:
                with redirect_stdout(f):
                    self.model.summary()

    def load_checkpoint(self, target=None):
        if target is not None and os.path.isfile(target):
            if self.model is None:
                self.compile()
            self.model.load_weights(target)

    def get_callbacks(self, logdir, checkpoint_path, monitor=cfg.MONITOR_PARAM, verbose=0):
        callbacks = [
            CSVLogger(filename=os.path.join(logdir, "epochs.log"),
                      separator=";",
                      append=True),
            TensorBoard(log_dir=logdir,
                        histogram_freq=10,
                        write_graph=True,
                        write_images=False,
                        update_freq="epoch",
                        profile_batch=0),
            ModelCheckpoint(filepath=checkpoint_path,
                            monitor=monitor,
                            save_best_only=True,
                            save_weights_only=True,
                            verbose=verbose),
            EarlyStopping(monitor=monitor,
                          min_delta=1e-8,
                          restore_best_weights=True,
                          verbose=verbose,
                          patience=self.stop_tolerance),
            ReduceLROnPlateau(monitor=monitor,
                              min_delta=1e-8,
                              factor=0.2,
                              patience=self.reduce_tolerance,
                              cooldown=self.cooldown,
                              verbose=verbose)
        ]

        return callbacks

    def compile(self, learning_rate=cfg.LEARNING_RATE, initial_step=0):
        if learning_rate is None:
            learning_rate = CustomSchedule(output_units=self.vocab_size + 2, initial_step=initial_step)
            self.learning_schedule = True
        input, outputs = self.architecture(self.input_size, self.vocab_size + 2)
        optimizer = RMSprop(learning_rate=learning_rate)
        # target = Input(name="target",shape=(82,))
        self.model = Model(inputs=input, outputs=outputs, name="model_inf")
        self.model.compile(optimizer=optimizer, loss=self.ctc_loss_lambda_function)

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.0,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            **kwargs):
        if callbacks and self.learning_schedule:
            callbacks = [x for x in callbacks if not isinstance(x, ReduceLROnPlateau)]
        out = self.model.fit(x=x, batch_size=batch_size, epochs=epochs, verbose=verbose,
                             callbacks=callbacks, validation_split=validation_split, validation_data=validation_data,
                             validation_freq=validation_freq, validation_steps=validation_steps, shuffle=shuffle,
                             class_weight=class_weight, sample_weight=sample_weight, initial_epoch=initial_epoch,
                             steps_per_epoch=steps_per_epoch, workers=workers, max_queue_size=max_queue_size,
                             use_multiprocessing=use_multiprocessing, **kwargs)
        return out

    def predict(self,
                x=None,
                batch_size=None,
                verbose=0,
                steps=1,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False,
                ctc_decode=True):
        print("Model Predict")
        out = self.model.predict(x=x, batch_size=batch_size, verbose=verbose, steps=steps,
                                 callbacks=callbacks, max_queue_size=max_queue_size, workers=workers,
                                 use_multiprocessing=use_multiprocessing)
        if not ctc_decode:
            return np.log(out.clip(min=1e-8)), []
        print("ctc decode")
        progbar = tf.keras.utils.Progbar(target=steps)
        batch_size = int(np.ceil(len(out) / steps))
        input_length = len(max(out, key=len))

        predicts, probabilities = [], []
        steps_done = 0
        while steps_done < steps:
            index = steps_done * batch_size
            until = index + batch_size

            x_test = np.asarray(out[index:until])
            x_test_len = np.asarray([input_length for _ in range(len(x_test))])

            decode, log = K.ctc_decode(x_test,
                                       x_test_len,
                                       greedy=self.greedy,
                                       beam_width=self.beam_width,
                                       top_paths=self.top_paths)

            probabilities.extend([np.exp(x) for x in log])
            decode = [[[int(p) for p in x if p != -1] for x in y] for y in decode]
            predicts.extend(np.swapaxes(decode, 0, 1))

            steps_done += 1
            if verbose == 1:
                progbar.update(steps_done)

        return predicts, probabilities

    @staticmethod
    def ctc_loss_lambda_function(y_true, y_pred):
        if len(y_true.shape) > 2:
            y_true = tf.squeeze(y_true)
        input_length = tf.math.reduce_sum(y_pred, axis=-1, keepdims=False)
        input_length = tf.math.reduce_sum(input_length, axis=-1, keepdims=True)
        label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype='int64')
        loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        loss = tf.reduce_mean(loss)
        return loss


class CustomSchedule(LearningRateSchedule):
    def __init__(self, output_units, initial_step=0, warmup_steps=4000):
        self.initial_step = initial_step
        self.warmup_steps = warmup_steps
        self.output_units = output_units
        self.output_units = tf.cast(self.output_units, dtype="float32")

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step + self.initial_step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.output_units) * tf.math.minimum(arg1, arg2)


class FullGatedConv2D(Conv2D):
    def __init__(self, filters, **kwargs):
        super(FullGatedConv2D, self).__init__(filters=filters * 2, **kwargs)
        self.nb_filters = filters

    def call(self, inputs):
        outputs = super(FullGatedConv2D, self).call(inputs)
        linear = Activation("linear")(outputs[:, :, :, :self.nb_filters])
        sigmoid = Activation("sigmoid")(outputs[:, :, :, self.nb_filters:])
        return Multiply()([linear, sigmoid])

    def compute_output_shape(self, input_shape):
        output_shape = super(FullGatedConv2D, self).compute_output_shape(input_shape)
        return tuple(output_shape[:3]) + (self.nb_filters * 2,)

    def get_config(self):
        config = super(FullGatedConv2D, self).get_config()
        config['nb_filters'] = filter
        del config['filter']
        return config


def flor(input_size, output_units):
    input_data = Input(name="input", shape=input_size)

    cnn = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer="he_uniform")(
        input_data)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=16, kernel_size=(3, 3), padding="same")(cnn)

    cnn = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")(cnn)

    cnn = Conv2D(filters=40, kernel_size=(2, 4), strides=(2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=40, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=48, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=56, kernel_size=(2, 4), strides=(2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)
    cnn = FullGatedConv2D(filters=56, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
    cnn = Dropout(rate=0.2)(cnn)

    cnn = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
    cnn = PReLU(shared_axes=[1, 2])(cnn)
    cnn = BatchNormalization(renorm=True)(cnn)

    cnn = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")(cnn)

    cnn = Reshape((cnn.get_shape()[1], cnn.get_shape()[2] * cnn.get_shape()[3]))(cnn)

    bgru = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.5))(cnn)
    bgru = Dense(units=256)(bgru)

    bgru = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.5))(bgru)
    output_data = Dense(units=output_units, activation="softmax")(bgru)

    return input_data, output_data
