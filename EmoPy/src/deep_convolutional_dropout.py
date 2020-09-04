from EmoPy.src.neuralnets import _FERNeuralNet
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Conv2D, ConvLSTM2D, Conv3D, MaxPooling2D, Dropout, \
    MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.losses import categorical_crossentropy
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.regularizers import l2
import json

from EmoPy.src.callback import PlotLosses


class DeepConvolutionalNNDropout(_FERNeuralNet):

    def __init__(self, image_size, channels, emotion_map,
                 verbose=False):
        self.channels = channels
        self.image_size = image_size
        self.verbose = verbose

        super().__init__(emotion_map)

    def fit(self, image_data, labels, validation_split, epochs=50):
        """
        Trains the neural net on the data provided.

        :param image_data: Numpy array of training data.
        :param labels: Numpy array of target (label) data.
        :param validation_split: Float between 0 and 1. Percentage of training data to use for validation
        :param batch_size:
        :param epochs: number of times to train over input dataset.
        """
        adam = optimizers.Adam(lr=0.001)
        model.compile(optimizer=adam,
                      loss='categorical_crossentropy', metrics=['accuracy'])

        # Add model checkpoint to save the model after each epoch
        history = self.model.fit(image_data, labels, epochs=epochs, validation_split=validation_split,
                                 callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3),
                                            EarlyStopping(
                                     monitor='val_acc', min_delta=0, patience=6, mode='auto'),
                                     ModelCheckpoint('../examples/output/checkpoints/deep_conv_dropout_weights.hd5',
                                                     monitor='val_loss', verbose=1, save_best_only=True)])
        return history

    def _init_model(self):
        model = Sequential()

        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(
            48, 48, 1), kernel_regularizer=l2(0.01), data_format='channels_last'))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.emotion_map.keys()), activation='softmax'))
        if self.verbose:
            model.summary()
        self.model = model
