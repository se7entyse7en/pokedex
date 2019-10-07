import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class SiameseNetworkHandler(object):

    _input_shape = (105, 105, 3)
    _encoder = None
    _model = None

    @classmethod
    def get_encoder(cls):
        if cls._encoder is None:
            cls._encoder = keras.models.Sequential([
                layers.Conv2D(64, (10, 10), activation='relu', input_shape=cls._input_shape),
                layers.MaxPool2D((2, 2)),
                layers.Conv2D(128, (7, 7), activation='relu'),
                layers.MaxPool2D((2, 2)),
                layers.Conv2D(128, (4, 4), activation='relu'),
                layers.MaxPool2D((2, 2)),
                layers.Conv2D(256, (4, 4), activation='relu'),
                layers.Flatten(),
                layers.Dense(4096, activation='sigmoid'),
            ], name='twin_network')

        return cls._encoder

    @classmethod
    def get_model(cls):
        if cls._model is None:
            left_side_input = keras.Input(shape=cls._input_shape, name='left_side_input')
            right_side_input = keras.Input(shape=cls._input_shape, name='right_side_input')

            encoder = cls.get_encoder()

            encoded_left = encoder(left_side_input)
            encoded_right = encoder(right_side_input)

            L1_layer = layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))
            L1_distance = L1_layer([encoded_left, encoded_right])
            out = layers.Dense(1, activation='sigmoid')(L1_distance)

            cls._model = keras.Model(inputs=[left_side_input, right_side_input],
                                 outputs=out, name='siamese_network')

            optimizer = keras.optimizers.Adam(lr=0.00006)
            cls._model.compile(loss='binary_crossentropy', optimizer=optimizer)

        return cls._model
