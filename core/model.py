import os
import random
import numpy as np
import tensorflow as tf


class AutoEncoderModelArchitecture:
    """
    Deterministic, feed-forward autoencoder architecture for representation learning
    and dimensionality reduction.

    This class implements a symmetric autoencoder using fully connected layers with
    batch normalization and L2 regularization in the encoder. The architecture is
    designed to learn stable, low-dimensional latent representations suitable for
    downstream tasks such as clustering, segmentation, or feature extraction.

    Key design characteristics:
    - Linear latent space to preserve geometric structure for distance-based methods
    - Encoder-side L2 regularization to prevent identity mapping
    - Batch normalization in the encoder for training stability
    - Layer normalization on the latent space to control scale drift
    - Mean Squared Error (MSE) reconstruction objective

    Reproducibility considerations:
    - Random seeds are set for Python, NumPy, and TensorFlow
    - Deterministic TensorFlow operations are requested where supported
    - CUDA devices are explicitly disabled to favor deterministic CPU execution
    - Full bitwise determinism is not guaranteed due to floating-point and optimizer behavior

    Parameters
    ----------
    seed : int
        Random seed used to initialize Python, NumPy, and TensorFlow randomness.

    latent_space : int
        Dimensionality of the latent (bottleneck) representation. Must be smaller (< input_space//2)
        than the input dimensionality.

    input_space : int
        Dimensionality of the input feature space.

    Attributes
    ----------
    model : tf.keras.Model
        The full autoencoder model mapping inputs to reconstructed outputs.

    encoder : tf.keras.Model
        Encoder sub-model mapping inputs to the latent representation.

    Notes
    -----
    This implementation is intended for structured/tabular data. It is not designed
    for convolutional, sequential, or variational autoencoder use cases.

    The learned latent representations can be accessed via `get_encoded_input`
    and are suitable for use with clustering algorithms such as KMeans, GMM,
    or hierarchical clustering.
    """
    def __init__(self, seed: int, latent_space: int, input_space: int):
        self._set_seed(seed)
        self._disable_cuda()

        self.seed = seed
        self.latent_space = latent_space
        self.input_space = input_space

        self.model = self._build_model()
        self._compile_model()

        self.encoder = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer("latent_space").output)

    def _set_seed(self, seed: int):
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Enforce deterministic ops where possible
        os.environ["TF_DETERMINISTIC_OPS"] = "1"

    def _disable_cuda(self):
        # Disable all GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        tf.config.set_visible_devices([], "GPU")

    def _build_model(self) -> tf.keras.models.Model:
        inp = tf.keras.layers.Input(shape=(self.input_space,), name="input_space")
        
        # encoder 
        x = tf.keras.layers.BatchNormalization()(inp)
        x = tf.keras.layers.Dense(self.input_space // 2, activation="relu", 
                                  use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        latent = tf.keras.layers.Dense(self.latent_space, activation="linear", 
                                       use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
        latent = tf.keras.layers.LayerNormalization(name="latent_space")(latent)

        # decoder 
        x = tf.keras.layers.Dense(self.input_space // 2, activation="relu")(latent)
        
        out = tf.keras.layers.Dense(self.input_space, activation="linear", name="reconstruction")(x)
        
        return tf.keras.models.Model(inputs=inp, outputs=out, name="autoencoder")

    def _compile_model(self):
        self.model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.MeanAbsoluteError()])
    
    def summary(self):
        return self.model.summary()

    def fit(self, x, y, **kwargs):
        return self.model.fit(x, y, 
                              callbacks=tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                                         patience=5,
                                                                         restore_best_weights=True), 
                               **kwargs)

    def get_reconstructed_input(self, x):
        return self.model.predict(x)
    
    def get_encoded_input(self, x):
        return self.encoder.predict(x)
    def save(self, path:str):
        self.model.save(path)
        return None
