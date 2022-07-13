
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from spectraclass.util.logs import LogManager, lgm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Input, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras import backend as K


class VAE:
    """ Variational auto encoder
    Encoder maps X onto a latent space Z
    Decoder samples Z from N(0,1)
    VAE_loss = Reconstruction_loss + KL_loss

    Reference
    See :cite:`kingma2013auto` Kingma, Diederik, Welling
    'Auto-Encodeing Variational Bayes'
    https://arxiv.org/abs/1312.6114 for details.

    beta VAE
    In Loss, the emphasis is on KL_loss
    and capacity of a bottleneck:
    VAE_loss = Reconstruction_loss + gamma*KL_loss

    Reference
    See :cite:`burgess2018understanding` Burges et al
    'Understanding disentangling in beta-VAE'
    https://arxiv.org/pdf/1804.03599.pdf for details.


    Parameters
    ----------
    encoder_neurons : list, optional (default=[128, 64, 32])
        The number of neurons per hidden layer in encoder.

    decoder_neurons : list, optional (default=[32, 64, 128])
        The number of neurons per hidden layer in decoder.

    hidden_activation : str, optional (default='relu')
        Activation function to use for hidden layers.
        All hidden layers are forced to use the same type of activation.
        See https://keras.io/activations/

    output_activation : str, optional (default='sigmoid')
        Activation function to use for output layer.
        See https://keras.io/activations/

    loss : str or obj, optional (default=keras.losses.mean_squared_error
        String (name of objective function) or objective function.
        See https://keras.io/losses/

    gamma : float, optional (default=1.0)
        Coefficient of beta VAE regime.
        Default is regular VAE.

    capacity : float, optional (default=0.0)
        Maximum capacity of a loss bottle neck.

    optimizer : str, optional (default='adam')
        String (name of optimizer) or optimizer instance.
        See https://keras.io/optimizers/

    epochs : int, optional (default=100)
        Number of epochs to train the model.

    batch_size : int, optional (default=32)
        Number of samples per gradient update.

    dropout_rate : float in (0., 1), optional (default=0.2)
        The dropout to be used across all layers.

    l2_regularizer : float in (0., 1), optional (default=0.1)
        The regularization strength of activity_regularizer
        applied on each layer. By default, l2 regularizer is used. See
        https://keras.io/regularizers/

    validation_size : float in (0., 1), optional (default=0.1)
        The percentage of data to be used for validation.

    preprocessing : bool, optional (default=True)
        If True, apply standardization on the data.

    verbose : int, optional (default=1)
        verbose mode.

        - 0 = silent
        - 1 = progress bar
        - 2 = one line per epoch.

        For verbose >= 1, model summary may be printed.

    random_state : random_state: int, RandomState instance or None, opti
        (default=None)
        If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the r
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. When fitting this is
        to define the threshold on the decision function.

    Attributes
    ----------
    encoding_dim_ : int
        The number of neurons in the encoding layer.

    compression_rate_ : float
        The ratio between the original feature and
        the number of neurons in the encoding layer.

    model_ : Keras Object
        The underlying AutoEncoder in Keras.

    history_: Keras Object
        The AutoEncoder training history.

    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self,  latent_dim, hidden_activation='relu', output_activation='sigmoid', loss=mse, optimizer='adam',
                 epochs=32, batch_size=32, dropout_rate=0.1, l2_regularizer=0.1, validation_size=0.1, preprocessing=True,
                 verbose=1, random_state=None,  gamma=1.0, capacity=0.0):
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.l2_regularizer = l2_regularizer
        self.validation_size = validation_size
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.random_state = random_state
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.capacity = capacity

    def sampling(self, args):
        """Reparametrisation by sampling from Gaussian, N(0,I)
        To sample from epsilon = Norm(0,I) instead of from likelihood Q(z|X)
        with latent variables z: z = z_mean + sqrt(var) * epsilon

        Parameters
        ----------
        args : tensor
            Mean and log of variance of Q(z|X).

        Returns
        -------
        z : tensor
            Sampled latent variable.
        """

        z_mean, z_log = args
        batch = K.shape(z_mean)[0]  # batch size
        dim = K.int_shape(z_mean)[1]  # latent dimension
        epsilon = K.random_normal(shape=(batch, dim))  # mean=0, std=1.0

        return z_mean + K.exp(0.5 * z_log) * epsilon

    def vae_loss(self, inputs, outputs, z_mean, z_log):
        """ Loss = Recreation loss + Kullback-Leibler loss
        for probability function divergence (ELBO).
        gamma > 1 and capacity != 0 for beta-VAE
        """
        reconstruction_loss = self.loss(inputs, outputs)
        reconstruction_loss *= self.n_features_
        kl_loss = 1 + z_log - K.square(z_mean) - K.exp(z_log)
        kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
        kl_loss = self.gamma * K.abs(kl_loss - self.capacity)

        return K.mean(reconstruction_loss + kl_loss)

    def _build_model(self):
        self.compute_network_layer_sizes()
        inputs = Input(shape=(self.n_features_,))
        layer = Dense(self.n_features_, activation=self.hidden_activation)(inputs)
        for neurons in self.encoder_neurons:
            layer = Dense(neurons, activation=self.hidden_activation, activity_regularizer=l2(self.l2_regularizer))(layer)
            layer = Dropout(self.dropout_rate)(layer)

        z_mean = Dense(self.latent_dim)(layer)
        z_log = Dense(self.latent_dim)(layer)
        z = Lambda(self.sampling, output_shape=(self.latent_dim,))([z_mean, z_log])

        encoder = Model(inputs, [z_mean, z_log, z])
        if self.verbose >= 1: encoder.summary()

        latent_inputs = Input(shape=(self.latent_dim,))
        layer = Dense(self.latent_dim, activation=self.hidden_activation)(latent_inputs)
        for neurons in self.decoder_neurons:
            layer = Dense(neurons, activation=self.hidden_activation)(layer)
            layer = Dropout(self.dropout_rate)(layer)
        outputs = Dense(self.n_features_, activation=self.output_activation)(layer)
        decoder = Model(latent_inputs, outputs)
        if self.verbose >= 1: decoder.summary()

        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs)
        vae.add_loss(self.vae_loss(inputs, outputs, z_mean, z_log))
        vae.compile(optimizer=self.optimizer)
        if self.verbose >= 1: vae.summary()
        return vae

    def compute_network_layer_sizes(self):
        nsize = self.n_features_
        self.encoder_neurons, self.decoder_neurons = [], []
        while True:
            self.encoder_neurons.append( nsize )
            nsize = nsize // 2
            if nsize <= self.latent_dim: break

        self.decoder_neurons = self.encoder_neurons[::-1]


    def fit(self, X: np.ndarray, y: np.ndarray=None ):

        self.n_samples_, self.n_features_ = X.shape[0], X.shape[1]

        if self.preprocessing:
            self.scaler_ = StandardScaler()
            X_norm = self.scaler_.fit_transform(X)
        else:
            X_norm = np.copy(X)

        np.random.shuffle(X_norm)
        self.model_ = self._build_model()
        self.history_ = self.model_.fit(X_norm,
                                        epochs=self.epochs,
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        validation_split=self.validation_size,
                                        verbose=self.verbose).history

        return self

    def _build_network( self, input_dims: int, model_dims: int, **kwargs  ):
        from tensorflow.keras.layers import Input, Dense
        from tensorflow.keras.models import Model
        from tensorflow.keras import losses, regularizers
        lgm().log(f"#AEC: BUILD AEC NETWORK")
        sparsity: float = kwargs.get( 'sparsity', 0.0 )
        reduction_factor = 2
        inputlayer = Input(shape=[input_dims])
        activation = 'tanh'
        optimizer = 'rmsprop'
        layer_dims, x = int(round(input_dims / reduction_factor)), inputlayer
        while layer_dims > model_dims:
            x = Dense(layer_dims, activation=activation)(x)
            layer_dims = int(round(layer_dims / reduction_factor))
        encoded = x = Dense(model_dims, activation=activation, activity_regularizer=regularizers.l1(sparsity))(x)
        layer_dims = int(round(model_dims * reduction_factor))
        while layer_dims < input_dims:
            x = Dense(layer_dims, activation=activation)(x)
            layer_dims = int(round(layer_dims * reduction_factor))
        decoded = Dense(input_dims, activation='linear')(x)
        #        modelcheckpoint = ModelCheckpoint('xray_auto.weights', monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
        #        earlystopping = EarlyStopping(monitor='loss', min_delta=0., patience=100, verbose=1, mode='auto')
        self._autoencoder = Model(inputs=[inputlayer], outputs=[decoded])
        self._encoder = Model(inputs=[inputlayer], outputs=[encoded])
        #        autoencoder.summary()
        #        encoder.summary()
        self._autoencoder.compile( loss=self.loss, optimizer=optimizer )
        self.log = lgm().log(f" BUILD Autoencoder network: input_dims = {input_dims} ")


