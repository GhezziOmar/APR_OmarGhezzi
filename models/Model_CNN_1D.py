import tensorflow.keras as K
import tensorflow as tf
from tensorflow.keras import regularizers

class CNN1D_Model():
    def __init__(self, kernel_size=10, conv_stride_kernel=1, pooling_size=2, pool_stride_kernel=2, dense_units=(256, 128)):
        self.kernel_size = kernel_size
        self.conv_stride_kernel = conv_stride_kernel
        self.pooling_size = pooling_size
        self.pool_stride_kernel = pool_stride_kernel
        self.dense_units = dense_units

        self.base_model=None
        self.model=None

        self.build_model()

    def build_model(self):
        self.model = K.models.Sequential()

        self.model.add(K.layers.Conv1D(128, kernel_size=self.kernel_size, strides=self.conv_stride_kernel, padding="same", activation="relu", input_shape=(128, 1)))
        self.model.add(K.layers.BatchNormalization())
        self.model.add(K.layers.MaxPool1D(pool_size=self.pooling_size, strides=self.pool_stride_kernel, padding="same"))
        self.model.add(K.layers.Dropout(0.2))

        self.model.add(K.layers.Conv1D(128, kernel_size=self.kernel_size, strides=self.conv_stride_kernel, padding="same", activation="relu"))
        self.model.add(K.layers.BatchNormalization())
        self.model.add(K.layers.MaxPool1D(pool_size=self.pooling_size, strides=self.pool_stride_kernel, padding="same"))
        self.model.add(K.layers.Dropout(0.2))

        self.model.add(K.layers.Flatten())
        self.model.add(K.layers.Dense(self.dense_units[0], activation='relu'))
        self.model.add(K.layers.Dropout(0.3))
        self.model.add(K.layers.Dense(self.dense_units[1], activation='relu'))
        self.model.add(K.layers.Dropout(0.3))
        self.model.add(K.layers.Dense(6, activation="softmax", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[tf.keras.metrics.CategoricalAccuracy()]
            )
        
    def set_hyperparameters(self, hyperparameters):
        if 'kernel_size' in hyperparameters:
            self.kernel_size = hyperparameters['kernel_size']
        if 'conv_stride_kernel' in hyperparameters:
            self.conv_stride_kernel = hyperparameters['conv_stride_kernel']
        if 'pooling_size' in hyperparameters:
            self.pooling_size = hyperparameters['pooling_size']
        if 'pool_stride_kernel' in hyperparameters:
            self.pool_stride_kernel = hyperparameters['pool_stride_kernel']
        if 'dense_units' in hyperparameters:
            self.dense_units = hyperparameters['dense_units']

        self.build_model()
        self.model.summary()

    def fit(self, x_train, y_train, batch_size, validation_data, epochs, validation_batch_size, callbacks=None):
        history = self.model.fit(x=x_train, y=y_train, batch_size=batch_size, validation_data=validation_data, epochs=epochs, validation_batch_size=validation_batch_size, callbacks=callbacks)
        return history
    
    def evaluate(self, x_test, y_test, verbose):
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=verbose)
        return loss, accuracy

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred
    
    def summary(self):
        self.model.summary()
