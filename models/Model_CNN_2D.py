import tensorflow.keras as K
import tensorflow as tf

class CNN2D_Model():
    def __init__(self, kernel_size=(3,3), pooling_size=(2,2), dropout_rate=0.3, dense_units=(128,64)):
        self.kernel_size = kernel_size
        self.pooling_size = pooling_size
        self.dropout_rate = dropout_rate
        self.dense_units = dense_units

        self.base_model=None
        self.model=None

        self.build_model()

    def build_model(self):
        self.model = K.models.Sequential()

        self.model.add(K.layers.Conv2D(32, kernel_size=self.kernel_size, padding="same", activation="relu", input_shape=(128, 188, 1)))
        self.model.add(K.layers.MaxPooling2D(self.pooling_size))  
        self.model.add(K.layers.BatchNormalization())

        self.model.add(K.layers.Conv2D(64, kernel_size=self.kernel_size, padding="same", activation="relu"))
        self.model.add(K.layers.MaxPooling2D(self.pooling_size))
        self.model.add(K.layers.BatchNormalization())
        
        self.model.add(K.layers.Conv2D(128, kernel_size=self.kernel_size, padding="same", activation="relu"))
        self.model.add(K.layers.MaxPooling2D(self.pooling_size))
        self.model.add(K.layers.BatchNormalization())
        
        self.model.add(K.layers.Conv2D(256, kernel_size=self.kernel_size, padding="same", activation="relu"))
        self.model.add(K.layers.MaxPooling2D(self.pooling_size))
        self.model.add(K.layers.BatchNormalization())

        self.model.add(K.layers.Flatten())
        self.model.add(K.layers.Dense(self.dense_units[0], activation='relu'))
        self.model.add(K.layers.Dropout(self.dropout_rate))
        self.model.add(K.layers.Dense(self.dense_units[1], activation='relu'))
        self.model.add(K.layers.Dropout(self.dropout_rate))
        self.model.add(K.layers.Dense(6, activation="softmax"))

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[tf.keras.metrics.CategoricalAccuracy()]
            )
        
    def set_hyperparameters(self, hyperparameters):
        if 'kernel_size' in hyperparameters:
            self.kernel_size = hyperparameters['kernel_size']
        if 'pooling_size' in hyperparameters:
            self.pooling_size = hyperparameters['pooling_size']
        if 'dropout_rate' in hyperparameters:
            self.dropout_rate = hyperparameters['dropout_rate']
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
