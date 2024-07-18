import sys
sys.path.append("../")

import tensorflow.compat.v1 as tf
from models.vggish_utils import vggish_slim, vggish_params, vggish_input
import tf_slim as slim
import numpy as np
from tqdm import tqdm

models_path = 'Models/'

class VGGish_Model():
    def __init__(self, num_classes, checkpoint_path, train_vggish=True, param_dict=None):
        self.num_classes = num_classes
        self.checkpoint_path = checkpoint_path
        self.train_vggish = train_vggish
        self.sess = None
        self.graph = None

        self.pooling = 'mean'
        self.dense_units = 64
        self.dropout_rate = 0.5
        if param_dict is not None:
            self.set_hyperparameters(param_dict)

        self.build_model()

    def set_hyperparameters(self, hyperparameters):
        if 'pooling' in hyperparameters:
            self.pooling = hyperparameters['pooling']
        if 'dense_units' in hyperparameters:
            self.dense_units = hyperparameters['dense_units']
        if 'dropout_rate' in hyperparameters:
            self.dropout_rate = hyperparameters['dropout_rate']
        

    def build_model(self):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        
        with self.graph.as_default():
            # Define VGGish.
            embeddings = vggish_slim.define_vggish_slim(training=self.train_vggish)

            # Define a shallow classification model and associated training ops on top
            # of VGGish.
            with tf.variable_scope('mymodel'):
                # Add a fully connected layer with 100 units. Add an activation function
                # to the embeddings since they are pre-activation.

                #flattened_embedding = tf.reshape(embeddings, [1, 3*embeddings.shape[1]])
                # Postprocess the results to produce whitened quantized embeddings.
                if self.pooling == 'mean':
                    pool = tf.reduce_mean(embeddings, axis=0)
                elif self.pooling == 'max':
                    pool = tf.reduce_max(embeddings, axis=0)
                reshape = tf.reshape(pool, [1, 128])
                tf.identity(pool, name='flattened')
                fc1 = slim.fully_connected(tf.nn.relu(reshape), self.dense_units)
                dp = tf.layers.dropout(fc1, rate=self.dropout_rate)
                fc2 = slim.fully_connected(dp, self.dense_units)

                # Add a classifier layer at the end, consisting of parallel logistic
                # classifiers, one per class. This allows for multi-class tasks.
                logits = slim.fully_connected(fc2, self.num_classes, activation_fn=None, scope='logits')
                self.prediction = tf.nn.softmax(logits, name='prediction')

                # Add training ops.
                with tf.variable_scope('train'):
                    global_step = tf.train.create_global_step()

                    # Labels are assumed to be fed as batch multi-hot vectors, with
                    # a 1 in the position of each positive class label, and 0 elsewhere.
                    self.labels_input = tf.placeholder(tf.float32, shape=(self.num_classes), name='labels_input')

                    # Cross-entropy label loss.
                    xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels_input)
                    self.loss = tf.reduce_mean(xent, name='loss_op')
                    tf.summary.scalar('loss', self.loss)

                    # We use the same optimizer and hyperparameters as used to train VGGish.
                    optimizer = tf.train.AdamOptimizer(
                        learning_rate=vggish_params.LEARNING_RATE,
                        epsilon=vggish_params.ADAM_EPSILON)
                    self.train_op = optimizer.minimize(self.loss, global_step=global_step)

            # Initialize variables
            self.sess.run(tf.global_variables_initializer())

            # Load pre-trained VGGish checkpoint
            print("Loading VGGish checkpoint...")
            vggish_slim.load_vggish_slim_checkpoint(self.sess, self.checkpoint_path)
            print("DONE\n")

    def get_model_dict(self):
        features_tensor = self.sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = self.sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)
        layers = {'conv1': 'vggish/conv1/Relu',
                    'pool1': 'vggish/pool1/MaxPool',
                    'conv2': 'vggish/conv2/Relu',
                    'pool2': 'vggish/pool2/MaxPool',
                    'conv3': 'vggish/conv3/conv3_2/Relu',
                    'pool3': 'vggish/pool3/MaxPool',
                    'conv4': 'vggish/conv4/conv4_2/Relu',
                    'pool4': 'vggish/pool4/MaxPool',
                    'fc1': 'vggish/fc1/fc1_2/Relu',
                    #'fc2': 'vggish/fc2/Relu',
                    'embedding': 'vggish/embedding',
                    'features': 'vggish/input_features', 
                    'prediction': 'mymodel/prediction',
                    'flattened': 'mymodel/flattened'
                }
        for k in layers:
            layers[k] = self.graph.get_tensor_by_name( layers[k] + ':0')

        return {'features': features_tensor,
                'embedding': embedding_tensor,
                'layers': layers,
                }
    
    def EmbeddingsFromVGGish(self, vgg, x, sr):
        '''Run the VGGish model, starting with a sound (x) at sample rate
        (sr). Return a dictionary of embeddings from the different layers
        of the model.'''
        # Produce a batch of log mel spectrogram examples.
        input_batch = vggish_input.waveform_to_examples(x, sr)
        # print('Log Mel Spectrogram example: ', input_batch[0])
        layer_names = vgg['layers'].keys()
        tensors = [vgg['layers'][k] for k in layer_names]
        results = self.sess.run(tensors,
                            feed_dict={vgg['features']: input_batch})
        resdict = {}
        for i, k in enumerate(layer_names):
            resdict[k] = results[i]
        return resdict
    
    def get_vggish_input_format(self, x_train, x_test):
        tmp_x_train = []
        tmp_x_test = []

        print("Converting train data...")
        for _, x in tqdm(enumerate(x_train), total=len(x_train), smoothing=0.9):
            tmp_x_train.append(vggish_input.wavfile_to_examples(x))

        print("Converting test data...")
        for _, x in tqdm(enumerate(x_test), total=len(x_test), smoothing=0.9):
            tmp_x_test.append(vggish_input.wavfile_to_examples(x))

        return np.array(tmp_x_train), np.array(tmp_x_test)

    def get_predicted_class(self, original_tensor):
        # Find the index of the maximum value in the vector
        argmax_indices = np.argmax(original_tensor, axis=1)
        # Create a binary tensor with 1 at the argmax positions and 0 elsewhere
        binary_tensor = np.zeros_like(original_tensor)
        rows = np.arange(original_tensor.shape[0])
        binary_tensor[rows, argmax_indices] = 1

        return binary_tensor

    def fit(self, x_train, y_train, batch_size, validation_data, epochs, validation_batch_size, callbacks=None):
        features_input = self.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        for epoch in range(epochs):
            # Training phase
            loss_train_list = []
            accuracy_train_list = []
            for i, _ in tqdm(enumerate(range(0, x_train.shape[0], 1)), total=len(range(0, x_train.shape[0], 1)), smoothing=0.9):
                _, train_loss_value, train_prediction = self.sess.run(
                    [self.train_op, self.loss, self.prediction],
                    feed_dict={features_input: vggish_input.wavfile_to_examples(x_train[i]), self.labels_input: y_train[i]})
                train_accuracy = np.sum(np.all(self.get_predicted_class(train_prediction)[0] == y_train[i], axis=0))
                accuracy_train_list.append(train_accuracy)
                loss_train_list.append(train_loss_value)

            # Validation phase (after completing an epoch)
            loss_val_list = []
            accuracy_val_list = []
            for i, _ in tqdm(enumerate(range(0, validation_data[0].shape[0], 1)), total=len(range(0, validation_data[0].shape[0], 1)), smoothing=0.9):
                val_loss_value, val_prediction = self.sess.run(
                    [self.loss, self.prediction],
                    feed_dict={features_input: vggish_input.wavfile_to_examples(validation_data[0][i]), self.labels_input: validation_data[1][i]})
                val_accuracy =  np.sum(np.all(self.get_predicted_class(val_prediction)[0] == validation_data[1][i], axis=0))
                accuracy_val_list.append(val_accuracy)
                loss_val_list.append(val_loss_value)

            print(f'Epoch {epoch + 1}, Train Loss: {round(np.mean(loss_train_list),4)}, Train Accuracy: {round(np.mean(accuracy_train_list),4)}, Val Loss: {round(np.mean(loss_val_list),4)}, Val Accuracy: {round(np.mean(accuracy_val_list),4)}')
        
    def train_and_evaluate(self, x_train, y_train, validation_data, x_test, y_test, epochs, patience=4):
        print(x_train.shape, validation_data[0].shape, y_train.shape, validation_data[1].shape)
        features_input = self.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        count = 0
        accuracy_test = float('-inf')
        best_accuracy_val = float('-inf')
        best_y_pred_list = None
        best_y_gt_list = None
        for epoch in range(epochs):
            # If the validation accuracy does not improve
            if count == patience:
                print("stop")
                return round(accuracy_test,4), best_y_pred_list, best_y_gt_list
            # Training phase
            loss_train_list = []
            accuracy_train_list = []
            for i, _ in tqdm(enumerate(range(0, x_train.shape[0], 1)), total=len(range(0, x_train.shape[0], 1)), smoothing=0.9):
                _, train_loss_value, train_prediction = self.sess.run(
                    [self.train_op, self.loss, self.prediction],
                    feed_dict={features_input: vggish_input.wavfile_to_examples(x_train[i]), self.labels_input: y_train[i]})
                train_accuracy = np.sum(np.all(self.get_predicted_class(train_prediction)[0] == y_train[i], axis=0))
                accuracy_train_list.append(train_accuracy)
                loss_train_list.append(train_loss_value)

            # Validation phase (after completing an epoch)
            loss_val_list = []
            accuracy_val_list = []
            for i, _ in tqdm(enumerate(range(0, validation_data[0].shape[0], 1)), total=len(range(0, validation_data[0].shape[0], 1)), smoothing=0.9):
                val_loss_value, val_prediction = self.sess.run(
                    [self.loss, self.prediction],
                    feed_dict={features_input: vggish_input.wavfile_to_examples(validation_data[0][i]), self.labels_input: validation_data[1][i]})
                val_accuracy =  np.sum(np.all(self.get_predicted_class(val_prediction)[0] == validation_data[1][i], axis=0))
                accuracy_val_list.append(val_accuracy)
                loss_val_list.append(val_loss_value)

            #print(f'Epoch {epoch + 1}, Train Loss: {round(np.mean(loss_train_list),4)}, Train Accuracy: {round(np.mean(accuracy_train_list),4)}, Val Loss: {round(np.mean(loss_val_list),4)}, Val Accuracy: {round(np.mean(accuracy_val_list),4)}')

            loss_test_list = []
            accuracy_test_list = []
            y_pred_list = []
            y_gt_list = []
            for i, _ in tqdm(enumerate(range(0, x_test.shape[0], 1)), total=len(range(0, x_test.shape[0], 1)), smoothing=0.9):
                test_loss_value, test_prediction = self.sess.run(
                    [self.loss, self.prediction],
                    feed_dict={features_input: vggish_input.wavfile_to_examples(x_test[i]), self.labels_input: y_test[i]})
                test_accuracy = np.sum(np.all(self.get_predicted_class(test_prediction)[0] == y_test[i], axis=0))
                y_pred_list.append(self.get_predicted_class(test_prediction)[0])
                y_gt_list.append(y_test[i])
                accuracy_test_list.append(test_accuracy)
                loss_test_list.append(test_loss_value)
            
            print(f'Epoch {epoch + 1}, Train Loss: {round(np.mean(loss_train_list),4)}, Train Accuracy: {round(np.mean(accuracy_train_list),4)}, Val Loss: {round(np.mean(loss_val_list),4)}, Val Accuracy: {round(np.mean(accuracy_val_list),4)}')
            print(f'Evaluation on Test Set -> Test Loss: {round(np.mean(loss_test_list),4)}, Test Accuracy: {round(np.mean(accuracy_test_list),4)}')

            if np.mean(accuracy_val_list) > best_accuracy_val:
                best_accuracy_val = np.mean(accuracy_val_list)
                accuracy_test = np.mean(accuracy_test_list)
                best_y_pred_list = y_pred_list
                best_y_gt_list = y_gt_list
                count = 0
            else:
                count += 1