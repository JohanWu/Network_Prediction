import random
import numpy as np
import keras
from keras.models import Sequential
from keras import layers

from matplotlib import pyplot as plt
from IPython.display import clear_output

# import tensorflow
# print('tensorflow version:', tensorflow.__version__)

ADD_LAYERS = 3 # Total layers will be 'ADD_LAYERS+2'

## Different RNN cells: GRU or SimpleRNN.
# RNN = layers.GRU
# RNN = layers.LSTM

## updatable plot
## a minimal example (sort of)
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show()
    
## LSTM parameters:    
# keras.layers.LSTM(units, 
#     activation='linear', 
#     recurrent_activation='hard_sigmoid', 
#     use_bias=True, 
#     kernel_initializer='glorot_uniform', 
#     recurrent_initializer='orthogonal', 
#     bias_initializer='zeros', 
#     unit_forget_bias=True, 
#     kernel_regularizer=None, 
#     recurrent_regularizer=None, 
#     bias_regularizer=None, 
#     activity_regularizer=None, 
#     kernel_constraint=None, 
#     recurrent_constraint=None, 
#     bias_constraint=None, 
#     dropout=0.0, 
#     recurrent_dropout=0.0, 
#     implementation=1, 
#     return_sequences=False, 
#     return_state=False, 
#     go_backwards=False, 
#     stateful=False, 
#     unroll=False)


def generate_data(tn_lfv_seq, timestep, ratio_validation): 
    '''Generate the training examples and validation examples from the entire latent feature vector sequence.

    input shape: (time, node, len_lfv)
    output: a list of (training_samples, validation_samples, testing_samples) for all nodes
    '''
    print("(#period, #node, len_lfv):", tn_lfv_seq.shape)
    nt_lfv_seq = np.swapaxes(tn_lfv_seq, 0, 1)
    print("(#node, #period, len_lfv):", nt_lfv_seq.shape)
    
    data = list()
    for n in range(nt_lfv_seq.shape[0]):
        ### generate training samples, validation samples, and the testing sample for each node
        # print((nt_lfv_seq[n]))
        ## get data samples from the sequence
        ## the length of the subsequence of latent feature vectors is timestep+1. The last vector is the vector for the label.
        samples = list()
        for t in range(timestep, nt_lfv_seq.shape[1]-1):
            samples.append(nt_lfv_seq[n][t-timestep:t+1])
        samples = np.array(samples)
        # print("shape of samples (#all_samples, timestep+1, len_lfv):", samples.shape)
        
        ## split training set and validation set
        np.random.shuffle(samples)
        training = list()
        validation = list()
        for sample in samples:
            if random.random() < ratio_validation:
                validation.append(sample)
            else:
                training.append(sample)
        training, validation = np.array(training), np.array(validation)
        # print("shape of training set:", training.shape)
        # print("shape of validation set:", validation.shape)

        ## get testing set
        testing = nt_lfv_seq[n][-1-timestep:]
        # print("shape of testing set:", testing.shape)
        
        data.append((training, validation, testing))
        
        # print("for node %d:" %n, data[n][0].shape, data[n][1].shape, data[n][2].shape)

    return data

def build_model(len_lfv, hidden_size, timestep, rnn_cell, activation_function='linear', loss_function='mse'):
    print('Build model...')
    model = Sequential()
    ## Multiple hidden layers
    '''
    model.add(rnn_cell(hidden_size, input_shape=(timestep, len_lfv), activation='linear', return_sequences=True))
    for _ in range(ADD_LAYERS):
        model.add(rnn_cell(hidden_size, activation='linear', return_sequences=True))
    '''
    model.add(rnn_cell(hidden_size, 
        input_shape=(timestep, len_lfv),
        activation=activation_function))
    model.add(layers.Dense(len_lfv))
    model.compile(loss=loss_function, #cosine_proximity, kullback_leibler_divergence
                  optimizer='adam',
                  metrics=['mse', 'mae', 'mape', 'cosine'])

    print(model.summary())
    return model
