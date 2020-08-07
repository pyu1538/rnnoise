#!/usr/bin/env python

from __future__ import print_function

import os
import sys
import argparse
import datetime

def file_path(path):
    if os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_file:{path} is not a valid file")

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

parser = argparse.ArgumentParser()
parser.add_argument('bin_file', nargs=1, type=file_path, help='Specify the checkpoint file')
parser.add_argument('-output_path', nargs=1, type=dir_path, help='Specify the checkpoint file')
parser.add_argument('-epochs', nargs=1, type=int, help='Specify epochs count')
parser.add_argument('-checkpoint_file', nargs='*', help='Specify the checkpoint file')
args = parser.parse_args()
print('Output path: ' + str(args.output_path))

import tensorflow as tf
# from tensorflow import keras

print(tf.version.VERSION)

import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.layers import concatenate
from keras import losses
from keras import regularizers
from keras.constraints import min_max_norm
import h5py

from keras.constraints import Constraint
from keras import backend as K
import numpy as np

#import tensorflow as tf
'''
from tensorflow.python.keras import backend

config = tf.compat.v1.ConfigProto(device_count={"CPU": 40},
                        inter_op_parallelism_threads=2,
                        intra_op_parallelism_threads=2)
#config.gpu_options.per_process_gpu_memory_fraction = 0.42
backend.set_session(tf.compat.v1.Session(config=config))

physical_devices = tf.config.list_physical_devices('CPU')
logical_devices = tf.config.list_logical_devices('CPU')
print(physical_devices)
print(logical_devices)
new_ld = []
for i in range(40):
  new_ld.append(tf.config.LogicalDeviceConfiguration())
try:
  tf.config.set_logical_device_configuration(
    physical_devices[0],
    new_ld)
  logical_devices = tf.config.list_logical_devices('CPU')
  print(logical_devices)
except:
  # Cannot modify logical devices once initialized.
  print('Cannot modify logical devices once initialized')
sys.exit('test yupeng')
'''

bin_file = args.bin_file[0]
input_path = None
checkpoint_name = None
if args.checkpoint_file and len(args.checkpoint_file) > 0:
    input_path = os.path.dirname(args.checkpoint_file[0])
    checkpoint_name = os.path.basename(args.checkpoint_file[0])
output_path = args.output_path[0]

checkpoint_path = os.path.join(output_path, 'rnnoise-{epoch:04d}.ckpt')
epochs = args.epochs[0]

# Create a callback that saves the model's weights
cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=1)

def my_crossentropy(y_true, y_pred):
    return K.mean(2*K.abs(y_true-0.5) * K.binary_crossentropy(y_pred, y_true), axis=-1)

def mymask(y_true):
    return K.minimum(y_true+1., 1.)

def msse(y_true, y_pred):
    return K.mean(mymask(y_true) * K.square(K.sqrt(y_pred) - K.sqrt(y_true)), axis=-1)

def mycost(y_true, y_pred):
    return K.mean(mymask(y_true) * (10*K.square(K.square(K.sqrt(y_pred) - K.sqrt(y_true))) + K.square(K.sqrt(y_pred) - K.sqrt(y_true)) + 0.01*K.binary_crossentropy(y_pred, y_true)), axis=-1)

def my_accuracy(y_true, y_pred):
    return K.mean(2*K.abs(y_true-0.5) * K.equal(y_true, K.round(y_pred)), axis=-1)

class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
            'c': self.c}

reg = 0.000001
constraint = WeightClip(0.499)

print('Build model...')
main_input = Input(shape=(None, 42), name='main_input')
tmp = Dense(24, activation='tanh', name='input_dense', kernel_constraint=constraint, bias_constraint=constraint)(main_input)
vad_gru = GRU(24, activation='relu', recurrent_activation='sigmoid', return_sequences=True, name='vad_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(tmp)
vad_output = Dense(1, activation='sigmoid', name='vad_output', kernel_constraint=constraint, bias_constraint=constraint)(vad_gru)
noise_input = keras.layers.concatenate([tmp, vad_gru, main_input])
noise_gru = GRU(48, activation='relu', recurrent_activation='sigmoid', return_sequences=True, name='noise_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(noise_input)
denoise_input = keras.layers.concatenate([vad_gru, noise_gru, main_input])

denoise_gru = GRU(96, activation='relu', recurrent_activation='sigmoid', return_sequences=True, name='denoise_gru', kernel_regularizer=regularizers.l2(reg), recurrent_regularizer=regularizers.l2(reg), kernel_constraint=constraint, recurrent_constraint=constraint, bias_constraint=constraint)(denoise_input)

denoise_output = Dense(22, activation='sigmoid', name='denoise_output', kernel_constraint=constraint, bias_constraint=constraint)(denoise_gru)

model = Model(inputs=main_input, outputs=[denoise_output, vad_output])

model.compile(loss=[mycost, my_crossentropy],
              metrics=[msse],
              optimizer='adam', loss_weights=[10, 0.5])

#model.save_weights(checkpoint_path.format(epoch=0))
# Loads the weights
if input_path and checkpoint_name:
    ck_path = os.path.join(input_path, checkpoint_name)
    print('Loading weights...' + ck_path)
    model.load_weights(ck_path)

batch_size = 32
print('Loading data...')
with h5py.File(bin_file, 'r') as hf:
    all_data = hf['data'][:]
print('done.')

window_size = 2000

nb_sequences = len(all_data)//window_size
print(nb_sequences, ' sequences')
x_train = all_data[:nb_sequences*window_size, :42]
x_train = np.reshape(x_train, (nb_sequences, window_size, 42))

y_train = np.copy(all_data[:nb_sequences*window_size, 42:64])
y_train = np.reshape(y_train, (nb_sequences, window_size, 22))

noise_train = np.copy(all_data[:nb_sequences*window_size, 64:86])
noise_train = np.reshape(noise_train, (nb_sequences, window_size, 22))

vad_train = np.copy(all_data[:nb_sequences*window_size, 86:87])
vad_train = np.reshape(vad_train, (nb_sequences, window_size, 1))

all_data = 0;
#x_train = x_train.astype('float32')
#y_train = y_train.astype('float32')

print(len(x_train), 'train sequences. x shape =', x_train.shape, 'y shape = ', y_train.shape)

print('Train...')
model.fit(x_train, [y_train, vad_train],
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1,
          callbacks=[cp_callback])
model.save(os.path.join(output_path, 'rnnoise-weights.hdf5'))
