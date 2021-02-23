import os
import time
import csv
import numpy as np
import pickle

import matplotlib.pyplot as plt
from keras import optimizers
from keras import regularizers
from keras.layers import Bidirectional, Dense, LSTM, MaxPooling1D, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

from dataset_generator import BreakfastActionTrainDataGenerator

import tensorflow as tf
from tensorflow.python.client import device_lib
from keras import backend as K


def read_dict(path):
    """
    Reads Python dictionary stored in a csv file
    """
    dictionary = {}
    for key, val in csv.reader(open(path)):
        dictionary[key] = eval(val)
    return dictionary


# Check if program is running on GPU
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
print(device_lib.list_local_devices())

DIR_PATH = ''
PARTITION_PATH = os.path.join(DIR_PATH, 'data/segment_partition.csv')
SEGMENT_LABELS_PATH = os.path.join(DIR_PATH, 'data/segment_labels.csv')

# Values for model architecture
batch_size = 16  # number of segments for an iteration of training
input_dim = 400  # dimension of an i3D video frame
hidden_dim = 400  # dimension of RNN hidden state
layer_dim = 1  # number of hidden RNN layers
output_dim = 48  # number of sub-action labels
num_epochs = 20

# Define LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(hidden_dim, return_sequences=True), input_shape=(None, input_dim)))
model.add(Bidirectional(LSTM(hidden_dim, return_sequences=True)))
model.add(MaxPooling1D(pool_size=6000, strides=None, padding='valid', data_format='channels_last'))
model.add(Dropout(rate=0.4))
model.add(Dense(output_dim, activation='softmax'))

# model.load_weights('./runs/segment-lstm-19-0.64.hdf5')

# Checkpoint callback
checkpoint_filename = "./runs/segment-lstm-{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(checkpoint_filename, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
callbacks_list = [checkpoint]

model.compile('adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

partition = read_dict(PARTITION_PATH)
# print(partition['training'])
# print(partition['validation'])

# Load labels
labels = read_dict(SEGMENT_LABELS_PATH)
# print(labels)

# Data generators for train/validation
training_generator = BreakfastActionTrainDataGenerator(partition['training'],
                                                       labels=labels,
                                                       batch_size=batch_size,
                                                       input_dim=input_dim,
                                                       output_dim=output_dim,
                                                       shuffle=True)
validation_generator = BreakfastActionTrainDataGenerator(partition['validation'],
                                                         labels=labels,
                                                         batch_size=batch_size,
                                                         input_dim=input_dim,
                                                         output_dim=output_dim,
                                                         shuffle=True)

# Train model
history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    workers=4,
                    epochs=num_epochs,
                    verbose=1,
                    callbacks=callbacks_list)

# Evaluate model
val_loss, val_acc = model.evaluate_generator(generator=validation_generator,
                                             workers=4,
                                             verbose=1)

# Save model
timestr = time.strftime("%Y%m%d_%H%M%S_")
model_filename = "./runs/final-segment-lstm_" + timestr + str(round(val_acc, 3)) + ".h5"
model.save(model_filename)

# Save accuracy plot
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("./runs/figures/final-segment-lstm_" + timestr + str(round(val_acc, 3)) + "_acc" + ".png")

# Save loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig("./runs/figures/final-segment-lstm_" + timestr + str(round(val_acc, 3)) + "_loss" + ".png")