import os
import csv
import argparse
from pathlib import Path
import time

import numpy as np
from keras.models import load_model

from dataset_generator import BreakfastActionTestDataGenerator


def read_dict(path):
    """
    Reads Python dictionary stored in a csv file
    """
    dictionary = {}
    for key, val in csv.reader(open(path)):
        dictionary[key] = eval(val)
    return dictionary


DIR_PATH = ''
PARTITION_PATH = os.path.join(DIR_PATH, 'data/segment_partition.csv')
 
file_path = 'runs/final-segment-lstm_20210220_143509_0.575.h5'

model = load_model(file_path)
model.summary()

# Data generator for test
input_dim = 400
partition = read_dict(PARTITION_PATH)
test_generator = BreakfastActionTestDataGenerator(partition['testing'],
                                                  batch_size=1,
                                                  input_dim=input_dim)

# Predict using model
print("Getting predictions...")
predictions = model.predict(test_generator,
                            workers=4,
                            verbose=2)
# print(predictions)

# Save raw predictions
model_name = file_path.split("runs/", 1)[1] 
timestr = time.strftime("%Y%m%d_%H%M%S")
print("Writing predictions...")
prediction_file_path = os.path.join(DIR_PATH, 'results/predictions_' + model_name + timestr + '.npy')
np.save(prediction_file_path, predictions)
print("predictions saved at ", prediction_file_path)

# Get final predictions (labels)
prediction_labels = np.argmax(predictions, axis=2)
print(prediction_labels)

# Create file according to submission format
print("Writing prediction labels...")
SUBMISSION_PATH = os.path.join(DIR_PATH, 'results/predictions_' + model_name + timestr + '.csv')
with open(SUBMISSION_PATH, 'w', newline='') as submission_file:
    writer = csv.writer(submission_file)
    writer.writerow(["Id", "Category"])
    for (i, label) in enumerate(prediction_labels):
        print(label)
        if label.size !=0:
            writer.writerow([i, label[0]])
submission_file.close()
print("Saved predictions to: ", SUBMISSION_PATH)
