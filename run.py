import os
import subprocess
import itertools
from datetime import datetime
import time
import glob
import random
import pickle
import json

import pandas as pd
import tensorflow as tf
import numpy as np

from sklearn.preprocessing import LabelEncoder
from skimage.io import imread
from skimage.transform import resize

from scipy.stats import randint, uniform

from util import etl, helpers, cnn_classifier

TRAIN_PATH = "./data/train_images.csv"    # has new augmented images
TEST_PATH = "./data/test.csv"
IMAGE_PATHS = glob.glob("./data/standardized_images/128x128/*.jpg")

VALIDATION_SIZE = 0.1
SEED = 42
np.random.seed(SEED)
TRAIN_SIZE = 1.0
CLASS_SIZE = 1.0
ITERATIONS = 1e2

params_range = {
    'conv1_num': (0, randint(1, 10)),
    'conv1_out': (2, randint(2, 8)),
    'conv2_num': (0, randint(1, 10)),
    'conv2_out': (2, randint(2, 8)),
    'd_out': (2, randint(4, 10)),
    'dropout': (0, uniform(0, 1.0)),
    'HEIGHT': 128,    # muultiple of 4 because of two k=2
    #'WIDTH': np.arange(128, 328, 4),
    'WIDTH': 128,
    'CHANNEL': 1,
    'BATCH_SIZE': 66,        # do 64 make sure not larger than VALIDATION_SIZE *
    'NUM_CLASSES': 99,
    'SEED': SEED,
    'VALIDATION_SIZE': VALIDATION_SIZE,
    'TRAIN_SIZE': TRAIN_SIZE,
    'CLASS_SIZE': CLASS_SIZE,
    'ITERATIONS': ITERATIONS,
    'LEARNING_RATE': (10, randint(-6, 1)),
    'report_interval': 10
}

def run(params_range, samplings=5):
    params = helpers.random_search(params_range, samplings)
    for param in params:
        param['WIDTH'] = param['HEIGHT']
        print 'running with the following parameters: \n %s' % json.dumps(param, indent=4)
        data = etl.load_data(train_path=TRAIN_PATH,
                             test_path=TEST_PATH,
                             image_paths=IMAGE_PATHS,
                             image_shape=(param['HEIGHT'], param['WIDTH']))
        model = cnn_classifier.CnnClassifier(data.train, data.test, data.le, param)
        model.train(param['ITERATIONS'])

if __name__ == '__main__':
    run(params_range, 2)


