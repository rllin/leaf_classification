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

TRAIN_PATH = "./data/train.csv"
TEST_PATH = "./data/test.csv"
IMAGE_PATHS = glob.glob("./data/images/*.jpg")

VALIDATION_SIZE = 0.1
SEED = 42
TRAIN_SIZE = 1.0
ITERATIONS = 1e2

params_range = {
    'conv1_num': (0, randint(1, 10)),
    'conv1_out': (2, randint(2, 8)),
    'conv2_num': (0, randint(1, 10)),
    'conv2_out': (2, randint(2, 8)),
    'd_out': (2, randint(4, 10)),
    'dropout': (0, uniform(0, 1.0)),
    'HEIGHT': (0, randint(170, 1700)),
    'WIDTH': (0, randint(170, 1700)),
    'CHANNEL': 1,
    'BATCH_SIZE': 64,
    'NUM_CLASSES': 99,
    'VALIDATION_SIZE': VALIDATION_SIZE,
    'SEED': SEED,
    'TRAIN_SIZE': TRAIN_SIZE,
    'CLASS_SIZE': 0.1,
    'ITERATIONS': ITERATIONS,
    'LEARNING_RATE': (10, randint(-6, 1)),
    'report_interval': 1
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
    run(params_range, 5)


