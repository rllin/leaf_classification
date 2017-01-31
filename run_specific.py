
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
if __name__ == '__main__':
    TRAIN_PATH = "./data/train_images.csv"    # has new augmented images
    TRAIN_PATH = "./data/train.csv"    # has new augmented images
    TEST_PATH = "./data/test.csv"
    IMAGE_PATHS = glob.glob("./data/128x128/*.jpg")
    IMAGE_PATHS = [e for e in IMAGE_PATHS if int(os.path.basename(os.path.splitext(e)[0])) <= 1584]


    fixed_params = {
        'HEIGHT': 128,    # muultiple of 4 because of two k=2
        #'WIDTH': np.arange(128, 328, 4),
        'WIDTH': 128,
        'BATCH_SIZE': 66, # do 64 make sure not larger than VALIDATION_SIZE *
        'NUM_CLASSES': 99,
        'ITERATIONS': 5e4,
        'SEED': 42,
        'TRAIN_SIZE': 1.0,
        'VALIDATION_SIZE': 0.2,
        'CLASS_SIZE': 1.0,
    }

    np.random.seed(fixed_params['SEED'])
    param = {
        'f_conv1_num': 8,
        'f_conv1_out': 512,
        'conv1_num': 7,
        'conv1_out': 64,
        'conv2_num': 7,
        'conv2_out': 32,
        #'conv2_out': 64,
        'conv3_num': 7,
        'conv3_out': 16,
        'd_out': 1024,
        'f_d_out': 1024,
        #'dropout': 0.57143340896097039,
        'dropout': 0.8255444234647442644,
        'f_dropout': 0.825544423647442644,
        'l2_penalty': 0.01,
        'CHANNEL': 1,
        'LEARNING_RATE': 0.001,
        'report_interval': 100 
    }

    data = etl.load_data(train_path=TRAIN_PATH,
                         test_path=TEST_PATH,
                         image_paths=IMAGE_PATHS,
                         image_shape=(fixed_params['HEIGHT'], fixed_params['WIDTH']))
    batches = etl.batch_generator(data.train, data.test,
                                  batch_size=fixed_params['BATCH_SIZE'],
                                  num_classes=fixed_params['NUM_CLASSES'],
                                  num_iterations=fixed_params['ITERATIONS'],
                                  seed=fixed_params['SEED'],
                                  train_size=fixed_params['TRAIN_SIZE'],
                                  val_size=fixed_params['VALIDATION_SIZE'],
                                  class_size=fixed_params['CLASS_SIZE'])


    param.update(fixed_params)
    print 'running with the following parameters: \n %s' % json.dumps(param, indent=4)
    model = cnn_classifier.CnnClassifier(data.train, data.test, data.le, batches, param)
    model.train(param['ITERATIONS'])
