import os
import glob
import json

import numpy as np

from scipy.stats import randint, uniform

from util import etl, helpers, cnn_classifier


def run(params_range, fixed_params, samplings=5):
    params = helpers.random_search(params_range, samplings)
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
    for param in params:
        param.update(fixed_params)
        print 'running with the following parameters: \n %s' % json.dumps(param, indent=4)
        model = cnn_classifier.CnnClassifier(data.train, data.test, data.le, batches, param)
	try:
            model.train(param['ITERATIONS'])
        except:
            print 'not enough memory'


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
        'ITERATIONS': 3e2,
        'SEED': 42,
        'TRAIN_SIZE': 1.0,
        'VALIDATION_SIZE': 0.2,
        'CLASS_SIZE': 1.0,
    }

    np.random.seed(fixed_params['SEED'])

    params_range = {
        'f_conv1_num': 8,
        'f_conv1_out': 512,
        'conv1_num': 5,
        'conv1_out': 64,
        'conv2_num': 7,
        'conv2_out': 32,
        'conv3_num': 7,
        'conv3_out': 16,
        'd_out': 1024,
        'f_d_out': 1024,
        'dropout': (0, uniform(0.5, 0.5)),
        'f_dropout': (0, uniform(0.5, 0.5)),
        'l2_penalty': (10, randint(-3, -1)),
        'CHANNEL': 1,
        'LEARNING_RATE': (10, randint(-3, -1)),
        'report_interval': 1e2
    }
    '''
    params_range = {
        'f_conv1_num': (0, randint(4, 10)),
        'f_conv1_out': (2, randint(4, 6)),
        'conv1_num': (0, randint(4, 10)),
        'conv1_out': (2, randint(6, 10)),
        'conv2_num': (0, randint(4, 10)),
        'conv2_out': (2, randint(4, 7)),
        'conv3_num': (0, randint(4, 10)),
        'conv3_out': (2, randint(3, 6)),
        'd_out': (2, randint(9, 11)),
        'f_d_out': (2, randint(9, 11)),
        'dropout': (0, uniform(0.5, 0.5)),
        'f_dropout': (0, uniform(0.5, 0.5)),
        'l2_penalty': (10, randint(-3, -1)),
        'CHANNEL': 1,
        'LEARNING_RATE': (10, randint(-3, -1)),
        'report_interval': 10
    }
    '''
    run(params_range, fixed_params, 20)


