import pickle
import glob

from util import cnn_classifier


TRAIN_PATH = "./data/train.csv"
TEST_PATH = "./data/test.csv"
IMAGE_PATHS = glob.glob("./data/images/*.jpg")
IMAGE_SHAPE = (128, 128, 1)
HEIGHT, WIDTH, CHANNEL = IMAGE_SHAPE

with open('./data/train.pickle', 'rb') as f:
    train = pickle.load(f)
with open('./data/test.pickle', 'rb') as f:
    test = pickle.load(f)
with open('./data/le.pickle', 'rb') as f:
    classes = pickle.load(f)

BATCH_SIZE = 64
NUM_CLASSES = 99
ITERATIONS = 1e3
SEED = 42
TRAIN_SIZE = 1.0
VALIDATION_SIZE = 0.1
CLASS_SIZE = 0.1


params = {
    'conv1_num': 5,
    'conv1_out': 32,
    'conv2_num': 5,
    'conv2_out': 64,
    'd_out': 1024,
    'dropout': 0.75,
    'HEIGHT': HEIGHT,
    'WIDTH': WIDTH,
    'CHANNEL': CHANNEL,
    'BATCH_SIZE': BATCH_SIZE,
    'NUM_CLASSES': NUM_CLASSES,
    'VALIDATION_SIZE': VALIDATION_SIZE,
    'SEED': SEED,
    'TRAIN_SIZE': TRAIN_SIZE,
    'CLASS_SIZE': 0.1,
    'ITERATIONS': ITERATIONS,
    'LEARNING_RATE': 0.0005,
    'report_interval': 1
}
model = cnn_classifier.CnnClassifier(train, test, classes, params)
model.train(20)



