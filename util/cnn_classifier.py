import os
from datetime import datetime
import time
import pickle
import glob
import functools
import json
import itertools

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

#from skimage.io import imread

from util import etl, helpers

def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


class CnnClassifier:
    def __init__(self, train, test, classes, batches, params):
        self.classes = classes
        self.params = params

        self.image = tf.placeholder(tf.float32, [
            self.params['BATCH_SIZE'],
            self.params['WIDTH'],
            self.params['HEIGHT'],
            self.params['CHANNEL']],
            name='x_image_pl')
        #self.size = tf.constant([self.params['HEIGHT'], self.params['WIDTH']])
        #self.image = tf.placeholder(tf.string, name='x_image_path_pl')
        self.label = tf.placeholder(tf.float32, [
            self.params['BATCH_SIZE'],
            int(round(self.params['NUM_CLASSES'] * self.params['CLASS_SIZE']))],
            name='classes_pl')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob_pl') #dropout (keep probability)
        self.batches = batches
        '''
        self.batches = etl.batch_generator(train, test,
                                           batch_size=self.params['BATCH_SIZE'],
                                           num_classes=self.params['NUM_CLASSES'],
                                           num_iterations=self.params['ITERATIONS'],
                                           seed=self.params['SEED'],
                                           train_size=self.params['TRAIN_SIZE'],
                                           val_size=self.params['VALIDATION_SIZE'],
                                           class_size=self.params['CLASS_SIZE'])
        '''

        self.train_batch = self.batches.gen_train()
        self.valid_batch = self.batches.gen_valid()
        self.valid_batch_one, _ = self.valid_batch.next()
        #self.valid_images = np.expand_dims(np.array([imread(im) for im in self.valid_batch_one['images']]), axis=4)

        self.net = {}
        self.prediction
        self.probability
        self.optimize
        self.error
        self.loss
        self.min_loss = 1e99
        self.saver = tf.train.Saver()
        self.last_ckpt, self.last_params, self.last_results = None, None, None

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    @define_scope(initializer=tf.global_variables_initializer())
    def prediction(self):
        # Construct model
        self.weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([self.params['conv1_num'], self.params['conv1_num'], 1, self.params['conv1_out']]), name='wc1'),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([self.params['conv2_num'], self.params['conv2_num'], self.params['conv1_out'], self.params['conv2_out']]), name='wc2'),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([self.params['WIDTH'] / 4 * self.params['HEIGHT'] / 4 * self.params['conv2_out'], self.params['d_out']]), name='wd1'),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([self.params['d_out'], int(round(self.params['NUM_CLASSES'] * self.params['CLASS_SIZE']))]), name='out')
        }
        self.biases = {
            'bc1': tf.Variable(tf.random_normal([self.params['conv1_out']]), name='bc1'),
            'bc2': tf.Variable(tf.random_normal([self.params['conv2_out']]), name='bc2'),
            'bd1': tf.Variable(tf.random_normal([self.params['d_out']]), name='bd1'),
            'out': tf.Variable(tf.random_normal([int(round(self.params['NUM_CLASSES'] * self.params['CLASS_SIZE']))]), name='b_out')
        }
        #x = helpers.conv_net(self.image, self.size, self.weights, self.biases, self.keep_prob, self.net)
        x = helpers.conv_net(self.image, self.weights, self.biases, self.keep_prob, self.net)
        #return self.prediction_to_probability(x)
        #return tf.nn.softmax_cross_entropy_with_logits(x, self.label)
        return x

    @define_scope
    def probability(self):
        return tf.nn.softmax(self.prediction)

    @define_scope
    def optimize(self):
        #self.loss = -tf.reduce_mean(tf.reduce_sum(self.label * tf.log(tf.clip_by_value(self.prediction, 1e-15, 1-1e-15)), reduction_indices=[1]))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.prediction, self.label))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.params['LEARNING_RATE']).minimize(self.loss)
        return self.optimizer

    @define_scope
    def accuracy(self):
        correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.label, 1))
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    @define_scope
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    def train(self, iterations):
        print "Iter \t Batch Loss \t Batch Accuracy \t Valid Loss \t Valid Accuracy \t Time delta\n"
        time_last = time.time()
        for i, batch in enumerate(self.train_batch):
            #images = np.expand_dims(np.array([imread(im) for im in batch['images']]), axis=4)
            print 'dropout: ', self.params['dropout']
            self.sess.run(self.optimizer, {
                self.image: batch['images'],
                #self.image: images,
                self.label: batch['ts'],
                self.keep_prob: self.params['dropout']
            })
            if i % self.params['report_interval'] == 0:
                # Calculate batch loss and accuracy
                batch_loss, batch_acc = self.sess.run([self.loss, self.accuracy], feed_dict={
                    self.image: batch['images'],
                    #self.image: images,
                    self.label: batch['ts'],
                    self.keep_prob: 1.
                })
                valid_loss, valid_acc = self.sess.run([self.loss, self.accuracy], feed_dict={
                    self.image: self.valid_batch_one['images'],
                    #self.image: self.valid_images,
                    self.label: self.valid_batch_one['ts'],
                    self.keep_prob: 1.
                })
                print "%d\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n" % (i, batch_loss, batch_acc, valid_loss, valid_acc, time.time() - time_last)
                time_last = time.time()
            if valid_acc > 0.90 and valid_loss < self.min_loss:
                self.min_loss = valid_loss
                self.save_results(valid_loss, i)
                self.save_params(valid_loss, i)

            if i >= iterations:
                break

    def run_against_test(self):
        preds_test = []
        ids_test = []
        for batch, num in self.batches.gen_test():
            res_test = self.sess.run([self.probability], feed_dict={
                self.image: batch['images'],
                self.keep_prob: 1
            })
            y_out = res_test[0]
            ids_test.append(batch['ids'])
            if num != len(y_out):
                y_out = y_out[:num]
            preds_test.append(y_out)
        ids_test = list(itertools.chain.from_iterable(ids_test))
        preds_test = np.concatenate(tuple(preds_test), axis=0)
        assert len(ids_test) == len(preds_test)
        return ids_test, preds_test


    def save_checkpoint(self, valid_loss, iteration):
        # haven't figured out how to reload in the context of
        # session variables being ownd also by this class
        model_folder = './tmp/models/'
        current_ckpt = 'model_%f_%i_%s' % (valid_loss, iteration, datetime.now().isoformat())
        os.makedirs('%s%s' % (model_folder, current_ckpt))
        self.saver.save(self.sess, '%s%s/%s.ckpt' % (model_folder, current_ckpt, current_ckpt))
        print 'saved checkpoint at %s%s' % (model_folder, current_ckpt)
        if self.last_ckpt is not None:
            for file in glob.glob('%s%s/*' % (model_folder, self.last_ckpt)):
                os.remove(file)
            os.rmdir('%s%s/' % (model_folder, self.last_ckpt))
            print 'deleted checkpoint at %s%s' % (model_folder, self.last_ckpt)
        self.last_ckpt = current_ckpt

    def save_params(self, valid_loss, iteration):
        current_params = './tmp/params/params_%f_%i_%s.json' % (valid_loss, iteration, datetime.now().isoformat())
        with open(current_params, 'w') as f:
            json.dump(self.params, f)
            print 'saved params at %s' % current_params
            if self.last_params is not None:
                os.remove(self.last_params)
                print 'deleted params at %s' % self.last_params
            self.last_params = current_params

    def save_results(self, valid_loss, iteration):
        ids_test, preds_test = self.run_against_test()

        preds_df = pd.DataFrame(preds_test, columns=self.classes.classes_)
        preds_df = preds_df.div(preds_df.sum(axis=1), axis=0)
        ids_test_df = pd.DataFrame(ids_test, columns=["id"])
        submission = pd.concat([ids_test_df, preds_df], axis=1)
        current_results = './tmp/results/results_%f_%i_%s.csv' % (valid_loss, iteration, datetime.now().isoformat())
        submission.to_csv(current_results, index=False)
        print 'saved results at %s' % current_results
        if self.last_results is not None:
            os.remove(self.last_results)
            print 'deleted results at %s' % self.last_results
        self.last_results = current_results

if __name__ == '__main__':
# loading data and setting up constants
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
    model = CnnClassifier(train, test, classes, params)
    model.train()

