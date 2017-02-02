import os
from datetime import datetime
import time
import glob
import functools
import json
import itertools

import numpy as np
import pandas as pd
import tensorflow as tf

from util import helpers

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
    """Trains and validates streams from batch object according to certain
    parameters.
    """
    def __init__(self, train, test, classes, batches, params, seed=42):
        self.classes = classes
        self.params = params
	tf.set_random_seed(seed)
        self.image = tf.placeholder(tf.float32, [
            self.params['BATCH_SIZE'],
            self.params['WIDTH'],
            self.params['HEIGHT'],
            self.params['CHANNEL']],
            name='x_image_pl')
        self.feature = tf.placeholder(tf.float32, [
            self.params['BATCH_SIZE'],
            64 * 3],
            name='x_features_pl')
        self.label = tf.placeholder(tf.float32, [
            self.params['BATCH_SIZE'],
            int(round(self.params['NUM_CLASSES'] * self.params['CLASS_SIZE']))],
            name='classes_pl')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob_pl') #dropout (keep probability)
        self.f_keep_prob = tf.placeholder(tf.float32, name='f_keep_prob_pl')

        # Construct model
        self.weights = {
            'f_wc1': tf.Variable(tf.random_normal([self.params['f_conv1_num'], 1, self.params['f_conv1_out']]), name='f_wc1'),
            'wc1': tf.Variable(tf.random_normal([self.params['conv1_num'], self.params['conv1_num'], 1, self.params['conv1_out']]), name='wc1'),
            'wc2': tf.Variable(tf.random_normal([self.params['conv2_num'], self.params['conv2_num'], self.params['conv1_out'], self.params['conv2_out']]), name='wc2'),
            #'wc3': tf.Variable(tf.random_normal([self.params['conv3_num'], self.params['conv3_num'], self.params['conv2_out'], self.params['conv3_out']]), name='wc333'),
            'wd1': tf.Variable(tf.random_normal([self.params['WIDTH'] / 4 * self.params['HEIGHT'] / 4 * self.params['conv2_out'], self.params['d_out']]), name='wd1'),
            #'wd1': tf.Variable(tf.random_normal([self.params['WIDTH'] / 8 * self.params['HEIGHT'] / 8 * self.params['conv3_out'], self.params['d_out']]), name='wd1'),
            'out': tf.Variable(tf.random_normal([self.params['d_out'], int(round(self.params['NUM_CLASSES'] * self.params['CLASS_SIZE']))]), name='out'),
            'f_out': tf.Variable(tf.random_normal([self.params['f_d_out'], int(round(self.params['NUM_CLASSES'] * self.params['CLASS_SIZE']))]), name='f_out'),
            'f_wd1': tf.Variable(tf.random_normal([64 * 3 / 2 * self.params['f_conv1_out'], self.params['f_d_out']]), name='f_out'),
            'i_conv_out': tf.Variable(tf.random_normal([int(round(self.params['NUM_CLASSES'] * self.params['CLASS_SIZE'])), int(round(self.params['NUM_CLASSES'] * self.params['CLASS_SIZE']))]), name='i_conv_out'),
            'f_conv_out': tf.Variable(tf.random_normal([int(round(self.params['NUM_CLASSES'] * self.params['CLASS_SIZE'])), int(round(self.params['NUM_CLASSES'] * self.params['CLASS_SIZE']))]), name='f_conv_out'),
        }
        self.biases = {
            'f_bc1': tf.Variable(tf.random_normal([self.params['f_conv1_out']]), name='f_bc1'),
            'bc1': tf.Variable(tf.random_normal([self.params['conv1_out']]), name='bc1'),
            'bc2': tf.Variable(tf.random_normal([self.params['conv2_out']]), name='bc2'),
            #'bc3': tf.Variable(tf.random_normal([self.params['conv3_out']]), name='bc3'),
            'bd1': tf.Variable(tf.random_normal([self.params['d_out']]), name='bd1'),
            'out': tf.Variable(tf.random_normal([int(round(self.params['NUM_CLASSES'] * self.params['CLASS_SIZE']))]), name='b_out'),
            'f_out': tf.Variable(tf.random_normal([int(round(self.params['NUM_CLASSES'] * self.params['CLASS_SIZE']))]), name='f_b_out'),
            'f_bd1': tf.Variable(tf.random_normal([self.params['f_d_out']]), name='f_b_out'),
            'f_i_conv_out': tf.Variable(tf.random_normal([int(round(self.params['NUM_CLASSES'] * self.params['CLASS_SIZE']))]))
        }

        self.batches = batches

        self.net = {}
        self.min_loss = 1e99
        self.last_ckpt, self.last_params, self.last_results = None, None, None

        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        tf.contrib.layers.summarize_variables()

    def restore(self, ckpt_file):
        self.saver.restore(self.sess, ckpt_file)

    def run_restore(self, ckpt_file):
        self.restore(ckpt_file)
        prediction, probability, loss, optimizer, accuracy, error, summaries = self.setup()
        self.run_against_test(probability)

    def setup(self):
        """Sets up convolution nets and calculates loss and other metrics."""
        features_prediction = helpers.f_conv_net(self.feature, self.weights, self.biases, self.f_keep_prob, self.net)
        if self.params['features_images'] == 'images and features':
            image_prediction = helpers.conv_net(self.image, self.weights, self.biases, self.keep_prob, self.net)
	    prediction = helpers.combine_f_i_nets(image_prediction, features_prediction, self.weights, self.biases, self.net)
        else:
            prediction = features_prediction
        probability = tf.nn.softmax(prediction)
        #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, self.label))
	l2_loss = self.params['l2_penalty'] * (tf.nn.l2_loss(self.weights['f_wc1'])
                                               + tf.nn.l2_loss(self.weights['wc1'])
                                               + tf.nn.l2_loss(self.weights['wc2'])
                                               + tf.nn.l2_loss(self.weights['wd1']))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, self.label) + l2_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.params['LEARNING_RATE']).minimize(loss)
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        mistakes = tf.not_equal(
            tf.argmax(self.label, 1), tf.argmax(prediction, 1))
        error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
        summaries = tf.summary.merge_all()
        self.sess.run(tf.global_variables_initializer())
        return prediction, probability, loss, optimizer, accuracy, error, summaries


    def train(self, iterations):
        """Takes batches and trains while printing to std.out and logging."""
        saved_before = False
        prediction, probability, loss, optimizer, accuracy, error, summaries = self.setup()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        #summaries_path = "tensorboard/%s/logs" % (timestamp)
        #summarywriter = tf.train.SummaryWriter(summaries_path, self.sess.graph)
	f = open('./tmp/logs/%s.txt' % timestamp, 'w')
        if self.params['features_images'] == 'images and features':
            print 'images and features'
            f.write('images and features\n')
        else:
            print 'only features'
            f.write('only features\n')
        f.write(json.dumps(self.params, indent=4) + '\n')
        print "Iter \t Batch Loss \t Batch Accuracy \t Valid Loss \t Valid Accuracy \t Time delta\n"
        f.write("Iter \t Batch Loss \t Batch Accuracy \t Valid Loss \t Valid Accuracy \t Time delta\n")
        time_last = time.time()
        train_loss, train_acc = [], []
        for i, batch in enumerate(self.batches.gen_train()):
            #images = np.expand_dims(np.array([imread(im) for im in batch['images']]), axis=4)
            res_train = self.sess.run([optimizer, loss, accuracy, summaries], {
                self.image: batch['images'],
                #self.image: images,
                self.feature: batch['features'],
                self.label: batch['ts'],
                self.keep_prob: self.params['dropout'],
                self.f_keep_prob: self.params['f_dropout']
            })
            train_loss.append(res_train[1])
            train_acc.append(res_train[2])
            #summarywriter.add_summary(res_train[3], i)

            if i % self.params['report_interval'] == 0:
                valid_loss, valid_acc, summary = self.run_against_valid(loss, accuracy, summaries)
                train_loss = sum(train_loss) / float(len(train_loss))
                train_acc = sum(train_acc) / float(len(train_acc)) * 100
                # Calculate batch loss and accuracy
                print "%d:\t  %.2f\t\t  %.1f\t\t  %.2f\t\t  %.2f \t\t %.2f" % (i, train_loss, train_acc, valid_loss, valid_acc, time.time() - time_last)
                f.write("%d:\t  %.2f\t\t  %.1f\t\t  %.2f\t\t  %.2f \t\t %.2f\n" % (i, train_loss, train_acc, valid_loss, valid_acc, time.time() - time_last))
                #summarywriter.add_summary(summary, i)
                time_last = time.time()
                train_loss = []
                train_acc = []

                if (valid_acc > 95.0 and valid_loss < self.min_loss) or (saved_before == False and i == iterations):
                    saved_before = True
                    self.min_loss = valid_loss
                    self.save_results(valid_loss, valid_acc, i, probability)
                    self.save_params(valid_loss, valid_acc, i)
                    self.save_checkpoint(valid_loss, valid_acc, i)

            if i >= iterations:
                break

    def run_against_valid(self, loss, accuracy, summaries):
        """Runs current network against validation batch."""
        cur_acc, cur_loss, tot_num = 0, 0, 0
        for batch_valid, num in self.batches.gen_valid():
            valid_loss, valid_acc, summary = self.sess.run(
                [loss, accuracy, summaries], feed_dict={
                    self.image: batch_valid['images'],
                    #self.image: self.valid_images,
                    self.feature: batch_valid['features'],
                    self.label: batch_valid['ts'],
                    self.keep_prob: 1.,
                    self.f_keep_prob: 1
                })
            cur_loss += valid_loss * num
            cur_acc += valid_acc * num
            tot_num += num
        valid_loss = cur_loss / float(tot_num)
        valid_acc = (cur_acc / float(tot_num)) * 100
        return valid_loss, valid_acc, summary

    def run_against_test(self, probability):
        """Runs trained network against test batch."""
        preds_test = []
        ids_test = []
        for batch, num in self.batches.gen_test():
            res_test = self.sess.run([probability], feed_dict={
                self.image: batch['images'],
                self.feature: batch['features'],
                self.keep_prob: 1,
                self.f_keep_prob: 1
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


    def save_checkpoint(self, valid_loss, valid_acc, iteration):
        """Saves checkpoint and overwrites previous save to save space."""
        model_folder = './tmp/models/'
        current_ckpt = 'model_%f_%f_%i_%s' % (valid_loss, valid_acc, iteration, datetime.now().isoformat())
        os.makedirs('%s%s' % (model_folder, current_ckpt))
        self.saver.save(self.sess, '%s%s/%s.ckpt' % (model_folder, current_ckpt, current_ckpt))
        print 'saved checkpoint at %s%s' % (model_folder, current_ckpt)
        if self.last_ckpt is not None:
            for file_name in glob.glob('%s%s/*' % (model_folder, self.last_ckpt)):
                os.remove(file_name)
            os.rmdir('%s%s/' % (model_folder, self.last_ckpt))
            print 'deleted checkpoint at %s%s' % (model_folder, self.last_ckpt)
        self.last_ckpt = current_ckpt

    def save_params(self, valid_loss, valid_acc, iteration):
        """Saves parameters that were used for this training session."""
        current_params = './tmp/params/params_%f_%f_%i_%s.json' % (valid_loss, valid_acc, iteration, datetime.now().isoformat())
        with open(current_params, 'w') as f:
            json.dump(self.params, f)
            print 'saved params at %s' % current_params
            if self.last_params is not None:
                os.remove(self.last_params)
                print 'deleted params at %s' % self.last_params
            self.last_params = current_params

    def save_results(self, valid_loss, valid_acc, iteration, probability):
        """Converts to and saves submission csv for Kaggle."""
        ids_test, preds_test = self.run_against_test(probability)
        preds_df = pd.DataFrame(preds_test, columns=self.classes.classes_)
        preds_df = preds_df.div(preds_df.sum(axis=1), axis=0)
        ids_test_df = pd.DataFrame(ids_test, columns=["id"])
        submission = pd.concat([ids_test_df, preds_df], axis=1)
        current_results = './tmp/results/results_%f_%f_%i_%s.csv' % (valid_loss, valid_acc, iteration, datetime.now().isoformat())
        submission.to_csv(current_results, index=False)
        print 'saved results at %s' % current_results
        if self.last_results is not None:
            os.remove(self.last_results)
            print 'deleted results at %s' % self.last_results
        self.last_results = current_results
