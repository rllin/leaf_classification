import json

import numpy as np
from scipy.stats import randint, uniform
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.util import pad
import tensorflow as tf


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def conv1d(x, W, b, stride=1):
    x = tf.nn.conv1d(x, W, stride=stride, padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def f_conv_net(x, weights, biases, f_dropout, net):
    net['f_conv1'] = conv1d(x, weights['f_wc1'], biases['f_bc1'])
    print 'f_conv1: ', net['f_conv1'].get_shape()
    net['f_pool1'] = tf.squeeze(maxpool2d(tf.expand_dims(net['f_conv1'], 1), k=2))
    print 'f_pool1: ', net['f_pool1'].get_shape()
    net['f_fc1'] = tf.reshape(net['f_pool1'], [-1, weights['f_wd1'].get_shape().as_list()[0]])
    print 'reshape pool1 to: ', net['f_fc1'].get_shape()
    net['f_fc1'] = tf.add(tf.matmul(net['f_fc1'], weights['f_wd1']), biases['f_bd1'])
    net['f_fc1'] = tf.nn.relu(net['f_fc1'])

    net['f_out'] = tf.nn.dropout(net['f_fc1'], f_dropout)
    print 'f_drop: ', net['f_out'].get_shape()
    net['f_out'] = tf.add(tf.matmul(net['f_fc1'], weights['f_out']), biases['f_out'])
    print net['f_out'].get_shape()
    print 'done'

    return net['f_out']


def conv_net(x, weights, biases, dropout, net):
    # Reshape input picture
    # Convolution Layer
    #
    #x = tf.to_float(tf.image.decode_jpeg(tf.read_file(x), channels=1))
    #x = tf.image.resize_images(x, size)
    print x.get_shape()
    net['conv1'] = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    net['pool1'] = maxpool2d(net['conv1'], k=2)

    # Convolution Layer
    net['conv2'] = conv2d(net['pool1'], weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    net['pool2'] = maxpool2d(net['conv2'], k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    net['fc1'] = tf.reshape(net['pool2'], [-1, weights['wd1'].get_shape().as_list()[0]])
    print 'reshape pool2 to: ', net['fc1'].get_shape()
    net['fc1'] = tf.add(tf.matmul(net['fc1'], weights['wd1']), biases['bd1'])
    net['fc1'] = tf.nn.relu(net['fc1'])
    # Apply Dropout
    net['fc1'] = tf.nn.dropout(net['fc1'], dropout)

    # Output, class prediction
    net['out'] = tf.add(tf.matmul(net['fc1'], weights['out']), biases['out'])

    print net['conv1'].get_shape()
    print net['pool1'].get_shape()
    print net['conv2'].get_shape()
    print net['pool2'].get_shape()
    print 'drop: ', net['fc1'].get_shape()
    print net['out'].get_shape()
    print 'done'

    return net['out']


def pad_upto(image, (target_height, target_width)):
    h, w = image.shape
    up, left = (target_height - h) / 2, (target_width - w) / 2
    down, right = target_height - h - up, target_width - w - left
    return pad(image, ((up, down), (left, right)), 'constant')

def resize_proportionally(image, (target_height, target_width)):
    h, w = image.shape
    h_mul, w_mul = target_height / float(h), target_width / float(w)
    if h_mul > w_mul:
        scaled = resize(image, (int(w_mul * h), target_width), mode='constant')
    else:
        scaled = resize(image, (target_height, int(h_mul * w)), mode='constant')
    return pad_upto(scaled, (target_height, target_width))

def scale_resize(image, (max_height, max_width), (target_height, target_width)):
    return resize(resize_proportionally(image, (max_height, max_width)), (target_height, target_width))



def random_search(params_range, samplings):
    params = [None,] * samplings
    for instance in range(samplings):
        param = {}
        for param_name, value in params_range.items():
            if type(value) == tuple:
                base, dist = value
                if base == 0:
                    param[param_name] = dist.rvs()
                else:
                    param[param_name] = base ** dist.rvs()
            elif type(value) == np.ndarray:
                param[param_name] = np.random.choice(value)
            else:
                param[param_name] = value
        params[instance] = param
    return params


if __name__ == '__main__':
    VALIDATION_SIZE = 0.1
    SEED = 42
    TRAIN_SIZE = 1.0
    ITERATIONS = 1e1
    params_range = {
	'conv1_num': (0, randint(1, 10)),
	'conv1_out': (2, randint(2, 8)),
	'conv2_num': (0, randint(1, 10)),
	'conv2_out': (2, randint(2, 8)),
	'd_out': (2, randint(4, 10)),
	'dropout': (0, uniform(0, 1.0)),
	'HEIGHT': 128,
	'WIDTH': 128,
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
    a = random_search(params_range, 5)
    print a[0]
