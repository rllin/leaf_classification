import numpy as np
from skimage.transform import resize
from skimage.util import pad
import tensorflow as tf


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    """Conv2D wrapper, with bias and relu activation."""
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def conv1d(x, W, b, stride=1):
    """Conv1d wrapper, with bias and relu activation."""
    x = tf.nn.conv1d(x, W, stride=stride, padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    """MaxPool2D wrapper."""
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def combine_f_i_nets(i_conv_net, f_conv_net, weights, biases, net):
    """Combines feature and image networks with scaling, trainable matrix."""
    net['combined'] = tf.add(
        tf.add(tf.matmul(i_conv_net, weights['i_conv_out']),
           tf.matmul(f_conv_net, weights['f_conv_out'])),
        biases['f_i_conv_out'])
    return net['combined']


def f_conv_net(x, weights, biases, f_dropout, net):
    """Takes features and constructs conv1d with pooling and fully connected
    neural net with dropout.
    """
    x = tf.expand_dims(x, 2)
    net['f_conv1'] = conv1d(x, weights['f_wc1'], biases['f_bc1'])
    net['f_pool1'] = tf.squeeze(maxpool2d(tf.expand_dims(net['f_conv1'], 1), k=2))

    net['f_fc1'] = tf.reshape(net['f_pool1'], [-1, weights['f_wd1'].get_shape().as_list()[0]])
    net['f_fc1'] = tf.add(tf.matmul(net['f_fc1'], weights['f_wd1']), biases['f_bd1'])
    net['f_fc1'] = tf.nn.relu(net['f_fc1'])

    net['f_out'] = tf.nn.dropout(net['f_fc1'], f_dropout)
    net['f_out'] = tf.add(tf.matmul(net['f_fc1'], weights['f_out']), biases['f_out'])

    return net['f_out']


def conv_net(x, weights, biases, dropout, net):
    """Takes image and constructs a 2 layered conv2d with pooling and
    fully connected layer with dropout."""
    # Convolution Layer
    net['conv1'] = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    net['pool1'] = maxpool2d(net['conv1'], k=2)

    # Convolution Layer
    net['conv2'] = conv2d(net['pool1'], weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    net['pool2'] = maxpool2d(net['conv2'], k=2)
    '''
    # Convolution Layer
    net['conv3'] = conv2d(net['pool2'], weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    net['pool3'] = maxpool2d(net['conv3'], k=2)
    '''

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    net['fc1'] = tf.reshape(net['pool2'], [-1, weights['wd1'].get_shape().as_list()[0]])
    net['fc1'] = tf.add(tf.matmul(net['fc1'], weights['wd1']), biases['bd1'])
    net['fc1'] = tf.nn.relu(net['fc1'])
    # Apply Dropout
    net['fc1'] = tf.nn.dropout(net['fc1'], dropout)

    # Output, class prediction
    net['out'] = tf.add(tf.matmul(net['fc1'], weights['out']), biases['out'])

    return net['out']


def pad_upto(image, (target_height, target_width)):
    """Pads image up to a target height and width."""
    h, w = image.shape
    up, left = (target_height - h) / 2, (target_width - w) / 2
    down, right = target_height - h - up, target_width - w - left
    return pad(image, ((up, down), (left, right)), 'constant')


def resize_proportionally(image, (target_height, target_width)):
    """Resizes an image proportionally by using pad_upto for remaining
    width or height."""
    h, w = image.shape
    h_mul, w_mul = target_height / float(h), target_width / float(w)
    if h_mul > w_mul:
        scaled = resize(image, (int(w_mul * h), target_width), mode='constant')
    else:
        scaled = resize(image, (target_height, int(h_mul * w)), mode='constant')
    return pad_upto(scaled, (target_height, target_width))


def scale_resize(image, (max_height, max_width), (target_height, target_width)):
    """Resizes image to target dimensions while keeping proportionality."""
    return resize(resize_proportionally(image, (max_height, max_width)), (target_height, target_width))


def random_search(params_range, samplings):
    """Takes a json / dictionary configuration of uniform, randint, or fixed
    value.
    Args:
        params_range: json / dictionary specification.
        samplings: number of samples to be returned.

    Returns:
        :obj:`list` of :obj:`dict`: dictionary of parameters to use.
    """
    params = [None,] * samplings
    for instance in range(samplings):
        param = {}
        for param_name, value in params_range.items():
            if isinstance(value, tuple):
                base, dist = value
                if base == 0:
                    param[param_name] = dist.rvs()
                else:
                    param[param_name] = base ** dist.rvs()
            elif isinstance(value, np.ndarray):
                param[param_name] = np.random.choice(value)
            else:
                param[param_name] = value
        params[instance] = param
    return params
