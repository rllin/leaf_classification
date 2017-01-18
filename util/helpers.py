import tensorflow as tf

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def conv_net(x, weights, biases, dropout, net):
    # Reshape input picture
    # Convolution Layer
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
    print net['fc1'].get_shape()
    print net['out'].get_shape()


    return net['out']


