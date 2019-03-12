
# Copyright 2017 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#!/usr/bin/env python

import tensorflow as tf

##TODO## : Ensure that you import the Horovod library

layers = tf.contrib.layers
learn = tf.contrib.learn

##DO NOT CHANGE##
tf.logging.set_verbosity(tf.logging.INFO)
##DO NOT CHANGE##

# As discussed in the lab, Horovod provides an easy option for scaling up your existing 
# single GPU model to multiple GPUs or systems. In this excercesize we will use a 
# slighly more complex model composed of a two layer convolutional neural netowrk. 
# As we will see there will be no need to change our existing model.
def conv_model(feature, target, mode):
    """2-layer convolution model."""
    # Convert the target to a one-hot tensor of shape (batch_size, 10) and
    # with a on-value of 1 for each one-hot vector of length 10.
    target = tf.one_hot(tf.cast(target, tf.int32), 10, 1, 0)

    # Reshape feature to 4d tensor with 2nd and 3rd dimensions being
    # image width and height final dimension being the number of color channels.
    feature = tf.reshape(feature, [-1, 28, 28, 1])

    # First conv layer will compute 32 features for each 5x5 patch
    with tf.variable_scope('conv_layer1'):
        h_conv1 = layers.conv2d(
            feature, 32, kernel_size=[5, 5], activation_fn=tf.nn.relu)
        h_pool1 = tf.nn.max_pool(
            h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Second conv layer will compute 64 features for each 5x5 patch.
    with tf.variable_scope('conv_layer2'):
        h_conv2 = layers.conv2d(
            h_pool1, 64, kernel_size=[5, 5], activation_fn=tf.nn.relu)
        h_pool2 = tf.nn.max_pool(
            h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # reshape tensor into a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    # Densely connected layer with 1024 neurons.
    h_fc1 = layers.dropout(
        layers.fully_connected(
            h_pool2_flat, 1024, activation_fn=tf.nn.relu),
        keep_prob=0.5,
        is_training=mode == tf.contrib.learn.ModeKeys.TRAIN)

    # Compute logits (1 per class) and compute loss.
    logits = layers.fully_connected(h_fc1, 10, activation_fn=None)
    loss = tf.losses.softmax_cross_entropy(target, logits)

    return tf.argmax(logits, 1), loss


def main(_):

    ##TODO## : Please make sure that you initialize Horovod.

    # Download and load MNIST dataset.
    mnist = learn.datasets.mnist.read_data_sets('MNIST-data-%d' % hvd.rank())

    # Build model...
    with tf.name_scope('input'):
        image = tf.placeholder(tf.float32, [None, 784], name='image')
        label = tf.placeholder(tf.float32, [None], name='label')
    predict, loss = conv_model(image, label, tf.contrib.learn.ModeKeys.TRAIN)

    ##TODO## : adjust learning rate based on number of GPUs.
    opt = tf.train.RMSPropOptimizer(0.001)

    ##TODO## : add Horovod Distributed Optimizer.

    global_step = tf.contrib.framework.get_or_create_global_step()
    train_op = opt.minimize(loss, global_step=global_step)

    hooks = [
        ##TODO## : Configure the BroadcastGlobalVariablesHook 


        ##TODO## : adjust number of steps based on number of GPUs.
        tf.train.StopAtStepHook(last_step=20000),

        ##DO NOT CHANGE##
        tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss': loss},
                                   every_n_iter=100),
        ##DO NOT CHANGE##
    ]

    ##TODO## : pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(hooks=hooks,
                                           config=config) as mon_sess:
        while not mon_sess.should_stop():
            # Run a training step synchronously.
            image_, label_ = mnist.train.next_batch(100)
            mon_sess.run(train_op, feed_dict={image: image_, label: label_})


if __name__ == "__main__":
    tf.app.run()
