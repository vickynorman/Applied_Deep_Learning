from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# Authors Ben Davies, Rory Hicks, Vicky Norman. University of Bristol. Jan 2019.

import os
import os.path
import tensorflow as tf
import numpy as np
import pickle
import librosa
from collections import Counter
# import torch.tensor as tensor

# 'Directory where the dataset will be stored and checkpoint.'
dataDir = os.getcwd() + '/dataset/'

# Number of ephocs
epochs = 100

# 'Number of steps between logging results to the console and saving summaries'
logFrequency = 1

# 'Number of steps between model saves.'
saveModel = 20

# Initialise hyperparameters
batchSize = 16
learningRate = 5.00e-5
imgWidth = 80
imgHeight = 80
imgChannels = 1
numClasses = 10
logDir = '{cwd}/logs/'.format(cwd=os.getcwd())


# Function that defines the architecture within the shallow neural network
def shallownn(x):

    xavier_initializer = tf.contrib.layers.xavier_initializer(uniform=True)

    x_image = tf.reshape(x, [-1, imgWidth, imgHeight, imgChannels])

    img_summary = tf.summary.image('Input_images', x_image)

    conv1 = tf.layers.conv2d(
        inputs=x_image,
        filters=16,
        kernel_size=[10, 23],
        padding='same',
        use_bias=True,
        activation=tf.nn.relu,
        kernel_initializer=xavier_initializer,
        bias_initializer=tf.zeros_initializer(),
        name='conv1'
    )
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[1,20],
        strides=[1,20],
        name='pool1'
    )
    pool1Flat = tf.contrib.layers.flatten(
        inputs=pool1,
        outputs_collections=None,
        scope=None
    )


    conv2 = tf.layers.conv2d(
        inputs=x_image,
        filters=16,
        kernel_size=[21, 20],
        padding='same',
        use_bias=True,
        activation=tf.nn.relu,
        kernel_initializer=xavier_initializer,
        bias_initializer=tf.zeros_initializer(),
        name='conv2'
    )
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[20, 1],
        strides=[20,1],
        name='pool2'
    )
    pool2Flat = tf.contrib.layers.flatten(
        inputs=pool2,
        outputs_collections=None,
        scope=None
    )

    merged = tf.concat([pool1Flat, pool2Flat], 1)

    dropout = tf.layers.dropout(merged,
        rate=0.1,
        noise_shape=None,
        seed=None,
        training=True,
        name='dropout'
        )

    fc11 = tf.layers.dense(
        inputs=dropout,
        units=200,
        activation=tf.nn.relu,
        use_bias=True,
        kernel_initializer=xavier_initializer,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        trainable=True,
        name=None,
        reuse=None
    )
    fc12 = tf.layers.dense(
        inputs=fc11,
        units=10,
        activation=tf.nn.relu,
        use_bias=True,
        kernel_initializer=xavier_initializer,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        trainable=True,
        name=None,
        reuse=None
    )

    return fc12, img_summary


# Function that defines the architecture within the deep neural network
def deepnn(x):

    # Initialiser for kernels
    xavier_initializer = tf.contrib.layers.xavier_initializer(uniform=True)

    x_image = tf.reshape(x, [-1, imgWidth, imgHeight, imgChannels])

    img_summary = tf.summary.image('Input_images', x_image)

    # Left hand side of the parallel architecture
    def firstlayer(img):

        conv11 = tf.layers.conv2d(
            inputs=img,
            filters=16,
            kernel_size=[10, 23],
            padding='same',
            use_bias=True,
            kernel_initializer=xavier_initializer,
            bias_initializer=tf.zeros_initializer(),
            name='conv11'
        )
        pool11 = tf.layers.max_pooling2d(
            inputs=conv11,
            pool_size=[2, 2],
            strides=2,
            name='pool11'
        )
        conv12 = tf.layers.conv2d(
            inputs=pool11,
            filters=32,
            kernel_size=[5, 11],
            padding='same',
            use_bias=True,
            kernel_initializer=xavier_initializer,
            bias_initializer=tf.zeros_initializer(),
            name='conv12'
        )
        pool12 = tf.layers.max_pooling2d(
            inputs=conv12,
            pool_size=[2, 2],
            strides=2,
            name='pool12'
        )
        conv13 = tf.layers.conv2d(
            inputs=pool12,
            filters=64,
            kernel_size=[3, 5],
            padding='same',
            use_bias=True,
            kernel_initializer=xavier_initializer,
            bias_initializer=tf.zeros_initializer(),
            name='conv13'
        )
        pool13 = tf.layers.max_pooling2d(
            inputs=conv13,
            pool_size=[2, 2],
            strides=2,
            name='pool13'
        )
        conv14 = tf.layers.conv2d(
            inputs=pool13,
            filters=128,
            kernel_size=[2, 4],
            padding='same',
            use_bias=True,
            kernel_initializer=xavier_initializer,
            bias_initializer=tf.zeros_initializer(),
            name='conv14'
        )
        pool14 = tf.layers.max_pooling2d(
            inputs=conv14,
            pool_size=[1, 5],
            strides=(1, 5),
            name='pool14'
        )

        pool1Flat = tf.contrib.layers.flatten(
            inputs=pool14,
            outputs_collections=None,
            scope=None
        )

        return pool1Flat

    # Right hand side of the parallel architecture
    def seclayer(img):

        conv21 = tf.layers.conv2d(
            inputs=img,
            filters=16,
            kernel_size=[21, 10],
            padding='same',
            use_bias=True,
            kernel_initializer=xavier_initializer,
            bias_initializer=tf.zeros_initializer(),
            name='conv21'
        )
        pool21 = tf.layers.max_pooling2d(
            inputs=conv21,
            pool_size=[2, 2],
            strides=2,
            name='pool21'
        )
        conv22 = tf.layers.conv2d(
            inputs=pool21,
            filters=32,
            kernel_size=[10, 5],
            padding='same',
            use_bias=True,
            kernel_initializer=xavier_initializer,
            bias_initializer=tf.zeros_initializer(),
            name='conv22'
        )
        pool22 = tf.layers.max_pooling2d(
            inputs=conv22,
            pool_size=[2, 2],
            strides=2,
            name='pool22'
        )
        conv23 = tf.layers.conv2d(
            inputs=pool22,
            filters=64,
            kernel_size=[5, 3],
            padding='same',
            use_bias=True,
            kernel_initializer=xavier_initializer,
            bias_initializer=tf.zeros_initializer(),
            name='conv23'
        )
        pool23 = tf.layers.max_pooling2d(
            inputs=conv23,
            pool_size=[2, 2],
            strides=2,
            name='pool23'
        )
        conv24 = tf.layers.conv2d(
            inputs=pool23,
            filters=128,
            kernel_size=[4, 2],
            padding='same',
            use_bias=True,
            kernel_initializer=xavier_initializer,
            bias_initializer=tf.zeros_initializer(),
            name='conv24'
        )
        pool24 = tf.layers.max_pooling2d(
            inputs=conv24,
            pool_size=[5, 1],
            strides=(5, 1),
            name='pool24'
        )

        pool2Flat = tf.contrib.layers.flatten(
            inputs=pool24,
            outputs_collections=None,
            scope=None
        )

        return pool2Flat

    a = firstlayer(x_image)
    b = seclayer(x_image)

    # Merge both sides of the parallel architecture
    merged = tf.concat([a, b], 1)

    dropout = tf.layers.dropout(merged,
                                rate=0.25,
                                noise_shape=None,
                                seed=None,
                                training=False,
                                name='dropout'
                                )

    fc1 = tf.layers.dense(
        inputs=dropout,
        units=200,
        activation=tf.nn.relu,
        use_bias=True,
        kernel_initializer=xavier_initializer,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        activity_regularizer=None,
        trainable=True,
        name=None,
        reuse=None
    )

    fc2 = tf.layers.dense(
        inputs=fc1,
        units=10,
        activation=tf.nn.relu,
        use_bias=True,
        kernel_initializer=xavier_initializer,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        trainable=True,
        name=None,
        reuse=None
    )

    return fc2, img_summary


# Function to calculate the raw accuracy between the labels and the predictions
def get_raw_accuracy(y_, y_conv):

    labels = tf.argmax(y_, 1)
    predictions = tf.argmax(y_conv, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))


    return accuracy


# Function to calculate the majority vote accuracy between the labels and the predictions
def get_maj_acc(labels, y_conv):

    preds = np.argmax(y_conv[0], axis=1)
    num = len(labels)
    maj_predictions = np.zeros(num)

    # Set prediction to be the majority vote of all segments for each track
    for i in range(int(num/15)):
        l = preds[i * 15: i * 15 + 15]
        data = Counter(l)
        vote = data.most_common(1)
        maj_predictions[i*15: i*15+15] = vote[0][0]

    maj = np.mean(np.equal(labels, maj_predictions))

    return maj


# Function to calculate the maximum accuracy between the labels and the predictions
def get_max_acc(labels, y_conv):

    y_conv = y_conv[0]
    num = len(labels)
    max_predictions = np.zeros(num)
    votes = np.zeros(10)
    for i in range(int(num/15)):
        for k in range(10):
            votes[k] = np.sum(y_conv[i*15: i*15+15,k])

        idx = np.argmax(votes, 0)
        max_predictions[i*15: i*15+15] = idx

    maxacc = np.mean(np.equal(labels, max_predictions))

    return maxacc


# Function to train and test a neural network of architecture 'netType' and whether the
# augment datset will be used shown by the bool 'augment'
def runNet(netType, augment):

    # Clear global graphs
    tf.reset_default_graph()

    # Initialise the log directory where the training logs will be stored
    runLogDir = os.path.join(logDir, '{nt}_Net_{ep}_epochs_Augmented_{ag}'.format(nt = netType, ep=epochs, ag=augment))

    # Define the dataset depending on if augment is True or False
    if augment == False:

    # Import Data
        with open('train_spectogram_dataset.pkl', 'rb') as f:
            train_set = pickle.load(f)
        dataSize = 11250

    elif augment == True:
        # Import Augmented Data, concatenate both pickle files
        with open('augment_train_spectogram_dataset1.pkl', 'rb') as f:
            train_1 = pickle.load(f)
        with open('augment_train_spectogram_dataset2.pkl', 'rb') as g:
            train_2 = pickle.load(g)

        train_1['data'] = np.concatenate((train_1['data'], train_2['data']), axis=0)
        train_1['track_id'] = np.concatenate((train_1['track_id'], train_2['track_id']), axis=0)
        train_1['labels'] = np.concatenate((train_1['labels'], train_2['labels']), axis=0)
        train_set = train_1

        dataSize = 47250

    # Import test data
    with open('test_spectogram_dataset.pkl', 'rb') as g:
        test_set = pickle.load(g)

    # Initialisers network placeholders
    with tf.variable_scope('inputs'):
        # Define input data as 80x80 spectrograms
        x = tf.placeholder(tf.float32, [None, imgWidth, imgHeight])
        # Define output data as 1x10 labels
        y_ = tf.placeholder(tf.float32, [None, numClasses])

    # Build graph depending on net specified
    if netType == 'Shallow':
        y_conv, img_summary = shallownn(x)
    elif netType == 'Deep':
        y_conv, img_summary = deepnn(x)

    # Define loss function as softmax cross entropy
    with tf.variable_scope('x_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    # Implement L1 regulisation on all trainable variables
    l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.0001, scope=None)
    weights = tf.trainable_variables()
    regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
    regularized_loss = cross_entropy + regularization_penalty

    # Define the optimiser as Adam Optimiser minimising the regularised loss
    optimiser = tf.train.AdamOptimizer(learningRate, beta1=0.9, beta2=0.999, epsilon=1e-08, ).minimize(
        regularized_loss)

    # calculate the prediction and the accuracy
    raw_accuracy = get_raw_accuracy(y_, y_conv)

    loss_summary = tf.summary.scalar('Loss', cross_entropy)
    acc_summary = tf.summary.scalar('Raw Accuracy', raw_accuracy)

    # summaries for TensorBoard visualisation
    validation_summary = tf.summary.merge([img_summary, acc_summary])
    training_summary = tf.summary.merge([img_summary, loss_summary])
    test_summary = tf.summary.merge([img_summary, acc_summary])

    # saver for checkpoints
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    with tf.Session() as sess:

        summary_writer = tf.summary.FileWriter(runLogDir + '_train', sess.graph, flush_secs=5)
        summary_writer_validation = tf.summary.FileWriter(runLogDir + '_validate', sess.graph, flush_secs=5)

        sess.run(tf.global_variables_initializer())
        # saver.load(sess, path, etc.)
        arr = np.arange(dataSize)

        for epoch in range(epochs):
            # Training and validation
            # Shuffle indexes of data images
            np.random.shuffle(arr)
            for step in range(0, dataSize, batchSize):
                # Training: Backpropagation using train set
                # Get samples in of batchSize from data
                trainImages = train_set['data'][arr[step:step + batchSize]]
                trainLabels = train_set['labels'][arr[step:step + batchSize]]

                # Train the optimiser
                _, summary_str = sess.run([optimiser, training_summary],
                                          feed_dict={x: trainImages, y_: trainLabels})

            # Add summary
            if epoch % (logFrequency + 1) == 0:
                summary_writer.add_summary(summary_str, epoch)

            # Validate on entire test set
            valImages = test_set['data']
            valLabels = test_set['labels']
            rawval_accuracy, summary_str = sess.run([raw_accuracy, validation_summary],
                                                    feed_dict={x: valImages, y_: valLabels})
            if epoch % 1 == 0:
                print('epoch %d, raw_accuracy on validation batch: %g' % (epoch, rawval_accuracy))
            summary_writer_validation.add_summary(summary_str, epoch)

            # Save the model checkpoint periodically.
            if epoch % saveModel == 0 or (epoch + 1) == epochs:
                checkpoint_path = os.path.join(runLogDir + '_train', 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=epoch)

        # Testing
        print('TESTING')

        # Test on entire test set
        testImages = test_set['data']
        testLabels = test_set['labels']
        raw_accuracy, _ = sess.run([raw_accuracy, test_summary], feed_dict={x: testImages, y_: testLabels})

        # calculate max and maj accuracies
        output = np.asarray(sess.run([y_conv], feed_dict={x: testImages}))
        max_acc = get_max_acc(np.argmax(testLabels, axis=1), output)

        maj_acc = get_maj_acc(np.argmax(testLabels, axis=1), output)

        accuracies = [raw_accuracy, max_acc, maj_acc]

    return accuracies


def main(_):

    shallow_accuracies = runNet('Shallow', False)
    shallow_aug_accuracies = runNet('Shallow', True)
    deep_accuracies = runNet('Deep', False)
    deep_aug_accuracies = runNet('Deep', True)

    print('Shallow raw accuracy on test set: %0.3f' % shallow_accuracies[0])
    print('Shallow max accuracy on test set: %0.3f' % shallow_accuracies[1])
    print('Shallow maj accuracy on test set: %0.3f' % shallow_accuracies[2])

    print('Shallow raw accuracy on augmented test set: %0.3f' % shallow_aug_accuracies[0])
    print('Shallow max accuracy on augmented test set: %0.3f' % shallow_aug_accuracies[1])
    print('Shallow maj accuracy on augmented test set: %0.3f' % shallow_aug_accuracies[2])

    print('Deep raw accuracy on test set: %0.3f' % deep_accuracies[0])
    print('Deep max accuracy on test set: %0.3f' % deep_accuracies[1])
    print('Deep maj accuracy on test set: %0.3f' % deep_accuracies[2])

    print('Deep raw accuracy on augmented test set: %0.3f' % deep_aug_accuracies[0])
    print('Deep max accuracy on augmented test set: %0.3f' % deep_aug_accuracies[1])
    print('Deep maj accuracy on augmented test set: %0.3f' % deep_aug_accuracies[2])


if __name__ == '__main__':
    tf.app.run(main=main)