# coding: UTF-8
'''''''''''''''''''''''''''''''''''''''''''''''''''''
   create time: 2018.03.14 Wed. 23h45m16s
   author: Chuanfeng Liu
   e-mail: microlj@126.com
   github: https://github.com/Knownx
'''''''''''''''''''''''''''''''''''''''''''''''''''''
import os
import config
import tensorflow as tf
import alexnet
import dataset
from datetime import datetime
import numpy as np

def fineTune():
    trainData = config.TRAIN_LIST
    #valFile = config.TEST_LIST
    filewriterPath = config.TENSORBOARD_PATH # To be used for tensorboard
    checkpointPath = config.SAVE_MODEL_PATH
    pretrainedParams = config.FINETUNE_LIST
    trainPercent = 0.7
    shuffle = True

    # Learning parameters
    learningRate = 0.001
    numEpochs = 2
    batch_size = 128

    # Network parameters
    dropoutRate = 0.5
    numClasses = 2
    trainLayers = ['fc8', 'fc7', 'fc6']

    datasets = dataset.Dataset(trainData, numClasses, shuffle, trainPercent)

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = int(np.ceil(datasets.train_length / batch_size))  # int(np.floor(len(trainList)/batch_size))
    val_batches_per_epoch = int(np.ceil(datasets.validation_length / batch_size))  # int(np.floor(len(valList)/batch_size))

    # Graph input and output
    x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
    y = tf.placeholder(tf.float32, [batch_size, numClasses])
    keep_prob = tf.placeholder(tf.float32)

    # Initialize Model
    model = alexnet.AlexNet(x, keep_prob, numClasses, trainLayers)

    # Link variable to model input
    score = model.fc8

    # Loss and Optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(loss)

    # Evaluation
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the FileWriter
    writer = tf.summary.FileWriter(filewriterPath)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    # Init
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        print('Initialize variables')
        sess.run(init)

        writer.add_graph(sess.graph)

        print('Load the pretrained model: {}'.format(pretrainedParams))
        model.pretrainedModel(sess)

        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(), filewriterPath))

        print('Start training procedure from: {}'.format(datetime.now()))

        # Loop over the number of epochs
        for epoch in range(numEpochs):
            print('{} Epoch number: {}'.format(datetime.now(), epoch+1))
            datasets.reset_ptr()
            for step in range(train_batches_per_epoch):
                # Get next batch of training set
                batch_x, batch_y = datasets.getNextBatch(batch_size, 'training')

                # Run the training optimizer
                curr_loss, _ = sess.run([loss, optimizer], feed_dict={x:batch_x, y:batch_y, keep_prob:dropoutRate})

                print('Epoch: {}, Batch: {}, Training loss: {}'.format(epoch+1, step, curr_loss))

            #checkpoint_name_train = os.path.join(checkpointPath, 'model_epoch' + str(epoch + 1) + '_train.ckpt')
            #saver.save(sess, checkpoint_name_train)

            # Load trained checkpoints
            # saver.restore(sess, tf.train.latest_checkpoint(checkpoint_name_train))

            # Validate the model on the entire validation set
            print ('{} Start validation'.format(datetime.now()))
            test_acc = 0
            test_count = 0
            for _ in range(val_batches_per_epoch):
                batch_tx, batch_ty = datasets.getNextBatch(batch_size, 'validation')
                acc = sess.run(accuracy, feed_dict={x:batch_tx, y:batch_ty, keep_prob:1})
                test_acc += acc
                test_count += 1
                accc = test_acc/test_count
                print ('Current Accuracy: {}'.format(accc))
            test_acc /= test_count

            print('{} Validation Accuracy = {:.4f}'.format(datetime.now(), test_acc))

            print('{} Saving checkpoint of model...'.format(datetime.now()))
            checkpoint_name = os.path.join(checkpointPath, 'model_epoch'+str(epoch+1)+'.ckpt')
            saver.save(sess, checkpoint_name)
            print('{} Model checkpoint saved at {}'.format(datetime.now(), checkpoint_name))

        print ('{} All finished!'.format(datetime.now()))
if __name__ == '__main__':
    fineTune()
