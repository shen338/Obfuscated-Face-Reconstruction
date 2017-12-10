

import tensorflow as tf
from AlexNet import AlexNet

from datetime import datetime
from imageReader import *
import numpy as np
import os



num_epoch = 10
learning_rate = 0.01
batch_size = 50
dropout_rate = 0.5
num_classes = 200
train_layers = ['fc8']
display_step = 50

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "/finetune_alexnet/tiny_imagenet"
checkpoint_path = "/finetune_alexnet/"

# create dir if they are not existing
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)

x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
dropout_prob = tf.placeholder(tf.float32)

model = AlexNet(x,dropout_prob,num_classes,train_layers)

score = model.fc8

var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
with tf.name_scope("cross_ent"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))

with tf.name_scope("train"):
  # Get gradients of all trainable variables
  gradients = tf.gradients(loss, var_list)
  gradients = list(zip(gradients, var_list))

  # Create optimizer and apply gradient descent to the trainable variables
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
  tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
  tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)

with tf.name_scope('accuracy'):
    correct_predict = tf.equal(tf.argmax(score,axis=1),tf.argmax(y,axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict,tf.float32))

tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# read data from imagereader
class TrainConfig(object):
  """Training configuration"""
  batch_size = 50
  num_epochs = 50
  summary_interval = 250
  eval_interval = 2000  # must be integer multiple of summary_interval
  lr = 0.01  # learning rate
  reg = 5e-4  # regularization
  momentum = 0.9
  dropout_keep_prob = 0.5
  #model_name = 'AlexNet'  # choose model
  #model = staticmethod(globals()[model_name])  # gets model by name

config = TrainConfig()

# generator the list that store the path and labels through the whole training process
filenames_labels_train = load_filenames_labels('train')
filenames_labels_validation = load_filenames_labels('val')

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(100000/batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(10000/batch_size).astype(np.int16)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    model.load_pretrained_alexnet(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    for epoch in range(num_epoch):
        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

        step = 1

        # shuffle images and labels every epoch
        random.shuffle(filenames_labels_train)
        random.shuffle(filenames_labels_validation)

        # construct a queue to keep the order of images and labels
        Q_train = Queue()
        Q_val = Queue()
        for items in filenames_labels_train:
            Q_train.put(items)
        for items in filenames_labels_validation:
            Q_val.put(items)
        while step < train_batches_per_epoch:

            # fetch a batch of data
            batch_data_train = single_batch(Q_train, 'train', config)
            #print(batch_data_train[0].shape)
            #print(batch_data_train[1].shape)
            # And run the training op
            sess.run(train_op, feed_dict={x: batch_data_train[0].eval(),
                                          y: batch_data_train[1].eval(),
                                          dropout_prob: dropout_rate})

            #Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: batch_data_train[0].eval(),
                                                        y: batch_data_train[1].eval(),
                                                        dropout_prob: 1.})
                writer.add_summary(s, epoch * train_batches_per_epoch + step)
            print(step)
            step += 1
        # Validate the model on the entire validation set
        print("{} Start validation".format(datetime.now()))

        test_acc = 0.
        test_count = 0
        for _ in range(val_batches_per_epoch):
            batch_data_val = single_batch(Q_val, 'val', config)
            acc = sess.run(accuracy, feed_dict={x: batch_data_val[0].eval(),
                                                y: batch_data_val[1].eval(), dropout_prob: 1.})
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        print("Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))


        print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch) + '.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))