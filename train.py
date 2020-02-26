import tensorflow as tf
import numpy as np
import os
import time
import datetime
from google.protobuf import text_format
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import gfile
from keras.models import model_from_json
import json
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from sklearn.utils import shuffle
# flags = tf.app.flags
# Model Hyperparameters

tf.app.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.app.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.app.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.app.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters

tf.app.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.app.flags.DEFINE_integer("num_epochs", 30, "Number of training epochs (default: 200)")
tf.app.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.app.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
#flags.DEFINE_string("inputFile", "final.csv", "Input file to build vocabulary from")

tf.app.flags.DEFINE_string("inputFile","train_data_new2.csv","Input file to build vocabulary from")
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.app.flags.FLAGS

words , count_words = data_helpers.read_data(FLAGS.inputFile)

#words , count_words = data_helpers.read_data(tf.app.flags.FLAGS.inputFile)
x_, y = data_helpers.get_data()
data = [len(x.split(" ")) for x in x_]

for i in range (0,len(data)):
    if(data[i] > 200):
        print(i)
max_document_length = 128
vocab_processor= learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_)))
vocab_dict = vocab_processor.vocabulary_._mapping
sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])
vocabulary = list(list(zip(*sorted_vocab))[0])
file = open("vocab_classifier1.txt","w")

file.writelines('{}:{}\n'.format(k,v) for k, v in vocab_dict.items())
file.close()

x_shuffled, y_shuffled = shuffle(x, y, random_state = 0)
dev_sample_index = -1 * int(.1 * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]


del x, y, x_shuffled, y_shuffled

session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)

with tf.Graph().as_default():
    with tf.Session() as sess:
        cnn = TextCNN(
                sequence_length=x_train[1].shape[0],
                num_classes=y_train[1].shape[0],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)
            # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)        
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
  

            # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab_classifier1"))
        saver = tf.train.Saver()
            # Initialize all variables
        init = tf.global_variables_initializer()
        sess.run(init)
        
        tf.train.write_graph(sess.graph.as_graph_def(), '.', 'tensorflowModel.pbtxt', as_text=True)

        def train_step(x_batch, y_batch):
            feed_dict = {
                cnn.inputTensor: x_batch,
                cnn.outputTensor: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
            _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            feed_dict = {
                cnn.inputTensor: x_batch,
                cnn.outputTensor: y_batch,
                cnn.dropout_keep_prob: 1.0
                }
            step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

            # Generate batches
        batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
        


        saver.save(sess,'./tensorflowModel.ckpt')
        tfModel = 'tensorflowModel'
        inputGraphPath = tfModel+'.pbtxt'
        checkpointPath = './'+tfModel+'.ckpt'
        inputSaverDefPath = ""
        inputBinary = False
        inputNode = "inputTensor"
        outputNode = "output/softmax"
        restore = "save/restore_all"
        filenameTensorName = "save/Const:0"
        outputFrozenGraph = 'frozen'+tfModel+'.pb'
        outputOptimizedGraph = 'classifier1.pb'
        clearDevices = True


        freeze_graph.freeze_graph(inputGraphPath, inputSaverDefPath ,
                                inputBinary, checkpointPath, outputNode,
                                restore, filenameTensorName,
                                outputFrozenGraph, clearDevices, "")
        inputGraph = tf.GraphDef()
        with tf.gfile.Open(outputFrozenGraph, "rb") as f:
            data2read = f.read()
            inputGraph.ParseFromString(data2read)

        outputGraph = optimize_for_inference_lib.optimize_for_inference(
                inputGraph,
                [inputNode], # an array of the input node(s)
                [outputNode], # an array of output nodes
                tf.int32.as_datatype_enum)

        # Save the optimized graph

        f = tf.gfile.FastGFile(outputOptimizedGraph, "w")
        f.write(outputGraph.SerializeToString()) 



