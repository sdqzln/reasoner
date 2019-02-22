# -*-encoding:utf-8-*-
import  tensorflow as tf
import numpy as np
from model_inter_attention_chinese_v3 import ATAELSTM_inner_attention_v3
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from data_util_chinese_inneraction import batch_iter, read_datasets, split_data_train_test, trans_form_location
import datetime
import time

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .2, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("domain_name", "camera", "the domain of the data.")
tf.flags.DEFINE_string("pre_training_word_embedding_file", "./data/glove.6B.300d.txt", "pre_trained word embeddings")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_integer("hidden_dim", 300, "Dimensionality of  rnn output  (default: 300)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("max_grad_norm", 100, "clip gradients to this norm [50]")
tf.flags.DEFINE_float("lr_rate", 0.005, "learninig rate....")


# Training parameters
tf.flags.DEFINE_integer('n_hop', 3, 'number of hop')
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_float("per_process_gpu_memory_fraction", 0.9, "per_process_gpu_memory_fraction")
tf.flags.DEFINE_string("visible_devices", '1', "CUDA_VISIBLE_DEVICES")

FLAGS = tf.flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.visible_devices  # 使用 GPU 1


def train(x_train_sentence, x_train_target, y_train, x_train_location, x_test_sentence, x_test_target, y_test,
          x_test_location, vocab_size, sequence_length, target_length):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.per_process_gpu_memory_fraction)
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement,
          gpu_options=gpu_options)
        sess = tf.Session(config=session_conf)
        num_classes = 2
        embedding_dim = FLAGS.embedding_dim
        hidden_dim = FLAGS.hidden_dim
        n_hop = FLAGS.n_hop
        with sess.as_default():
            grnn = ATAELSTM_inner_attention_v3(sequence_length, num_classes, vocab_size+1, target_length, embedding_dim, hidden_dim, n_hop)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(FLAGS.lr_rate)
            grads_and_vars = optimizer.compute_gradients(grnn.loss)
            # clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], FLAGS.max_grad_norm), gv[1]) \
            #                           for gv in grads_and_vars]
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
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
            loss_summary = tf.summary.scalar("loss", grnn.loss)
            acc_summary = tf.summary.scalar("accuracy", grnn.accuracy)
            macro_summary = tf.summary.scalar("macro-f1", grnn.macro)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary, macro_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_context_batch, x_target_batch, y_batch, x_train_location_batch):
                """
                A single training step
                """
                feed_dict = {
                  grnn.input_x: x_context_batch,
                  grnn.input_x_target: x_target_batch,
                  grnn.input_y: y_batch,
                  grnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, grnn.loss, grnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_context_batch, x_target_batch, y_batch, x__location_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  grnn.input_x: x_context_batch,
                  grnn.input_x_target: x_target_batch,
                  grnn.input_y: y_batch,
                  grnn.dropout_keep_prob: 1.0,
                }
                step, summaries, loss, accuracy, macro = sess.run(
                    [global_step, dev_summary_op, grnn.loss, grnn.accuracy, grnn.macro],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}, macro {:g}".format(time_str, step, loss, accuracy, macro))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = batch_iter(
                list(zip(x_train_sentence, x_train_target, x_train_location, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_context_batch, x_target_batch,  x_train_location_batch, y_batch = zip(*batch)
                train_step(x_context_batch, x_target_batch, y_batch, x_train_location_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_test_sentence, x_test_target, y_test, x_test_location, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


def main(argv=None):
    base_dir = os.path.abspath(os.path.curdir)
    data_base_dir = os.path.join(base_dir, "chinese_review_datasets")
    sentences, targets, labels, locations = read_datasets(FLAGS.domain_name, data_base_dir)
    tokenzier = Tokenizer()
    tokenzier.fit_on_texts(sentences+targets)
    word_index = tokenzier.word_index
    vocab_size = len(word_index)
    print("vocab_size:", vocab_size)
    data = split_data_train_test(sentences, targets, labels, locations, FLAGS.dev_sample_percentage, shuffer=True)
    train_shuffled_sentence, train_shuffled_targets, train_shuffled_labels, train_shuffled_locations, \
    test_shuffled_sentence, test_shuffled_targets, test_shuffled_labels, test_shuffled_locations = data
    print("database size:", len(sentences))
    print("train/test:{}/{}".format(len(train_shuffled_sentence), len(test_shuffled_sentence)))
    train_context = tokenzier.texts_to_sequences(train_shuffled_sentence)
    train_target = tokenzier.texts_to_sequences(train_shuffled_targets)
    test_coentxt = tokenzier.texts_to_sequences(test_shuffled_sentence)
    test_target = tokenzier.texts_to_sequences(test_shuffled_targets)
    train_sen_maxlen = max([len(x) for x in train_context])
    train_target_maxlen = max([len(x) for x in train_target])
    test_sen_maxlen = max([len(x) for x in test_coentxt])
    test_target_maxlen = max([len(x) for x in test_target])
    sen_maxlen= max(train_sen_maxlen, test_sen_maxlen)
    target_maxlen = max(train_target_maxlen, test_target_maxlen)
    print("sentence max len:", sen_maxlen)
    print("target max len:", target_maxlen)
    train_context = sequence.pad_sequences(train_context, maxlen=sen_maxlen, padding="post")
    train_target = sequence.pad_sequences(train_target, maxlen=target_maxlen, padding="post")
    train_location = trans_form_location(train_shuffled_locations, sen_maxlen)
    test_coentxt = sequence.pad_sequences(test_coentxt, maxlen=sen_maxlen, padding="post")
    test_target = sequence.pad_sequences(test_target, maxlen=target_maxlen, padding="post")
    test_location = trans_form_location(test_shuffled_locations, sen_maxlen)


    train(train_context, train_target, train_shuffled_labels, train_location,
          test_coentxt, test_target, test_shuffled_labels, test_location,
          vocab_size, sen_maxlen, target_maxlen)

if __name__ == '__main__':
    tf.app.run()