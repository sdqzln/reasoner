import tensorflow as tf
import numpy as np
from data_utils_ram_15 import get_data_info, read_data, load_word_embeddings
from model_inter_attention_v3 import ATAELSTM_RelationNetwork_v3
import os


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', 32, 'number of example per batch')
tf.app.flags.DEFINE_integer('n_epoch', 15, 'number of epoch')
tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
tf.app.flags.DEFINE_integer('n_hop', 3, 'number of hop')
tf.app.flags.DEFINE_integer('pre_processed', 0, 'Whether the data is pre-processed')
tf.app.flags.DEFINE_float('learning_rate', 0.005, 'learning rate')
tf.app.flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')
tf.app.flags.DEFINE_float('dropout', 0.5, 'dropout')

tf.app.flags.DEFINE_string('embedding_fname', 'data/glove.840B.300d.txt', 'embedding file name')
tf.app.flags.DEFINE_string('train_fname', 'data/ABSA-15_Restaurants_Train_Final.xml', 'training file name')
tf.app.flags.DEFINE_string('test_fname', 'data/ABSA15_Restaurants_Test.xml', 'testing file name')
tf.app.flags.DEFINE_string('data_info', 'data/data_info_ram_2015.txt', 'the file saving data information')
tf.app.flags.DEFINE_string('train_data', 'data/train_data_ram_2015.txt', 'the file saving training data')
tf.app.flags.DEFINE_string('test_data', 'data/test_data_ram_2015.txt', 'the file saving testing data')
tf.app.flags.DEFINE_string('visible_devices', '1', 'visible_devices')
tf.flags.DEFINE_float("memory_fraction", 0.5, "per_process_gpu_memory_fraction")


os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.visible_devices  # 使用 GPU 1


def main(_):
    print('Loading data info ...')
    FLAGS.word2id, FLAGS.max_sentence_len, FLAGS.max_aspect_len = get_data_info(FLAGS.train_fname, FLAGS.test_fname, FLAGS.data_info, FLAGS.pre_processed)

    print('Loading training data and testing data ...')
    train_data = read_data(FLAGS.train_fname, FLAGS.word2id, FLAGS.max_sentence_len, FLAGS.max_aspect_len, FLAGS.train_data, FLAGS.pre_processed)
    test_data = read_data(FLAGS.test_fname, FLAGS.word2id, FLAGS.max_sentence_len,  FLAGS.max_aspect_len, FLAGS.test_data, FLAGS.pre_processed)

    print('Loading pre-trained word vectors ...')
    FLAGS.word2vec = load_word_embeddings(FLAGS.embedding_fname, FLAGS.embedding_dim, FLAGS.word2id)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.memory_fraction)
    session_conf = tf.ConfigProto(
        gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        model = ATAELSTM_RelationNetwork_v3(FLAGS, sess)
        model.build_model()
        model.run(train_data, test_data)


if __name__ == '__main__':
    tf.app.run()
