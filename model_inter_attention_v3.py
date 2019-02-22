import tensorflow as tf
from tensorflow.python.ops import math_ops
import numpy as np
import time
from data_utils_ram_15 import get_batch_index
from Attention_inter import inter_attention_v3_3


class ATAELSTM_RelationNetwork_v3(object):

    def __init__(self, config, sess):
        self.embedding_dim = config.embedding_dim
        self.batch_size = config.batch_size
        self.n_epoch = config.n_epoch
        self.n_hidden = config.n_hidden
        self.n_class = config.n_class
        self.n_hop = config.n_hop
        self.learning_rate = config.learning_rate
        self.l2_reg = config.l2_reg
        self.dropout = config.dropout

        self.word2id = config.word2id
        self.max_sentence_len = config.max_sentence_len
        self.max_aspect_len = config.max_aspect_len
        self.word2vec = config.word2vec
        self.sess = sess

        self.timestamp = str(int(time.time()))

    def build_model(self):
        with tf.name_scope('inputs'):
            self.sentences = tf.placeholder(tf.int32, [None, self.max_sentence_len])
            self.aspects = tf.placeholder(tf.int32, [None, self.max_aspect_len])
            self.sentence_lens = tf.placeholder(tf.int32, None)
            self.sentence_locs = tf.placeholder(tf.float32, [None, self.max_sentence_len])
            self.labels = tf.placeholder(tf.int32, [None, self.n_class])
            self.dropout_keep_prob = tf.placeholder(tf.float32)

            inputs = tf.nn.embedding_lookup(self.word2vec, self.sentences)
            inputs = tf.cast(inputs, tf.float32)
            inputs = tf.nn.dropout(inputs, keep_prob=self.dropout_keep_prob)
            aspect_inputs = tf.nn.embedding_lookup(self.word2vec, self.aspects)
            aspect_inputs = tf.cast(aspect_inputs, tf.float32)
            aspect_inputs = tf.reduce_mean(aspect_inputs, 1)
            aspect_inputs_3dim = tf.reshape(aspect_inputs, [-1, 1, self.embedding_dim])
            aspect_inputs_til = tf.tile(aspect_inputs_3dim, [1, self.max_sentence_len, 1])

            memory = []
            dynamic_batch_size = tf.shape(self.sentences)[0]
            # memory_0 = tf.constant(np.random.normal(0.0, 0.001, size=(dynamic_batch_size, self.embedding_dim)))
            memory_0 = tf.random_normal([dynamic_batch_size, self.embedding_dim], mean=0.0, stddev=0.01,
                                        dtype=tf.float32)
            memory.append(memory_0)

        for i in range(self.n_hop):
            with tf.name_scope("lstm"), tf.variable_scope("lstm", reuse=tf.AUTO_REUSE):
                inputs_concate_aspect = tf.concat([inputs, aspect_inputs_til], axis=-1)
                cell = tf.contrib.rnn.BasicLSTMCell(self.embedding_dim)
                outputs, _ = tf.nn.dynamic_rnn(cell, inputs_concate_aspect, dtype=tf.float32)

            with tf.name_scope("attention"), tf.variable_scope("attention", reuse=tf.AUTO_REUSE):
                memory_final_3dim = tf.reshape(memory[-1], [-1, 1, self.embedding_dim])
                memory_final_3dim_til = tf.tile(memory_final_3dim, [1, self.max_sentence_len, 1])
                # memory_final_3dim = tf.reshape(memory[-1], [-1, self.embedding_dim, 1])
                # attention_output = attention_10(outputs, aspect_inputs_til, memory_final_3dim)
                # attention_output = attention_10_v2(outputs, memory_final_3dim_til)
                # attention_output = inter_attention_context_mat_context_mat_memory_w(outputs, memory_final_3dim_til)
                # attention_output = inter_attention_context_concat_context(outputs)
                # v3
                # attention_output = inter_attention_context_concate_memory_concate_aspect(outputs, aspect_inputs_til, memory_final_3dim_til)
                # v3.1
                # attention_output = inter_attention_context_concate_memory_concate_aspect_nonmaxpooling(outputs, aspect_inputs_til, memory_final_3dim_til)
                # v3.2
                # aspect_til_2 = tf.tile(aspect_inputs_3dim, [1, 2, 1])
                # attention_output = inter_attention_v3_2(outputs, aspect_inputs_til, memory_final_3dim_til, aspect_til_2)
                # v3.3
                attention_output = inter_attention_v3_3(outputs, aspect_inputs_til, memory_final_3dim_til)
                memory.append(attention_output)

        with tf.name_scope("output"):
            # self.W = tf.get_variable("W_1", shape=[self.embedding_dim, 3],
            #                            initializer=tf.contrib.layers.xavier_initializer())
            # self.b = tf.Variable(tf.constant(0.1, shape=[3]), name="b_1")
            self.W = tf.get_variable("W_1", shape=[self.embedding_dim, 3],
                                     initializer=tf.contrib.layers.xavier_initializer())
            self.b = tf.Variable(tf.constant(0.1, shape=[3]), name="b_1")
            self.scores = tf.nn.xw_plus_b(memory[-1], self.W, self.b)

        with tf.name_scope('loss'):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.labels))
            self.global_step = tf.Variable(0, name="tr_global_step", trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost,
                                                                                               global_step=self.global_step)

        with tf.name_scope('predict'):
            self.predict_label = tf.argmax(self.scores, 1)
            self.correct_pred = tf.equal(self.predict_label, tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_sum(tf.cast(self.correct_pred, tf.int32))

        summary_loss = tf.summary.scalar('loss', self.cost)
        summary_acc = tf.summary.scalar('acc', self.accuracy)
        self.train_summary_op = tf.summary.merge([summary_loss, summary_acc])
        self.test_summary_op = tf.summary.merge([summary_loss, summary_acc])
        _dir = 'logs/' + str(self.timestamp) + '_r' + str(self.learning_rate) + '_b' + str(
            self.batch_size) + '_l' + str(self.l2_reg)
        self.train_summary_writer = tf.summary.FileWriter(_dir + '/train', self.sess.graph)
        self.test_summary_writer = tf.summary.FileWriter(_dir + '/test', self.sess.graph)

    def train(self, data):
        sentences, aspects, sentence_lens, sentence_locs, labels = data
        cost, cnt = 0., 0
        for sample, num in self.get_batch_data(sentences, aspects, sentence_lens, sentence_locs, labels,
                                               self.batch_size, True, self.dropout):
            _, loss, step, summary = self.sess.run([self.optimizer, self.cost, self.global_step, self.train_summary_op],
                                                   feed_dict=sample)
            self.train_summary_writer.add_summary(summary, step)
            cost += loss * num
            cnt += num

        _, train_acc = self.test(data)
        return cost / cnt, train_acc

    def test(self, data):
        sentences, aspects, sentence_lens, sentence_locs, labels = data
        cost, acc, cnt = 0., 0, 0
        for sample, num in self.get_batch_data(sentences, aspects, sentence_lens, sentence_locs, labels, len(sentences),
                                               False, 1.0):
            loss, accuracy, step, summary = self.sess.run(
                [self.cost, self.accuracy, self.global_step, self.test_summary_op], feed_dict=sample)
            cost += loss * num
            acc += accuracy
            cnt += num

        self.test_summary_writer.add_summary(summary, step)
        return cost / cnt, acc / cnt

    def analysis(self, data, tag):
        sentences, aspects, sentence_lens, sentence_locs, labels = data
        with open('analysis/' + tag + '_' + str(self.timestamp) + '.txt', 'w') as f:
            for sample, num in self.get_batch_data(sentences, aspects, sentence_lens, sentence_locs, labels,
                                                   int(len(sentences) / 2) + 1, False, 1.0):
                scores, correct_pred, predict_label = self.sess.run(
                    [self.scores, self.correct_pred, self.predict_label], feed_dict=sample)
                for a, b, c in zip(scores, correct_pred, predict_label):
                    for i in a:
                        i = str(i).replace('\n', '')
                        f.write('%s\n' % i)
                    b = str(b).replace('\n', '')
                    c = str(c).replace('\n', '')
                    f.write('%s\n%s\n' % (b, c))
        print('Finishing analyzing %s data' % tag)

    def run(self, train_data, test_data):
        saver = tf.train.Saver(tf.trainable_variables())
        print('Training ...')
        self.sess.run(tf.global_variables_initializer())
        max_acc, step = 0., -1
        for i in range(self.n_epoch):
            train_loss, train_acc = self.train(train_data)
            test_loss, test_acc = self.test(test_data)
            if test_acc > max_acc:
                max_acc = test_acc
                step = i
                saver.save(self.sess, 'models/model_iter' + str(self.timestamp), global_step=i)
            print('epoch %s: train-loss=%.6f; train-acc=%.6f; test-loss=%.6f; test-acc=%.6f;' % (
            str(i), train_loss, train_acc, test_loss, test_acc))
        print('The max accuracy of testing results is %s of step %s' % (max_acc, step))
        print('Analyzing ...')
        saver.restore(self.sess, tf.train.latest_checkpoint('models/'))

    def get_batch_data(self, sentences, aspects, sentence_lens, sentence_locs, labels, batch_size, is_shuffle,
                       keep_prob):
        for index in get_batch_index(len(sentences), batch_size, is_shuffle):
            feed_dict = {
                self.sentences: sentences[index],
                self.aspects: aspects[index],
                self.sentence_lens: sentence_lens[index],
                self.sentence_locs: sentence_locs[index],
                self.labels: labels[index],
                self.dropout_keep_prob: keep_prob,
            }
            yield feed_dict, len(index)
