from __future__ import division
import tensorflow as tf
from Attention_inter import inter_attention_v3_3


def tf_f1_score(y_true, y_pred):
    """Computes 3 different f1 scores, micro macro
    weighted.
    micro: f1 score accross the classes, as 1
    macro: mean of f1 scores per class
    weighted: weighted average of f1 scores per class,
            weighted from the support of each class


    Args:
        y_true (Tensor): labels, with shape (batch, num_classes)
        y_pred (Tensor): model's predictions, same shape as y_true

    Returns:
        tuple(Tensor): (micro, macro, weighted)
                    tuple of the computed f1 scores
    """

    f1s = [0, 0, 0]

    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)

    for i, axis in enumerate([None, 0]):
        TP = tf.count_nonzero(y_pred * y_true, axis=axis)
        FP = tf.count_nonzero(y_pred * (y_true - 1), axis=axis)
        FN = tf.count_nonzero((y_pred - 1) * y_true, axis=axis)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

        f1s[i] = tf.reduce_mean(f1)

    weights = tf.reduce_sum(y_true, axis=0)
    weights /= tf.reduce_sum(weights)

    f1s[2] = tf.reduce_sum(f1 * weights)

    micro, macro, weighted = f1s
    return micro, macro, weighted


class ATAELSTM_inner_attention_v3(object):
    def __init__(self, sequence_length, num_classes, vocab_size, target_length, embedding_size, hidden_size, n_hop):
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_x_target = tf.placeholder(tf.int32, [None, target_length], name="input_x_target")
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope("embedding_layer"), tf.device("/cpu:0"):
            self.W_context_target = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.01, 0.01, name="W_context"))
            self.embedding_context_chars = tf.nn.embedding_lookup(self.W_context_target, self.input_x)
            self.embedding_tartget_chars = tf.nn.embedding_lookup(self.W_context_target, self.input_x_target)
            self.embedding_target_chars_average = tf.reduce_sum(self.embedding_tartget_chars, axis=1, keep_dims=True)

            self.embedding_context_chars_dropout = tf.nn.dropout(self.embedding_context_chars, self.dropout_keep_prob)
            self.embedding_target_chars_dropout = tf.nn.dropout(self.embedding_target_chars_average, self.dropout_keep_prob)
            self.embedding_target_chars_dropout_tile = tf.tile(self.embedding_target_chars_dropout, [1, sequence_length, 1])
            self.context_concatenate_target = tf.concat([self.embedding_context_chars_dropout, self.embedding_target_chars_dropout_tile], axis=-1)
        memory = []
        dynamic_batch_size = tf.shape(self.input_x)[0]
        # memory_0 = tf.constant(np.random.normal(0.0, 0.001, size=(dynamic_batch_size, self.embedding_dim)))
        memory_0 = tf.random_normal([dynamic_batch_size, embedding_size], mean=0.0, stddev=0.01, dtype=tf.float32)
        memory.append(memory_0)
        for i in range(n_hop):
            with tf.name_scope("lstm_layer"), tf.variable_scope("lstm", reuse=tf.AUTO_REUSE):
                cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
                outputs, _ = tf.nn.dynamic_rnn(cell, self.context_concatenate_target, dtype=tf.float32)

            with tf.name_scope("attention_layer"),tf.variable_scope("attention", reuse=tf.AUTO_REUSE):
                memory_final_3dim = tf.reshape(memory[-1], [-1, 1, embedding_size])
                memory_final_3dim_til = tf.tile(memory_final_3dim, [1, sequence_length, 1])
                # v3.3
                attention_output = inter_attention_v3_3(outputs, self.embedding_target_chars_dropout_tile, memory_final_3dim_til)
                memory.append(attention_output)

        with tf.name_scope("output_layer"):
            self.W = tf.get_variable("W_1", shape=[embedding_size, num_classes],
                                     initializer=tf.contrib.layers.xavier_initializer())
            self.b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_1")
            self.scores = tf.nn.xw_plus_b(memory[-1], self.W, self.b)
            self.predictions = tf.argmax(self.scores, axis=1, name="predictions")

        with tf.name_scope("loss"):
            # reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        with tf.name_scope("multi-label-f1"):
            y_pred = tf.one_hot(self.predictions, num_classes)
            y_true = self.input_y
            self.micro, self.macro, self.weighted = tf_f1_score(y_true, y_pred)