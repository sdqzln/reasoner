import tensorflow as tf
from tensorflow.contrib import layers


def inter_attention_context_mat_context(context, return_alphas=False):
    """
    attention计算方式：s = Relu(context.context + b), a = softmax(s)
    其中，"." 表示矩阵相乘
    :param context: [batch_size, sequence_length, dim]
    :param return_alphas:
    :return:
    """

    context_transpose = tf.transpose(context, [0, 2, 1])
    regularizer = layers.l2_regularizer(0.1)
    b_omega_0 = tf.get_variable(name="B0",
                                shape=[context.shape[1].value, 1],
                                initializer=tf.constant_initializer(0.1),
                                regularizer=regularizer)
    context_matrix = tf.nn.relu(tf.matmul(context, context_transpose) + b_omega_0)

    with tf.name_scope("max-pooling"):
        context_matrix_expand = tf.expand_dims(context_matrix, axis=-1)
        pool_output = tf.nn.max_pool(
            value=context_matrix_expand,
            ksize=[1, 1, context.shape[1].value, 1],
            strides=[1, 1, 1, 1],
            padding="VALID"
        )
    pool_output_reshape = tf.reshape(pool_output, [-1, 1, context.shape[1].value])
    alpha = tf.nn.softmax(pool_output_reshape)
    outputs = tf.reshape(tf.matmul(alpha, context), [-1, context.shape[2].value])

    if not return_alphas:
        return outputs
    else:
        return outputs, alpha


def inter_attention_context_mat_context_mat_memory_w(context, memory, return_alphas=False):
    """
    attention 计算方式：s = Relu(context.context_transpose.memory.w + b), a = softmax(s)
    其中，"." 表示矩阵相乘
    :param context:[batch_size, sequence_length, dim0]
    :param memory:[batch_size, sequence_length, dim1]
    :param return_alphas:
    :return:
    """

    context_transpose = tf.transpose(context, [0, 2, 1])
    regularizer = layers.l2_regularizer(0.1)
    w = tf.get_variable(
        name='W0',
        shape=[memory.shape[-1].value, memory.shape[1].value],
        initializer=tf.random_uniform_initializer(-0.1, 0.1),
        regularizer=tf.contrib.layers.l2_regularizer(0.1)
    )
    b_omega_0 = tf.get_variable(name="B0",
                        shape=[context.shape[1].value, 1],
                        initializer=tf.constant_initializer(0.1),
                        regularizer=regularizer)
    context_matrix = tf.nn.relu(tf.tensordot(tf.matmul(tf.matmul(context, context_transpose), memory), w, axes=1) + b_omega_0)
    # context_matrix = tf.nn.relu(tf.matmul(context, context_transpose) + b_omega_0)

    with tf.name_scope("max-pooling"):
        context_matrix_expand = tf.expand_dims(context_matrix, axis=-1)
        pool_output = tf.nn.max_pool(
            value= context_matrix_expand,
            ksize=[1,1, context.shape[1].value, 1],
            strides=[1,1,1,1],
            padding="VALID"
        )
    pool_output_reshape = tf.reshape(pool_output, [-1, 1, context.shape[1].value])
    alpha = tf.nn.softmax(pool_output_reshape)
    outputs = tf.reshape(tf.matmul(alpha, context), [-1, context.shape[2].value])\

    if not return_alphas:
        return outputs
    else:
        return outputs, alpha


def inter_attention_context_concat_context(context, return_alphas=False):
    """
    attention计算方式：s = Relu(W[context:context_transpose] + b), a = softmax(s)
    :param context: [batch_size, sequence_length, dim]
    :param return_alphas:
    :return:
    """
    with tf.device('/cpu:0'):
        context_til = tf.tile(context, [1, 1, context.shape[1].value])
        context_til_reshape = tf.reshape(context_til, [-1, context.shape[1].value, context.shape[2].value, context.shape[1].value])
        context_til_reshape = tf.transpose(context_til_reshape, [0, 1, 3, 2])
        context_transpose = tf.transpose(context, [0, 2, 1])
        context_transpose_til = tf.tile(context_transpose, [1, context.shape[1].value, 1])
        context_transpose_til_reshape = tf.reshape(context_transpose_til, [-1, context.shape[1].value, context.shape[2].value, context.shape[1].value])
        context_transpose_til_reshape = tf.transpose(context_transpose_til_reshape, [0, 1, 3, 2])
        context_concate = tf.concat([context_til_reshape, context_transpose_til_reshape], axis=-1)
    regularizer = layers.l2_regularizer(0.1)
    w = tf.get_variable(
        name='W0',
        shape=[2*context.shape[-1].value, 1],
        initializer=tf.random_uniform_initializer(-0.1, 0.1),
        regularizer=tf.contrib.layers.l2_regularizer(0.1)
    )
    b_omega_0 = tf.get_variable(name="B0",
                        shape=[context.shape[1].value, 1],
                        initializer=tf.constant_initializer(0.1),
                        regularizer=regularizer)
    context_matrix = tf.nn.relu(tf.tensordot(context_concate, w, axes=1) + b_omega_0)

    with tf.name_scope("max-pooling"):
        pool_output = tf.nn.max_pool(
            value= context_matrix,
            ksize=[1,1, context.shape[1].value, 1],
            strides=[1,1,1,1],
            padding="VALID"
        )
    pool_output_reshape = tf.reshape(pool_output, [-1, 1, context.shape[1].value])
    alpha = tf.nn.softmax(pool_output_reshape)
    outputs = tf.reshape(tf.matmul(alpha, context), [-1, context.shape[2].value])


    if not return_alphas:
        return outputs
    else:
        return outputs, alpha


def inter_attention_context_mat_context_concate_memory_concate_aspect(context, aspect, memory, return_alphas=False):
    """
    attention 计算方式：
    s1 = Relu(context.context_transpose.memory.w + b),
    s = Relu(W.[s1;aspect;memory] + b)
    a = softmax(s)
    其中，"." 表示矩阵相乘
    :param context:[batch_size, sequence_length, dim0]
    :param aspect:[batch_size, sequence_length, dim1]
    :param memory:[batch_size, sequence_length, dim2]
    :param return_alphas:
    :return:
    """

    context_transpose = tf.transpose(context, [0, 2, 1])
    regularizer = layers.l2_regularizer(0.1)
    w = tf.get_variable(
        name='W0',
        shape=[context.shape[1].value, context.shape[-1].value],
        initializer=tf.random_uniform_initializer(-0.1, 0.1),
        regularizer=tf.contrib.layers.l2_regularizer(0.1)
    )
    b_omega_0 = tf.get_variable(name="B0",
                        shape=[context.shape[1].value, 1],
                        initializer=tf.constant_initializer(0.1),
                        regularizer=regularizer)
    w_1 = tf.get_variable(
        name='W1',
        shape=[3*context.shape[-1].value, context.shape[1].value],
        initializer=tf.random_uniform_initializer(-0.1, 0.1),
        regularizer=tf.contrib.layers.l2_regularizer(0.1)
    )
    b_omega_1 = tf.get_variable(name="B1",
                                shape=[context.shape[1].value, 1],
                                initializer=tf.constant_initializer(0.1),
                                regularizer=regularizer)
    context_matrix = tf.nn.relu(tf.tensordot(tf.matmul(context, context_transpose), w, axes=1) + b_omega_0)
    context_matrix_2 = tf.nn.relu(tf.tensordot(tf.concat([context_matrix, aspect, memory], axis=-1), w_1, axes=1) + b_omega_1)

    with tf.name_scope("max-pooling"):
        context_matrix_2_expand = tf.expand_dims(context_matrix_2, axis=-1)
        pool_output = tf.nn.max_pool(
            value= context_matrix_2_expand,
            ksize=[1,1, context.shape[1].value, 1],
            strides=[1,1,1,1],
            padding="VALID"
        )
    pool_output_reshape = tf.reshape(pool_output, [-1, 1, context.shape[1].value])
    alpha = tf.nn.softmax(pool_output_reshape)
    outputs = tf.reshape(tf.matmul(alpha, context), [-1, context.shape[2].value])

    if not return_alphas:
        return outputs
    else:
        return outputs, alpha

def inter_attention_context_concate_memory_concate_aspect(context, aspect, memory, return_alphas=False):
    """
    attention 计算方式：
    s = Relu(W.[context; aspect; memory] + b)， w:[embedding_dim, sequence]
    a = softmax(maxpooling(s))
    其中，"." 表示矩阵相乘
    :param context:[batch_size, sequence_length, dim0]
    :param aspect:[batch_size, sequence_length, dim1]
    :param memory:[batch_size, sequence_length, dim2]
    :param return_alphas:
    :return:
    """

    concate_vector = tf.concat([context, aspect, memory], axis=-1)
    regularizer = layers.l2_regularizer(0.1)
    w_1 = tf.get_variable(
        name='W1',
        shape=[concate_vector.shape[-1].value, context.shape[1].value],
        initializer=tf.random_uniform_initializer(-0.1, 0.1),
        regularizer=tf.contrib.layers.l2_regularizer(0.1)
    )
    b_omega_1 = tf.get_variable(name="B1",
                                shape=[context.shape[1].value, 1],
                                initializer=tf.constant_initializer(0.1),
                                regularizer=regularizer)

    context_matrix_2 = tf.nn.relu(tf.tensordot(concate_vector, w_1, axes=1) + b_omega_1)

    with tf.name_scope("max-pooling"):
        context_matrix_2_expand = tf.expand_dims(context_matrix_2, axis=-1)
        pool_output = tf.nn.max_pool(
            value= context_matrix_2_expand,
            ksize=[1,1, context.shape[1].value, 1],
            strides=[1,1,1,1],
            padding="VALID"
        )
    pool_output_reshape = tf.reshape(pool_output, [-1, 1, context.shape[1].value])
    alpha = tf.nn.softmax(pool_output_reshape)
    outputs = tf.reshape(tf.matmul(alpha, context), [-1, context.shape[2].value])

    if not return_alphas:
        return outputs
    else:
        return outputs, alpha


def inter_attention_context_concate_memory_concate_aspect_nonmaxpooling(context, aspect, memory, return_alphas=False):
    """
    attention 计算方式：
    s = Relu(W.[context; aspect; memory] + b), w:[embedding, 1]
    a = softmax(s)
    其中，"." 表示矩阵相乘
    :param context:[batch_size, sequence_length, dim0]
    :param aspect:[batch_size, sequence_length, dim1]
    :param memory:[batch_size, sequence_length, dim2]
    :param return_alphas:
    :return:
    """

    concate_vector = tf.concat([context, aspect, memory], axis=-1)
    regularizer = layers.l2_regularizer(0.1)
    w_1 = tf.get_variable(
        name='W1',
        shape=[concate_vector.shape[-1].value, 1],
        initializer=tf.random_uniform_initializer(-0.1, 0.1),
        regularizer=tf.contrib.layers.l2_regularizer(0.1)
    )
    b_omega_1 = tf.get_variable(name="B1",
                                shape=[1, 1],
                                initializer=tf.constant_initializer(0.1),
                                regularizer=regularizer)

    context_matrix = tf.nn.relu(tf.tensordot(concate_vector, w_1, axes=1) + b_omega_1)
    context_matrix_2 = tf.reshape(context_matrix, [-1, 1, context.shape[1].value])
    alpha = tf.nn.softmax(context_matrix_2)
    outputs = tf.reshape(tf.matmul(alpha, context), [-1, context.shape[2].value])

    if not return_alphas:
        return outputs
    else:
        return outputs, alpha

# v3.2
def inter_attention_v3_2(context, aspect, memory, aspect_2, return_alphas=False):
    """
    attention 计算方式：
    s = Relu(W.[context; aspect; memory] + b)
    a = softmax(s)
    其中，"." 表示矩阵相乘
    :param context:[batch_size, sequence_length, dim0]
    :param aspect:[batch_size, sequence_length, dim1]
    :param memory:[batch_size, sequence_length, dim2]
    :param return_alphas:
    :return:
    """


    concate_vector = tf.concat([context, aspect, memory], axis=-1)
    regularizer = layers.l2_regularizer(0.1)

    w_0 = tf.get_variable(
        name='W0',
        shape=[concate_vector.shape[-1].value, context.shape[1].value],
        initializer=tf.random_uniform_initializer(-0.1, 0.1),
        regularizer=tf.contrib.layers.l2_regularizer(0.1)
    )
    b_omega_0 = tf.get_variable(name="B0",
                                shape=[context.shape[1].value, 1],
                                initializer=tf.constant_initializer(0.1),
                                regularizer=regularizer)

    context_matrix_0 = tf.nn.relu(tf.tensordot(concate_vector, w_0, axes=1) + b_omega_0)

    with tf.name_scope("max-pooling"):
        context_matrix_0_expand = tf.expand_dims(context_matrix_0, axis=-1)
        pool_output = tf.nn.max_pool(
            value=context_matrix_0_expand,
            ksize=[1, 1, context.shape[1].value, 1],
            strides=[1, 1, 1, 1],
            padding="VALID"
        )
    pool_output_reshape = tf.reshape(pool_output, [-1, 1, context.shape[1].value])
    alpha_0 = tf.nn.softmax(pool_output_reshape)
    outputs_0 = tf.matmul(alpha_0, context)


    w_1 = tf.get_variable(
        name='W1',
        shape=[concate_vector.shape[-1].value, 1],
        initializer=tf.random_uniform_initializer(-0.1, 0.1),
        regularizer=tf.contrib.layers.l2_regularizer(0.1)
    )
    b_omega_1 = tf.get_variable(name="B1",
                                shape=[1, 1],
                                initializer=tf.constant_initializer(0.1),
                                regularizer=regularizer)

    context_matrix_1 = tf.nn.relu(tf.tensordot(concate_vector, w_1, axes=1) + b_omega_1)
    context_matrix_1 = tf.reshape(context_matrix_1, [-1, 1, context.shape[1].value])
    alpha_1 = tf.nn.softmax(context_matrix_1)
    outputs_1 = tf.matmul(alpha_1, context)

    outputs_concate = tf.concat([outputs_0, outputs_1], axis=1)
    outputs_concate_aspect2 = tf.concat([outputs_concate, aspect_2], axis=-1)
    w_2 = tf.get_variable(
        name='W2',
        shape=[context.shape[-1].value*2, 1],
        initializer=tf.random_uniform_initializer(-0.1, 0.1),
        regularizer=tf.contrib.layers.l2_regularizer(0.1)
    )
    b_omega_2 = tf.get_variable(name="B2",
                                shape=[1, 1],
                                initializer=tf.constant_initializer(0.1),
                                regularizer=regularizer)

    context_matrix_2 = tf.nn.relu(tf.tensordot(outputs_concate_aspect2, w_2, axes=1) + b_omega_2)
    context_matrix_2 = tf.reshape(context_matrix_2, [-1, 1, outputs_concate_aspect2.shape[1].value])
    alpha = tf.nn.softmax(context_matrix_2)
    outputs = tf.reshape(tf.matmul(alpha, outputs_concate), [-1, context.shape[2].value])


    if not return_alphas:
        return outputs
    else:
        return outputs, alpha

# v3.3
def inter_attention_v3_3(context, aspect, memory, return_alphas=False):
    """
    attention 计算方式：
    s = Relu(W.[context; aspect; memory] + b)
    a = softmax(s)
    其中，"." 表示矩阵相乘
    :param context:[batch_size, sequence_length, dim0]
    :param aspect:[batch_size, sequence_length, dim1]
    :param memory:[batch_size, sequence_length, dim2]
    :param return_alphas:
    :return:
    """


    concate_vector = tf.concat([context, aspect, memory], axis=-1)
    regularizer = layers.l2_regularizer(0.1)

    w_0 = tf.get_variable(
        name='W0',
        shape=[concate_vector.shape[-1].value, context.shape[1].value],
        initializer=tf.random_uniform_initializer(-0.1, 0.1),
        regularizer=tf.contrib.layers.l2_regularizer(0.1)
    )
    b_omega_0 = tf.get_variable(name="B0",
                                shape=[context.shape[1].value, 1],
                                initializer=tf.constant_initializer(0.1),
                                regularizer=regularizer)

    context_matrix_0 = tf.nn.relu(tf.tensordot(concate_vector, w_0, axes=1) + b_omega_0)

    with tf.name_scope("max-pooling"):
        context_matrix_0_expand = tf.expand_dims(context_matrix_0, axis=-1)
        pool_output = tf.nn.max_pool(
            value=context_matrix_0_expand,
            ksize=[1, 1, context.shape[1].value, 1],
            strides=[1, 1, 1, 1],
            padding="VALID"
        )
    pool_output_reshape = tf.reshape(pool_output, [-1, 1, context.shape[1].value])
    alpha_0 = tf.nn.softmax(pool_output_reshape)
    outputs_0 = tf.reshape(tf.matmul(alpha_0, context), [-1,  context.shape[-1].value])


    w_1 = tf.get_variable(
        name='W1',
        shape=[concate_vector.shape[-1].value, 1],
        initializer=tf.random_uniform_initializer(-0.1, 0.1),
        regularizer=tf.contrib.layers.l2_regularizer(0.1)
    )
    b_omega_1 = tf.get_variable(name="B1",
                                shape=[1, 1],
                                initializer=tf.constant_initializer(0.1),
                                regularizer=regularizer)

    context_matrix_1 = tf.nn.relu(tf.tensordot(concate_vector, w_1, axes=1) + b_omega_1)
    context_matrix_1 = tf.reshape(context_matrix_1, [-1, 1, context.shape[1].value])
    alpha_1 = tf.nn.softmax(context_matrix_1)
    outputs_1 = tf.reshape(tf.matmul(alpha_1, context), [-1, context.shape[-1].value])

    outputs_concate = tf.concat([outputs_0, outputs_1], axis=-1)
    w_2 = tf.get_variable(
        name='W2',
        shape=[context.shape[-1].value*2, context.shape[-1].value],
        initializer=tf.random_uniform_initializer(-0.1, 0.1),
        regularizer=tf.contrib.layers.l2_regularizer(0.1)
    )
    b_omega_2 = tf.get_variable(name="B2",
                                shape=[1, 1],
                                initializer=tf.constant_initializer(0.1),
                                regularizer=regularizer)

    outputs = tf.nn.relu(tf.tensordot(outputs_concate, w_2, axes=1) + b_omega_2)


    if not return_alphas:
        return outputs
    else:
        return outputs, alpha_0, alpha_1

# v7
def inter_attention_context_concate_memory_concate_aspect(context, aspect, memory, return_alphas=False):
    """
    attention 计算方式：
    s = Relu(W.[context; aspect; memory] + b)， w:[embedding_dim, sequence]
    a = softmax(maxpooling(s))
    其中，"." 表示矩阵相乘
    :param context:[batch_size, sequence_length, dim0]
    :param aspect:[batch_size, sequence_length, dim1]
    :param memory:[batch_size, sequence_length, dim2]
    :param return_alphas:
    :return:
    """

    concate_vector = tf.concat([context, aspect, memory], axis=-1)
    regularizer = layers.l2_regularizer(0.1)
    w_1 = tf.get_variable(
        name='W1',
        shape=[concate_vector.shape[-1].value, context.shape[1].value],
        initializer=tf.random_uniform_initializer(-0.1, 0.1),
        regularizer=tf.contrib.layers.l2_regularizer(0.1)
    )
    b_omega_1 = tf.get_variable(name="B1",
                                shape=[context.shape[1].value, 1],
                                initializer=tf.constant_initializer(0.1),
                                regularizer=regularizer)

    context_matrix_2 = tf.nn.relu(tf.tensordot(concate_vector, w_1, axes=1) + b_omega_1)

    with tf.name_scope("max-pooling"):
        context_matrix_2_expand = tf.expand_dims(context_matrix_2, axis=-1)
        pool_output = tf.nn.max_pool(
            value=context_matrix_2_expand,
            ksize=[1,1, context.shape[1].value, 1],
            strides=[1,1,1,1],
            padding="VALID"
        )
    pool_output_reshape = tf.reshape(pool_output, [-1, 1, context.shape[1].value])
    alpha = tf.nn.softmax(pool_output_reshape)
    outputs = tf.reshape(tf.matmul(alpha, context), [-1, context.shape[2].value])

    if not return_alphas:
        return outputs
    else:
        return outputs, alpha