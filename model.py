#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class Model():
    def __init__(self, learning_rate=0.001, batch_size=16, num_steps=32, num_words=5000, dim_embedding=128, rnn_layers=3, embedding_file=None):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_words = num_words     # vocabulary size
        self.dim_embedding = dim_embedding   # hidden size which equals to
        self.rnn_layers = rnn_layers   # lstm layers which is the number of lstm_cells
        self.learning_rate = learning_rate
        self.embedding_file = embedding_file

    def build(self):
        # global step
        self.global_step = tf.Variable(
            0, trainable=False, name='self.global_step', dtype=tf.int64)

        self.X = tf.placeholder(
            tf.int32, shape=[None, self.num_steps], name='input')
        self.Y = tf.placeholder(
            tf.int32, shape=[None, self.num_steps], name='label')

        self.keep_prob = tf.placeholder(tf.float32, name='self.keep_prob')

        with tf.variable_scope('embedding'):
            if self.embedding_file:
                # if embedding file provided, use it.
                embedding = np.load(self.embedding_file)
                embed = tf.constant(embedding, name='embedding')
            else:
                # if not, initialize an embedding and train it.
                embed = tf.get_variable(
                    'embedding', [self.num_words, self.dim_embedding])
                tf.summary.histogram('embed', embed)

            # get the embedding of input data. shape=[batch_size, num_steps, dim_embedding]
            data = tf.nn.embedding_lookup(embed, self.X)

        with tf.variable_scope('rnn'):
            ########################## This is the start line of my code for rnn ############################################

            # step1: define LSTM_cell forget_bias=1.0
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.dim_embedding, state_is_tuple=True)

            # step2：add dropout layer, generally we only use output_keep_prob
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=self.keep_prob)

            # step3：implement multi-layer LSTM by calling MultiRNNCell
            mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.rnn_layers, state_is_tuple=True)

            # step4：initialize state with zero
            self.state_tensor = mlstm_cell.zero_state(self.batch_size, dtype=tf.float32)

            # step5：run it by calling dynamic_rnn
            outputs_tensor, self.outputs_state_tensor = tf.nn.dynamic_rnn(mlstm_cell, inputs=data, initial_state=self.state_tensor, time_major=False)

            #h_state = outputs_tensor[:, -1, :]
            #h_state = self.outputs_state_tensor
            tf.summary.histogram('outputs_state_tensor', self.outputs_state_tensor)

            ########################## This is the bottom line of my code for rnn ############################################



        # concate every time step
        # shape=[batch_size, num_steps, dim_embedding] ---> [batch_size, num_steps*dim_embedding]
        seq_output = tf.concat(outputs_tensor, 1)

        # flatten it
        seq_output_final = tf.reshape(seq_output, [-1, self.dim_embedding])

        with tf.variable_scope('softmax'):
            ########################## This is the start line of my code for softmax ############################################

            W = tf.Variable(tf.truncated_normal([self.dim_embedding, self.num_words], stddev=0.1), dtype=tf.float32)
            bias = tf.Variable(tf.constant(0.1,shape=[self.num_words]), dtype=tf.float32)
            #bias = tf.Variable(tf.zero(self.num_words))
            logits = tf.matmul(seq_output_final, W) + bias

            ########################## This is the bottom line of my code for softmax ############################################

        tf.summary.histogram('logits', logits)

        self.predictions = tf.nn.softmax(logits, name='predictions')

        y_one_hot = tf.one_hot(self.Y, self.num_words)
        y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
        loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y_reshaped)
        self.loss = tf.reduce_mean(loss)

        # gradient clip
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(
            zip(grads, tvars), global_step=self.global_step)

        tf.summary.scalar('loss', self.loss)

        self.merged_summary_op = tf.summary.merge_all()
