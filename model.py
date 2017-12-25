import tensorflow as tf
import numpy as np

def fc(inputs, num_out, name, activation_fn=None, biased=True):
    w_init = tf.random_normal_initializer(stddev=0.02)
    return tf.layers.dense(inputs=inputs, units=num_out, activation=activation_fn, kernel_initializer=w_init, use_bias=biased, name=name)


def concat(inputs, axis, name):
    return tf.concat(values=inputs, axis=axis, name=name)

def batch_normalization(inputs, is_training, name, activation_fn=None):
    output = tf.layers.batch_normalization(
                    inputs,
                    momentum=0.95,
                    epsilon=1e-5,
                    training=is_training,
                    name=name
                )

    if activation_fn is not None:
        output = activation_fn(output)

    return output

def reshape(inputs, shape, name):
    return tf.reshape(inputs, shape, name)

def Conv2d(input, k_h, k_w, c_o, s_h, s_w, name, activation_fn=None, padding='VALID', biased=False):
    c_i = input.get_shape()[-1]
    w_init = tf.random_normal_initializer(stddev=0.02)

    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable(name='weights', shape=[k_h, k_w, c_i, c_o], initializer=w_init)
        output = convolve(input, kernel)

        if biased:
            biases = tf.get_variable(name='biases', shape=[c_o])
            output = tf.nn.bias_add(output, biases)
        if activation_fn is not None:
            output = activation_fn(output, name=scope.name)

        return output

def add(inputs, name):
    return tf.add_n(inputs, name=name)

def UpSample(inputs, size, method, align_corners, name):
    return tf.image.resize_images(inputs, size, method, align_corners)

def flatten(input, name):
    input_shape = input.get_shape()
    dim = 1
    for d in input_shape[1:].as_list():
        dim *= d
        input = tf.reshape(input, [-1, dim])
    
    return input

class Generator:
    def __init__(self, input_z, input_rnn, is_training, reuse):
        self.input_z = input_z
        self.input_rnn = input_rnn
        self.is_training = is_training
        self.reuse = reuse
        self.t_dim = 128
        self.gf_dim = 128
        self.image_size = 64
        self.c_dim = 3
        self._build_model()

    def _build_model(self):
        s = self.image_size
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

        gf_dim = self.gf_dim
        t_dim = self.t_dim
        c_dim = self.c_dim

        with tf.variable_scope("generator", reuse=self.reuse):
            net_txt = fc(inputs=self.input_rnn, num_out=t_dim, activation_fn=tf.nn.leaky_relu, name='rnn_fc')
            net_in = concat([self.input_z, net_txt], axis=1, name='concat_z_txt')

            net_h0 = fc(inputs=net_in, num_out=gf_dim*8*s16*s16, name='g_h0/fc', biased=False)
            net_h0 = batch_normalization(net_h0, activation_fn=None, is_training=self.is_training, name='g_h0/batch_norm')
            net_h0 = reshape(net_h0, [-1, s16, s16, gf_dim*8], name='g_h0/reshape')
            
            net = Conv2d(net_h0, 1, 1, gf_dim*2, 1, 1, name='g_h1_res/conv2d')
            net = batch_normalization(net, activation_fn=tf.nn.relu, is_training=self.is_training, name='g_h1_res/batch_norm')
            net = Conv2d(net, 3, 3, gf_dim*2, 1, 1, name='g_h1_res/conv2d2', padding='SAME')
            net = batch_normalization(net, activation_fn=tf.nn.relu, is_training=self.is_training, name='g_h1_res/batch_norm2')
            net = Conv2d(net, 3, 3, gf_dim*8, 1, 1, name='g_h1_res/conv2d3', padding='SAME')
            net = batch_normalization(net, activation_fn=None, is_training=self.is_training, name='g_h1_res/batch_norm3')

            net_h1 = add([net_h0, net], name='g_h1_res/add')
            net_h1_output = tf.nn.relu(net_h1)
            
            net_h2 = UpSample(net_h1_output, size=[s8, s8], method=1, align_corners=False, name='g_h2/upsample2d')
            net_h2 = Conv2d(net_h2, 3, 3, gf_dim*4, 1, 1, name='g_h2/conv2d', padding='SAME')
            net_h2 = batch_normalization(net_h2, activation_fn=None, is_training=self.is_training, name='g_h2/batch_norm')

            net = Conv2d(net_h2, 1, 1, gf_dim, 1, 1, name='g_h3_res/conv2d')
            net = batch_normalization(net, activation_fn=tf.nn.relu, is_training=self.is_training, name='g_h3_res/batch_norm')
            net = Conv2d(net, 3, 3, gf_dim, 1, 1, name='g_h3_res/conv2d2', padding='SAME')
            net = batch_normalization(net, activation_fn=tf.nn.relu, is_training=self.is_training, name='g_h3_res/batch_norm2')
            net = Conv2d(net, 3, 3, gf_dim*4, 1, 1, name='g_h3_res/conv2d3', padding='SAME')
            net = batch_normalization(net, activation_fn=None, is_training=self.is_training, name='g_h3_res/batch_norm3')
            
            net_h3 = add([net_h2, net], name='g_h3/add')
            net_h3_outputs = tf.nn.relu(net_h3)

            net_h4 = UpSample(net_h3_outputs, size=[s4, s4], method=1, align_corners=False, name='g_h4/upsample2d')
            net_h4 = Conv2d(net_h4, 3, 3, gf_dim*2, 1, 1, name='g_h4/conv2d', padding='SAME')
            net_h4 = batch_normalization(net_h4, activation_fn=tf.nn.relu, is_training=self.is_training, name='g_h4/batch_norm')

            net_h5 = UpSample(net_h4, size=[s2, s2], method=1, align_corners=False, name='g_h5/upsample2d')
            net_h5 = Conv2d(net_h5, 3, 3, gf_dim, 1, 1, name='g_h5/conv2d', padding='SAME')
            net_h5 = batch_normalization(net_h5, activation_fn=tf.nn.relu, is_training=self.is_training, name='g_h5/batch_norm')

            net_ho = UpSample(net_h5, size=[s, s], method=1, align_corners=False, name='g_ho/upsample2d')
            net_ho = Conv2d(net_ho, 3, 3, c_dim, 1, 1, name='g_ho/conv2d', padding='SAME', biased=True) ## biased = True

            self.outputs = tf.nn.tanh(net_ho)
            self.logits = net_ho

class Discriminator:
    def __init__(self, input_image, input_rnn, is_training, reuse):
        self.input_image = input_image
        self.input_rnn = input_rnn
        self.is_training = is_training
        self.reuse = reuse
        self.df_dim = 64
        self.t_dim = 128
        self.image_size = 64
        self._build_model()

    def _build_model(self):
        s = self.image_size
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

        df_dim = self.df_dim
        t_dim = self.t_dim

        with tf.variable_scope("discriminator", reuse=self.reuse):
            net_h0 = Conv2d(self.input_image, 4, 4, df_dim, 2, 2, name='d_h0/conv2d', activation_fn=tf.nn.leaky_relu, padding='SAME', biased=True)

            net_h1 = Conv2d(net_h0, 4, 4, df_dim*2, 2, 2, name='d_h1/conv2d', padding='SAME')
            net_h1 = batch_normalization(net_h1, activation_fn=tf.nn.leaky_relu, is_training=self.is_training, name='d_h1/batchnorm')

            net_h2 = Conv2d(net_h1, 4, 4, df_dim*4, 2, 2, name='d_h2/conv2d', padding='SAME')
            net_h2 = batch_normalization(net_h2, activation_fn=tf.nn.leaky_relu, is_training=self.is_training, name='d_h2/batchnorm')

            net_h3 = Conv2d(net_h2, 4, 4, df_dim*8, 2, 2, name='d_h3/conv2d', padding='SAME')
            net_h3 = batch_normalization(net_h3, activation_fn=None, is_training=self.is_training, name='d_h3/batchnorm')

            net = Conv2d(net_h3, 1, 1, df_dim*2, 1, 1, name='d_h4_res/conv2d')
            net = batch_normalization(net, activation_fn=tf.nn.leaky_relu, is_training=self.is_training, name='d_h4_res/batchnorm')
            net = Conv2d(net, 3, 3, df_dim*2, 1, 1, name='d_h4_res/conv2d2', padding='SAME')
            net = batch_normalization(net, activation_fn=tf.nn.leaky_relu, is_training=self.is_training, name='d_h4_res/batchnorm2')
            net = Conv2d(net, 3, 3, df_dim*8, 1, 1, name='d_h4_res/conv2d3', padding='SAME')
            net = batch_normalization(net, activation_fn=None, is_training=self.is_training, name='d_h4_res/batchnorm3')

            net_h4 = add([net_h3, net], name='d_h4/add')
            net_h4_outputs = tf.nn.leaky_relu(net_h4)

            net_txt = fc(self.input_rnn, num_out=t_dim, activation_fn=tf.nn.leaky_relu, name='d_reduce_txt/dense')
            net_txt = tf.expand_dims(net_txt, axis=1, name='d_txt/expanddim1')
            net_txt = tf.expand_dims(net_txt, axis=1, name='d_txt/expanddim2')
            net_txt = tf.tile(net_txt, [1, 4, 4, 1], name='d_txt/tile')
            
            net_h4_concat = concat([net_h4_outputs, net_txt], axis=3, name='d_h3_concat')

            net_h4 = Conv2d(net_h4_concat, 1, 1, df_dim*8, 1, 1, name='d_h3/conv2d_2')
            net_h4 = batch_normalization(net_h4, activation_fn=tf.nn.leaky_relu, is_training=self.is_training, name='d_h3/batch_norm_2')

            net_ho = Conv2d(net_h4, s16, s16, 1, s16, s16, name='d_ho/conv2d', biased=True) # biased = True

            self.outputs = tf.nn.sigmoid(net_ho)
            self.logits = net_ho

class rnn_encoder:
    def __init__(self, input_seqs, is_training, reuse):
        self.input_seqs = input_seqs
        self.is_training = is_training
        self.reuse = reuse
        self.t_dim = 128  
        self.rnn_hidden_size = 128
        self.vocab_size = 8000
        self.word_embedding_size = 256
        self.keep_prob = 1.0
        self.batch_size = 64
        self._build_model()

    def _build_model(self):
        w_init = tf.random_normal_initializer(stddev=0.02)
        LSTMCell = tf.contrib.rnn.BasicLSTMCell

        with tf.variable_scope("rnnftxt", reuse=self.reuse):
            word_embed_matrix = tf.get_variable('rnn/wordembed', 
                shape=(self.vocab_size, self.word_embedding_size),
                initializer=tf.random_normal_initializer(stddev=0.02),
                dtype=tf.float32)
            embedded_word_ids = tf.nn.embedding_lookup(word_embed_matrix, self.input_seqs)

            # RNN encoder
            LSTMCell = tf.contrib.rnn.BasicLSTMCell(self.t_dim, reuse=self.reuse)
            initial_state = LSTMCell.zero_state(self.batch_size, dtype=tf.float32)
            
            rnn_net = tf.nn.dynamic_rnn(cell=LSTMCell,
                                    inputs=embedded_word_ids,
                                    initial_state=initial_state,
                                    dtype=np.float32,
                                    time_major=False,
                                    scope='rnn/dynamic')

            self.rnn_net = rnn_net
            self.outputs = rnn_net[0][:, -1, :]

class cnn_encoder:
    def __init__(self, inputs, is_training=True, reuse=False):
        self.inputs = inputs
        self.is_training = is_training
        self.reuse = reuse
        self.df_dim = 64
        self.t_dim = 128
        self._build_model()

    def _build_model(self):
        df_dim = self.df_dim

        with tf.variable_scope('cnnftxt', reuse=self.reuse):
            net_h0 = Conv2d(self.inputs, 4, 4, df_dim, 2, 2, name='cnnf/h0/conv2d', activation_fn=tf.nn.leaky_relu, padding='SAME', biased=True)
            net_h1 = Conv2d(net_h0, 4, 4, df_dim*2, 2, 2, name='cnnf/h1/conv2d', padding='SAME')
            net_h1 = batch_normalization(net_h1, activation_fn=tf.nn.leaky_relu, is_training=self.is_training, name='cnnf/h1/batch_norm')

            net_h2 = Conv2d(net_h1, 4, 4, df_dim*4, 2, 2, name='cnnf/h2/conv2d', padding='SAME')
            net_h2 = batch_normalization(net_h2, activation_fn=tf.nn.leaky_relu, is_training=self.is_training, name='cnnf/h2/batch_norm')

            net_h3 = Conv2d(net_h2, 4, 4, df_dim*8, 2, 2, name='cnnf/h3/conv2d', padding='SAME')
            net_h3 = batch_normalization(net_h3, activation_fn=tf.nn.leaky_relu, is_training=self.is_training, name='cnnf/h3/batch_norm')

            net_h4 = flatten(net_h3, name='cnnf/h4/flatten')
            net_h4 = fc(net_h4, num_out=self.t_dim, name='cnnf/h4/embed', biased=False)
        
        self.outputs = net_h4