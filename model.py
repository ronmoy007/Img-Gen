#! /usr/bin/python
# -*- coding: utf8 -*-



import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import tensorlayer as tl
from tensorlayer.layers import *
import os

batch_size = 64

z_dim = 512         # Noise dimension
image_size = 64     # 64 x 64
c_dim = 3           # for rgb


## for text-to-image mapping ===================================================
t_dim = 128         # text feature dimension
rnn_hidden_size = t_dim
vocab_size = 8000
word_embedding_size = 256
keep_prob = 1.0

def Generar_Embedding_RNN(input_seqs, is_train=True, reuse=False, return_embed=False):
    """ txt --> t_dim """
    w_init = tf.random_normal_initializer(stddev=0.02)
    if tf.__version__ <= '0.12.1':
        LSTMCell = tf.nn.rnn_cell.LSTMCell
    else:
        LSTMCell = tf.contrib.rnn.BasicLSTMCell
    with tf.variable_scope("rnnftxt", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        capa_Input = EmbeddingInputlayer(
                     inputs = input_seqs,
                     vocabulary_size = vocab_size,
                     embedding_size = word_embedding_size,
                     E_init = w_init,
                     name = 'rnn/wordembed')
        capa_Din = DynamicRNNLayer(capa_Input,
                     cell_fn = LSTMCell,
                     cell_init_args = {'state_is_tuple' : True, 'reuse': reuse},  # for TF1.1, TF1.2 dont need to set reuse
                     n_hidden = rnn_hidden_size,
                     dropout = (keep_prob if is_train else None),
                     initializer = w_init,
                     sequence_length = tl.layers.retrieve_seq_length_op2(input_seqs),
                     return_last = True,
                     name = 'rnn/dynamic')
        return capa_Din

def Generar_Encoder_CNN(tensor_Imagenes, is_train=True, reuse=False, name='cnnftxt', return_h3=False):
    """ 64x64 --> t_dim, for text-image mapping """
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64

    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(True)

        capa_Input = InputLayer(tensor_Imagenes, name='/in')
        capa_h0 = Conv2d(capa_Input, df_dim, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=w_init, name='cnnf/h0/conv2d')

        capa_h1 = Conv2d(capa_h0, df_dim*2, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='cnnf/h1/conv2d')
        capa_h1 = BatchNormLayer(capa_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='cnnf/h1/batch_norm')

        capa_h2 = Conv2d(capa_h1, df_dim*4, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='cnnf/h2/conv2d')
        capa_h2 = BatchNormLayer(capa_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='cnnf/h2/batch_norm')

        capa_h3 = Conv2d(capa_h2, df_dim*8, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='cnnf/h3/conv2d')
        capa_h3 = BatchNormLayer(capa_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='cnnf/h3/batch_norm')

        capa_h4 = FlattenLayer(capa_h3, name='cnnf/h4/flatten')
        capa_h4 = DenseLayer(capa_h4, n_units= (z_dim if name == 'z_encoder' else t_dim),
                act=tf.identity,
                W_init = w_init, b_init = None, name='cnnf/h4/embed')
    if return_h3:
        return capa_h4, capa_h3
    else:
        return capa_h4

## default g1, d1 ==============================================================
def Generador_ResNet(input_z, t_txt=None, is_train=True, reuse=False, batch_size=batch_size):
    """ z + (txt) --> 64x64 """

    s = image_size
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
    gf_dim = 128

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        capa_Input = InputLayer(input_z, name='g_inputz')

        if t_txt is not None:
            capa_txt = InputLayer(t_txt, name='g_input_txt')
            capa_txt = DenseLayer(capa_txt, n_units=t_dim,
                act=lambda x: tl.act.lrelu(x, 0.2), W_init=w_init, name='g_reduce_text/dense')
            capa_Input = ConcatLayer([capa_Input, capa_txt], concat_dim=1, name='g_concat_z_txt')

        capa_h0 = DenseLayer(capa_Input, gf_dim*8*s16*s16, act=tf.identity,
                W_init=w_init, b_init=None, name='g_h0/dense')
        capa_h0 = BatchNormLayer(capa_h0,  #act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h0/batch_norm')
        capa_h0 = ReshapeLayer(capa_h0, [-1, s16, s16, gf_dim*8], name='g_h0/reshape')

        capa = Conv2d(capa_h0, gf_dim*2, (1, 1), (1, 1),
                padding='VALID', act=None, W_init=w_init, b_init=None, name='g_h1_res/conv2d')
        capa = BatchNormLayer(capa, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h1_res/batch_norm')
        capa = Conv2d(capa, gf_dim*2, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h1_res/conv2d2')
        capa = BatchNormLayer(capa, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h1_res/batch_norm2')
        capa = Conv2d(capa, gf_dim*8, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h1_res/conv2d3')
        capa = BatchNormLayer(capa, # act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h1_res/batch_norm3')
        capa_h1 = ElementwiseLayer(layer=[capa_h0, capa], combine_fn=tf.add, name='g_h1_res/add')
        capa_h1.outputs = tf.nn.relu(capa_h1.outputs)

        capa_h2 = UpSampling2dLayer(capa_h1, size=[s8, s8], is_scale=False, method=1,
                align_corners=False, name='g_h2/upsample2d')
        capa_h2 = Conv2d(capa_h2, gf_dim*4, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h2/conv2d')
        capa_h2 = BatchNormLayer(capa_h2,# act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h2/batch_norm')

        capa = Conv2d(capa_h2, gf_dim, (1, 1), (1, 1),
                padding='VALID', act=None, W_init=w_init, b_init=None, name='g_h3_res/conv2d')
        capa = BatchNormLayer(capa, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h3_res/batch_norm')
        capa = Conv2d(capa, gf_dim, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h3_res/conv2d2')
        capa = BatchNormLayer(capa, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h3_res/batch_norm2')
        capa = Conv2d(capa, gf_dim*4, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h3_res/conv2d3')
        capa = BatchNormLayer(capa, #act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h3_res/batch_norm3')
        capa_h3 = ElementwiseLayer(layer=[capa_h2, capa], combine_fn=tf.add, name='g_h3/add')
        capa_h3.outputs = tf.nn.relu(capa_h3.outputs)

        capa_h4 = UpSampling2dLayer(capa_h3, size=[s4, s4], is_scale=False, method=1,
                align_corners=False, name='g_h4/upsample2d')
        capa_h4 = Conv2d(capa_h4, gf_dim*2, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h4/conv2d')
        capa_h4 = BatchNormLayer(capa_h4, act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h4/batch_norm')

        capa_h5 = UpSampling2dLayer(capa_h4, size=[s2, s2], is_scale=False, method=1,
                align_corners=False, name='g_h5/upsample2d')
        capa_h5 = Conv2d(capa_h5, gf_dim, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h5/conv2d')
        capa_h5 = BatchNormLayer(capa_h5, act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h5/batch_norm')

        capa_Output = UpSampling2dLayer(capa_h5, size=[s, s], is_scale=False, method=1,
                align_corners=False, name='g_ho/upsample2d')
        capa_Output = Conv2d(capa_Output, c_dim, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, name='g_ho/conv2d')
        logits = capa_Output.outputs
        capa_Output.outputs = tf.nn.tanh(capa_Output.outputs)
    return capa_Output, logits

def Discriminador_ResNet(input_images, t_txt=None, is_train=True, reuse=False):
    """ 64x64 + (txt) --> real/fake """

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    df_dim = 64  # 64 for flower, 196 for MSCOCO
    s = 64 # output image size [64]
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        capa_Input = InputLayer(input_images, name='d_input/images')
        capa_h0 = Conv2d(capa_Input, df_dim, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=w_init, name='d_h0/conv2d')

        capa_h1 = Conv2d(capa_h0, df_dim*2, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h1/conv2d')
        capa_h1 = BatchNormLayer(capa_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h1/batchnorm')
        capa_h2 = Conv2d(capa_h1, df_dim*4, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h2/conv2d')
        capa_h2 = BatchNormLayer(capa_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h2/batchnorm')
        capa_h3 = Conv2d(capa_h2, df_dim*8, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h3/conv2d')
        capa_h3 = BatchNormLayer(capa_h3, #act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h3/batchnorm')

        capa = Conv2d(capa_h3, df_dim*2, (1, 1), (1, 1), act=None,
                padding='VALID', W_init=w_init, b_init=None, name='d_h4_res/conv2d')
        capa = BatchNormLayer(capa, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h4_res/batchnorm')
        capa = Conv2d(capa, df_dim*2, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h4_res/conv2d2')
        capa = BatchNormLayer(capa, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h4_res/batchnorm2')
        capa = Conv2d(capa, df_dim*8, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h4_res/conv2d3')
        capa = BatchNormLayer(capa, #act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h4_res/batchnorm3')
        capa_h4 = ElementwiseLayer(layer=[capa_h3, capa], combine_fn=tf.add, name='d_h4/add')
        capa_h4.outputs = tl.act.lrelu(capa_h4.outputs, 0.2)

        if t_txt is not None:
            capa_txt = InputLayer(t_txt, name='d_input_txt')
            capa_txt = DenseLayer(capa_txt, n_units=t_dim,
                   act=lambda x: tl.act.lrelu(x, 0.2),
                   W_init=w_init, name='d_reduce_txt/dense')
            capa_txt = ExpandDimsLayer(capa_txt, 1, name='d_txt/expanddim1')
            capa_txt = ExpandDimsLayer(capa_txt, 1, name='d_txt/expanddim2')
            capa_txt = TileLayer(capa_txt, [1, 4, 4, 1], name='d_txt/tile')
            capa_h4_concat = ConcatLayer([capa_h4, capa_txt], concat_dim=3, name='d_h3_concat')
            # 243 (ndf*8 + 128 or 256) x 4 x 4
            capa_h4 = Conv2d(capa_h4_concat, df_dim*8, (1, 1), (1, 1),
                    padding='VALID', W_init=w_init, b_init=None, name='d_h3/conv2d_2')
            capa_h4 = BatchNormLayer(capa_h4, act=lambda x: tl.act.lrelu(x, 0.2),
                    is_train=is_train, gamma_init=gamma_init, name='d_h3/batch_norm_2')

        capa_Output = Conv2d(capa_h4, 1, (s16, s16), (s16, s16), padding='VALID', W_init=w_init, name='d_ho/conv2d')

        logits = capa_Output.outputs
        capa_Output.outputs = tf.nn.sigmoid(capa_Output.outputs)
    return capa_Output, logits
