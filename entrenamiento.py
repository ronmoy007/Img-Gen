#! /usr/bin/python
# -*- coding: utf8 -*-
import sys
sys.path.append('/Users/Marco/miniconda3/envs/dl-proj3/lib/python3.7/site-packages')

""" GAN-CLS """
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.prepro import *
from tensorlayer.cost import *
import numpy as np
import scipy
from scipy.io import loadmat
import time, os, re, nltk
import pickle

from utileria import *
from model import *
import model

import os

def main_train():

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    ####################### Carga de pickles #######################
    print("---Carga de archivos pickle...")
    with open("_vocab.pickle", 'rb') as f:
        tensor_Vocabulario = pickle.load(f)
    with open("_image_train.pickle", 'rb') as f:
        pckl_images_train = pickle.load(f)
    with open("_image_test.pickle", 'rb') as f:
        pckl_images_test = pickle.load(f)
    with open("_n.pickle", 'rb') as f:
        nbr_CantFrases_Train, nbr_CantFrases_Test, nbr_Cant_Frases_X_Imag, nbr_CantImag_Train, nbr_CantImag_Test = pickle.load(f)
    with open("_caption.pickle", 'rb') as f:
        arr_Id_Frases_train, arr_Id_Frases_test = pickle.load(f)

    arr_Imagenes_Train = np.array(pckl_images_train)
    arr_Imagenes_Test = np.array(pckl_images_test)

    ni = int(np.ceil(np.sqrt(batch_size)))

    ####################### Generación de carpetas para muestras y checkpoints #######################
    tl.files.exists_or_mkdir("samples/step1_gan-cls")
    tl.files.exists_or_mkdir("samples/step_pretrain_encoder")
    tl.files.exists_or_mkdir("checkpoint")
    str_DirCheckPoints = "checkpoint"


    ####################### Declaración de tensores "placeholder" #######################
    tensor_Imagenes_Reales = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name = 'real_image')
    tensor_Imagenes_Falsas = tf.placeholder('float32', [batch_size ,image_size, image_size, 3], name = 'wrong_image')
    tensor_Frases_Reales = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='real_caption_input')
    tensor_Frases_Falsas = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='wrong_caption_input')
    tensor_Ruido_Aleatorio = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')

    ####################### Declaración de tensores badados en redes #######################
    cnn_Encoder = Generar_Encoder_CNN(tensor_Imagenes_Reales, is_train=True, reuse=False)
    tensor_Enc_ImgReales = cnn_Encoder.outputs
    tensor_Emb_FrsReales = Generar_Embedding_RNN(tensor_Frases_Reales, is_train=True, reuse=False).outputs
    tensor_Enc_ImgFalsas = Generar_Encoder_CNN(tensor_Imagenes_Falsas, is_train=True, reuse=True).outputs
    tensor_Emb_FrsFalsas = Generar_Embedding_RNN(tensor_Frases_Falsas, is_train=True, reuse=True).outputs

    ####################### Generación de las redes que se usarán #######################
    nbr_Alpha = 0.2
    nbr_Embedding_Perdida = tf.reduce_mean(tf.maximum(0., nbr_Alpha - cosine_similarity(tensor_Enc_ImgReales, tensor_Emb_FrsReales) + cosine_similarity(tensor_Enc_ImgReales, tensor_Emb_FrsFalsas))) + \
                tf.reduce_mean(tf.maximum(0., nbr_Alpha - cosine_similarity(tensor_Enc_ImgReales, tensor_Emb_FrsReales) + cosine_similarity(tensor_Enc_ImgFalsas, tensor_Emb_FrsReales)))

    rnn_Embedding = Generar_Embedding_RNN(tensor_Frases_Reales, is_train=False, reuse=True)

    rsnt_Imagenes_Falsas, _ = model.Generador_ResNet(tensor_Ruido_Aleatorio,
                    rnn_Embedding.outputs,
                    is_train=True, reuse=False, batch_size=batch_size)

    rsnt_Discriminador, disc_fake_image_logits = model.Discriminador_ResNet(
                    rsnt_Imagenes_Falsas.outputs, rnn_Embedding.outputs, is_train=True, reuse=False)

    _, disc_real_image_logits = model.Discriminador_ResNet(
                    tensor_Imagenes_Reales, rnn_Embedding.outputs, is_train=True, reuse=True)

    _, disc_mismatch_logits = model.Discriminador_ResNet(
                    tensor_Imagenes_Reales,
                    Generar_Embedding_RNN(tensor_Frases_Falsas, is_train=False, reuse=True).outputs,
                    is_train=True, reuse=True)

    rnn_Embeding_Aux = Generar_Embedding_RNN(tensor_Frases_Reales, is_train=False, reuse=True)
    rsnt_Generador, _ = model.Generador_ResNet(tensor_Ruido_Aleatorio,
                    rnn_Embeding_Aux.outputs,
                    is_train=False, reuse=True, batch_size=batch_size)

    ####################### Declaración de variables de pérdida #######################
    d_loss1 = tl.cost.sigmoid_cross_entropy(disc_real_image_logits, tf.ones_like(disc_real_image_logits), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(disc_mismatch_logits,  tf.zeros_like(disc_mismatch_logits), name='d2')
    d_loss3 = tl.cost.sigmoid_cross_entropy(disc_fake_image_logits, tf.zeros_like(disc_fake_image_logits), name='d3')
    d_loss = d_loss1 + (d_loss2 + d_loss3) * 0.5
    g_loss = tl.cost.sigmoid_cross_entropy(disc_fake_image_logits, tf.ones_like(disc_fake_image_logits), name='g')

    ####################### Variables de redes #######################
    lr = 0.0002
    nbr_Decay_lr = 0.5
    nbr_Decay_Cada = 100
    nbr_Beta1 = 0.5

    cnn_vars = tl.layers.get_variables_with_name('cnn', True, True)
    rnn_vars = tl.layers.get_variables_with_name('rnn', True, True)
    d_vars = tl.layers.get_variables_with_name('discriminator', True, True)
    g_vars = tl.layers.get_variables_with_name('generator', True, True)

    ####################### Declaración de optimizadores #######################
    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr, trainable=False)
    optm_Discriminador = tf.train.AdamOptimizer(lr_v, beta1=nbr_Beta1).minimize(d_loss, var_list=d_vars )
    optm_Generador = tf.train.AdamOptimizer(lr_v, beta1=nbr_Beta1).minimize(g_loss, var_list=g_vars )
    grads, _ = tf.clip_by_global_norm(tf.gradients(nbr_Embedding_Perdida, rnn_vars + cnn_vars), 10)
    optimizador_Embedding = tf.train.AdamOptimizer(lr_v, beta1=nbr_Beta1)
    oper_Gradientes = optimizador_Embedding.apply_gradients(zip(grads, rnn_vars + cnn_vars))

    ###============================ TRAINING ====================================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tl.layers.initialize_global_variables(sess)

    ####################### Rutas de checkpoints #######################
    str_Rnn_Embedding_NombreArch = os.path.join(str_DirCheckPoints, 'rnn_Embedding.npz')
    str_Cnn_Encoder_NombreArch = os.path.join(str_DirCheckPoints, 'cnn_Encoder.npz')
    str_Rsnt_Generador = os.path.join(str_DirCheckPoints, 'rsnt_Generador.npz')
    str_Rsnt_Discriminador = os.path.join(str_DirCheckPoints, 'rsnt_Discriminador.npz')

    ####################### Carga de checkpoints #######################
    #load_and_assign_npz(sess=sess, name=str_Rnn_Embedding_NombreArch, model=rnn_Embedding)
    #load_and_assign_npz(sess=sess, name=str_Cnn_Encoder_NombreArch, model=cnn_Encoder)
    #load_and_assign_npz(sess=sess, name=str_Rsnt_Generador, model=rsnt_Generador)
    #load_and_assign_npz(sess=sess, name=str_Rsnt_Discriminador, model=rsnt_Discriminador)

    ## seed for generation, z and sentence ids
    sample_size = batch_size
    sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)
    n = int(sample_size / ni)
    list_Oraciones = ["A man running."] * n + \
                      ["a dog playing in the grass."] * n + \
                      ["the woman standing by the tree."] * n + \
                      ["a boy with a ball."] * n + \
                      ["a girl at the stairs."] * n + \
                      ["swimming in the pool."] * n + \
                      ["a boy and a girl by the window."] * n +\
                      ["a woman walking with her little dog."] * n

    for i, str_Oracion in enumerate(list_Oraciones):
        print("seed: %s" % str_Oracion)
        str_Oracion = PrepararFrase(str_Oracion)
        list_Oraciones[i] = [tensor_Vocabulario.word_to_id(word) for word in nltk.tokenize.word_tokenize(str_Oracion)] + [tensor_Vocabulario.end_id]
    arr_Oraciones = tl.prepro.pad_sequences(list_Oraciones, padding='post')

    ####################### Variables de entrenamiento #######################
    nbr_Cant_Epocas = 2 # Jajaja
    nbr_Freq_Print = 1
    nbr_Cant_BatchXEpoca = int(nbr_CantImag_Train / batch_size)
    nbr_Cant_BatchXEpoca = 2 # Jajaja

    ####################### Entrenamiento por épocas #######################
    for epoch in range(0, nbr_Cant_Epocas+1):
        start_time = time.time()

        if epoch !=0 and (epoch % nbr_Decay_Cada == 0):
            new_lr_decay = nbr_Decay_lr ** (epoch // nbr_Decay_Cada)
            sess.run(tf.assign(lr_v, lr * new_lr_decay))
            log = " ** new learning rate: %f" % (lr * new_lr_decay)
            print(log)
            # logging.debug(log)
        elif epoch == 0:
            log = " ** init lr: %f  nbr_Decay_Cada: %d, nbr_Decay_lr: %f" % (lr, nbr_Decay_Cada, nbr_Decay_lr)
            print(log)

        ####################### train s tep X época #######################
        for step in range(nbr_Cant_BatchXEpoca):
            step_time = time.time()

            ## get matched text
            nbr_Index = get_random_int(min=0, max=nbr_CantFrases_Train-1, number=batch_size)
            arr_Frases_Reales = arr_Id_Frases_train[nbr_Index]
            arr_Frases_Reales = tl.prepro.pad_sequences(arr_Frases_Reales, padding='post')

            ## get real image
            arr_Imagenes_Reales = arr_Imagenes_Train[np.floor(np.asarray(nbr_Index).astype('float')/nbr_Cant_Frases_X_Imag).astype('int')]

            ## get wrong caption
            nbr_Index = get_random_int(min=0, max=nbr_CantFrases_Train-1, number=batch_size)
            arr_Frases_Falsas = arr_Id_Frases_train[nbr_Index]
            arr_Frases_Falsas = tl.prepro.pad_sequences(arr_Frases_Falsas, padding='post')

            ## get wrong image
            nbr_Index2 = get_random_int(min=0, max=nbr_CantImag_Train-1, number=batch_size)
            arr_Imagenes_Falsas = arr_Imagenes_Train[nbr_Index2]

            ## get noise
            nbr_Ruido_Aleatorio = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)

            arr_Imagenes_Reales = threading_data(arr_Imagenes_Reales, prepro_img, mode='train')
            arr_Imagenes_Falsas = threading_data(arr_Imagenes_Falsas, prepro_img, mode='train')

            ## updates text-to-image mapping
            if epoch < 50:
                nbr_Error_Embeding, _ = sess.run([nbr_Embedding_Perdida, oper_Gradientes], feed_dict={
                                                tensor_Imagenes_Reales : arr_Imagenes_Reales,
                                                tensor_Imagenes_Falsas : arr_Imagenes_Falsas,
                                                tensor_Frases_Reales : arr_Frases_Reales,
                                                tensor_Frases_Falsas : arr_Frases_Falsas})
            else:
                nbr_Error_Embeding = 0

            ## updates D
            nbr_Error_Discriminator, _ = sess.run([d_loss, optm_Discriminador], feed_dict={
                            tensor_Imagenes_Reales : arr_Imagenes_Reales,
                            # tensor_Imagenes_Falsas : arr_Imagenes_Falsas,
                            tensor_Frases_Falsas : arr_Frases_Falsas,
                            tensor_Frases_Reales : arr_Frases_Reales,
                            tensor_Ruido_Aleatorio : nbr_Ruido_Aleatorio})
            ## updates G
            nbr_Error_Generator, _ = sess.run([g_loss, optm_Generador], feed_dict={
                            tensor_Frases_Reales : arr_Frases_Reales,
                            tensor_Ruido_Aleatorio : nbr_Ruido_Aleatorio})

            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4fs, d_loss: %.8f, g_loss: %.8f, nbr_Embedding_Perdida: %.8f" \
                        % (epoch, nbr_Cant_Epocas, step, nbr_Cant_BatchXEpoca, time.time() - step_time, nbr_Error_Discriminator, nbr_Error_Generator, nbr_Error_Embeding))

        if (epoch + 1) % nbr_Freq_Print == 0:
            print(" ** Epoch %d took %fs" % (epoch, time.time()-start_time))
            img_gen, rnn_out = sess.run([rsnt_Generador.outputs, rnn_Embedding.outputs], feed_dict={
                                        tensor_Frases_Reales : arr_Oraciones,
                                        tensor_Ruido_Aleatorio : sample_seed})

            save_images(img_gen, [8, 8], 'samples/step1_gan-cls/train_{:02d}.png'.format(epoch))


        ####################### Guardamos el modelo #######################
        if (epoch != 0) and (epoch % 2) == 0:
            tl.files.save_npz(cnn_Encoder.all_params, name=str_Cnn_Encoder_NombreArch, sess=sess)
            tl.files.save_npz(rnn_Embedding.all_params, name=str_Rnn_Embedding_NombreArch, sess=sess)
            tl.files.save_npz(rsnt_Generador.all_params, name=str_Rsnt_Generador, sess=sess)
            tl.files.save_npz(rsnt_Discriminador.all_params, name=str_Rsnt_Discriminador, sess=sess)
            print("[*] Save checkpoints SUCCESS!")

        if (epoch != 0) and (epoch % 100) == 0:
            tl.files.save_npz(cnn_Encoder.all_params, name=str_Cnn_Encoder_NombreArch+str(epoch), sess=sess)
            tl.files.save_npz(rnn_Embedding.all_params, name=str_Rnn_Embedding_NombreArch+str(epoch), sess=sess)
            tl.files.save_npz(rsnt_Generador.all_params, name=str_Rsnt_Generador+str(epoch), sess=sess)
            tl.files.save_npz(rsnt_Discriminador.all_params, name=str_Rsnt_Discriminador+str(epoch), sess=sess)


if __name__ == '__main__':

    main_train()
