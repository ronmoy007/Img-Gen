import tensorflow as tf
import os
import random
import scipy
import scipy.misc
import numpy as np
import re
import string

def load_and_assign_npz(sess=None, name="", model=None):
    assert model is not None
    assert sess is not None
    if not os.path.exists(name):
        print("[!] Loading {} model failed!".format(name))
        return False
    else:
        params = tl.files.load_npz(name=name)
        tl.files.assign_params(sess, params, model)
        print("[*] Loading {} model SUCCESS!".format(name))
        return model

#prepro ?
def get_random_int(min=0, max=10, number=5):
    """Return a list of random integer by the given range and quantity.

    Examples
    ---------
    >>> r = get_random_int(min=0, max=10, number=5)
    ... [10, 2, 3, 3, 7]
    """
    return [random.randint(min,max) for p in range(0,number)]

def PrepararFrase(line):
    prep_line = re.sub('[%s]' % re.escape(string.punctuation), ' ', line.rstrip())
    prep_line = prep_line.replace('-', ' ')
    return prep_line


## Save images
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def save_images(images, size, image_path):
    return imsave(images, size, image_path)

from tensorlayer.prepro import *
def prepro_img(x, mode=None):
    if mode=='train':
    # rescale [0, 255] --> (-1, 1), random flip, crop, rotate
    #   paper 5.1: During mini-batch selection for training we randomly pick
    #   an image view (e.g. crop, flip) of the image and one of the captions
    # flip, rotate, crop, resize : https://github.com/reedscot/icml2016/blob/master/data/donkey_folder_coco.lua
    # flip : https://github.com/paarthneekhara/text-to-image/blob/master/Utils/image_processing.py
        x = flip_axis(x, axis=1, is_random=True)
        x = rotation(x, rg=16, is_random=True, fill_mode='nearest')
        x = imresize(x, size=[64+15, 64+15], interp='bilinear', mode=None)
        x = crop(x, wrg=64, hrg=64, is_random=True)
        x = x / (255. / 2.)
        x = x - 1.
        # x = x * 0.9999
    elif mode=='train_stackGAN':
        x = flip_axis(x, axis=1, is_random=True)
        x = rotation(x, rg=16, is_random=True, fill_mode='nearest')
        x = imresize(x, size=[316, 316], interp='bilinear', mode=None)
        x = crop(x, wrg=256, hrg=256, is_random=True)
        x = x / (255. / 2.)
        x = x - 1.
    elif mode=='rescale':
        x = (x + 1.) / 2.
    elif mode=='debug':
        x = flip_axis(x, axis=1, is_random=False)
        x = x / 255.
    elif mode=='translation':
        x = x / (255. / 2.)
        x = x - 1.
    else:
        raise Exception("Not support : %s" % mode)
    return x
