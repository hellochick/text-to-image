import tensorflow as tf
import os
import random
import scipy
import scipy.misc
import numpy as np
import re
import string
import threading
import scipy.ndimage as ndi
from skimage import transform
from skimage import exposure
import skimage

dictionary_path = './dictionary'
word2Id_dict = dict(np.load(dictionary_path + '/word2Id.npy'))
id2word_dict = dict(np.load(dictionary_path + '/id2Word.npy'))

def sent2IdList(line, MAX_SEQ_LENGTH=20):
    MAX_SEQ_LIMIT = MAX_SEQ_LENGTH
    padding = 0
    prep_line = re.sub('[%s]' % re.escape(string.punctuation), ' ', line.rstrip())
    prep_line = prep_line.replace('-', ' ')
    prep_line = prep_line.replace('-', ' ')
    prep_line = prep_line.replace('  ', ' ')
    prep_line = prep_line.replace('.', '')
    tokens = prep_line.split(' ')
    tokens = [
        tokens[i] for i in range(len(tokens))
        if tokens[i] != ' ' and tokens[i] != ''
    ]
    l = len(tokens)
    padding = MAX_SEQ_LIMIT - l
    for i in range(padding):
        tokens.append('<PAD>')
    
    line = [
        word2Id_dict[tokens[k]]
        if tokens[k] in word2Id_dict else word2Id_dict['<RARE>']
        for k in range(len(tokens))
    ]

    return line

def IdList2sent(caption):
    sentence = []
    for ID in caption:
        if ID != word2Id_dict['<PAD>']:
            sentence.append(id2word_dict[ID])

    return sentence

def get_random_int(min=0, max=10, number=5):
    """Return a list of random integer by the given range and quantity.
    Examples
    ---------
    >>> r = get_random_int(min=0, max=10, number=5)
    ... [10, 2, 3, 3, 7]
    """
    return [random.randint(min,max) for p in range(0,number)]

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

# Data Augmentation reference: https://github.com/tensorlayer/tensorlayer/tree/master/tensorlayer
def threading_data(data=None, fn=None, **kwargs):
    def apply_fn(results, i, data, kwargs):
        results[i] = fn(data, **kwargs)

    ## start multi-threaded reading.
    results = [None] * len(data) ## preallocate result list
    threads = []
    for i in range(len(data)):
        t = threading.Thread(
                        name='threading_and_return',
                        target=apply_fn,
                        args=(results, i, data[i], kwargs)
                        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    return np.asarray(results)

def apply_transform(x, transform_matrix, channel_index=2, fill_mode='nearest', cval=0., order=1):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=order, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def rotation(x, rg=20, is_random=False, row_index=0, col_index=1, channel_index=2,
                    fill_mode='nearest', cval=0.):
    if is_random:
        theta = np.pi / 180 * np.random.uniform(-rg, rg)
    else:
        theta = np.pi /180 * rg
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x

def crop(x, wrg, hrg, is_random=False, row_index=0, col_index=1, channel_index=2):
    h, w = x.shape[row_index], x.shape[col_index]
    assert (h > hrg) and (w > wrg), "The size of cropping should smaller than the original image"
    if is_random:
        h_offset = int(np.random.uniform(0, h-hrg) -1)
        w_offset = int(np.random.uniform(0, w-wrg) -1)
        return x[h_offset: hrg+h_offset ,w_offset: wrg+w_offset]
    else:   # central crop
        h_offset = int(np.floor((h - hrg)/2.))
        w_offset = int(np.floor((w - wrg)/2.))
        h_end = h_offset + hrg
        w_end = w_offset + wrg
        return x[h_offset: h_end, w_offset: w_end]

def flip_axis(x, axis, is_random=False):
    if is_random:
        factor = np.random.uniform(-1, 1)
        if factor > 0:
            x = np.asarray(x).swapaxes(axis, 0)
            x = x[::-1, ...]
            x = x.swapaxes(0, axis)
            return x
        else:
            return x
    else:
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x

def imresize(x, size=[100, 100], interp='bilinear', mode=None):
    if x.shape[-1] == 1:
        # greyscale
        x = scipy.misc.imresize(x[:,:,0], size, interp=interp, mode=mode)
        return x[:, :, np.newaxis]
    elif x.shape[-1] == 3:
        # rgb, bgr ..
        return scipy.misc.imresize(x, size, interp=interp, mode=mode)
    else:
        raise Exception("Unsupported channel %d" % x.shape[-1])
        
def prepro_img(x, mode=None):
    # rescale [0, 255] --> (-1, 1), random flip, crop, rotate

    if mode=='train':
        x = flip_axis(x, axis=1, is_random=True)
        x = rotation(x, rg=16, is_random=True, fill_mode='nearest')
        x = imresize(x, size=[64+15, 64+15], interp='bilinear', mode=None)
        x = crop(x, wrg=64, hrg=64, is_random=True)
        x = x / (255. / 2.)
        x = x - 1.
        # x = x * 0.9999

    return x

def cosine_similarity(v1, v2):
    cost = tf.reduce_sum(tf.multiply(v1, v2), 1) / (tf.sqrt(tf.reduce_sum(tf.multiply(v1, v1), 1)) * tf.sqrt(tf.reduce_sum(tf.multiply(v2, v2), 1)))
    return cost

def combine_and_save_image_sets(image_sets, directory):
    for i in range(len(image_sets[0])):
        combined_image = []
        for set_no in range(len(image_sets)):
            combined_image.append( image_sets[set_no][i] )
            combined_image.append( np.zeros((image_sets[set_no][i].shape[0], 5, 3)) )
        combined_image = np.concatenate( combined_image, axis = 1 )

        scipy.misc.imsave( os.path.join( directory,  'combined_{}.jpg'.format(i) ), combined_image)