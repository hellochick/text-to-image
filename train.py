import numpy as np
import tensorflow as tf
from model import *

import pandas as pd
import os
import scipy
from scipy.io import loadmat
import re
import string
from utils import *
import random
import time
import argparse

import warnings
warnings.filterwarnings('ignore')

dictionary_path = './dictionary'
vocab = np.load(dictionary_path + '/vocab.npy')
print('there are {} vocabularies in total'.format(len(vocab)))

word2Id_dict = dict(np.load(dictionary_path + '/word2Id.npy'))
id2word_dict = dict(np.load(dictionary_path + '/id2Word.npy'))

train_images = np.load('train_images.npy', encoding='latin1')
train_captions = np.load('train_captions.npy', encoding='latin1')

assert len(train_images) == len(train_captions)

print('----example of captions[0]--------')
for caption in train_captions[0]:
    print(IdList2sent(caption))

captions_list = []
for captions in train_captions:
    assert len(captions) >= 5
    captions_list.append(captions[:5])

train_captions = np.concatenate(captions_list, axis=0) 

n_captions_train = len(train_captions)
n_captions_per_image = 5
n_images_train = len(train_images)

print('Total captions: ', n_captions_train)
print('----example of captions[0] (modified)--------')
for caption in train_captions[:5]:
    print(IdList2sent(caption))

lr = 0.0002
lr_decay = 0.5      
decay_every = 100  
beta1 = 0.5
checkpoint_dir = './checkpoint'

z_dim = 512         # Noise dimension
image_size = 64     # 64 x 64
c_dim = 3           # for rgb
batch_size = 64
ni = int(np.ceil(np.sqrt(batch_size)))

### Testing setting
sample_size = batch_size
sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)

sample_sentence = ["the flower shown has yellow anther red pistil and bright red petals."] * int(sample_size/ni) + \
                  ["this flower has petals that are yellow, white and purple and has dark lines"] * int(sample_size/ni) + \
                  ["the petals on this flower are white with a yellow center"] * int(sample_size/ni) + \
                  ["this flower has a lot of small round pink petals."] * int(sample_size/ni) + \
                  ["this flower is orange in color, and has petals that are ruffled and rounded."] * int(sample_size/ni) + \
                  ["the flower has yellow petals and the center of it is brown."] * int(sample_size/ni) + \
                  ["this flower has petals that are blue and white."] * int(sample_size/ni) +\
                  ["these white flowers have petals that start off white in color and end in a white towards the tips."] * int(sample_size/ni)
for i, sent in enumerate(sample_sentence):
    sample_sentence[i] = sent2IdList(sent)

print(sample_sentence[0])
def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))
    
def train():
    t_real_image = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name = 'real_image')
    t_wrong_image = tf.placeholder('float32', [batch_size ,image_size, image_size, 3], name = 'wrong_image')
    t_real_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='real_caption_input')
    t_wrong_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='wrong_caption_input')
    t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')

    ### Training Phase - CNN - RNN mapping
    net_cnn = cnn_encoder(t_real_image, is_training=True, reuse=False)
    x = net_cnn.outputs
    v = rnn_encoder(t_real_caption, is_training=True, reuse=False).outputs
    x_w = cnn_encoder(t_wrong_image, is_training=True, reuse=True).outputs
    v_w = rnn_encoder(t_wrong_caption, is_training=True, reuse=True).outputs

    alpha = 0.2 # margin alpha
    rnn_loss = tf.reduce_mean(tf.maximum(0., alpha - cosine_similarity(x, v) + cosine_similarity(x, v_w))) + \
                tf.reduce_mean(tf.maximum(0., alpha - cosine_similarity(x, v) + cosine_similarity(x_w, v)))

    ### Training Phase - GAN
    net_rnn = rnn_encoder(t_real_caption, is_training=False, reuse=True)
    net_fake_image = Generator(t_z, net_rnn.outputs, is_training=True, reuse=False)
            
    net_disc_fake = Discriminator(net_fake_image.outputs, net_rnn.outputs, is_training=True, reuse=False)
    disc_fake_logits = net_disc_fake.logits

    net_disc_real = Discriminator(t_real_image, net_rnn.outputs, is_training=True, reuse=True)
    disc_real_logits = net_disc_real.logits

    net_disc_mismatch = Discriminator(t_real_image, 
                                rnn_encoder(t_wrong_caption, is_training=False, reuse=True).outputs,
                                is_training=True, reuse=True)
    disc_mismatch_logits = net_disc_mismatch.logits

    d_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real_logits,     labels=tf.ones_like(disc_real_logits),      name='d1'))
    d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_mismatch_logits, labels=tf.zeros_like(disc_mismatch_logits), name='d2'))
    d_loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_logits,     labels=tf.zeros_like(disc_fake_logits),     name='d3'))
    d_loss = d_loss1 + (d_loss2 + d_loss3) * 0.5

    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_logits, labels=tf.ones_like(disc_fake_logits), name='g'))

    ### Testing Phase
    net_g = Generator(t_z, 
                    rnn_encoder(t_real_caption, is_training=False, reuse=True).outputs,
                    is_training=False, reuse=True)

    rnn_vars = [var for var in tf.trainable_variables() if 'rnn' in var.name]
    g_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]
    d_vars = [var for var in tf.trainable_variables() if 'discrim' in var.name]
    cnn_vars = [var for var in tf.trainable_variables() if 'cnn' in var.name]

    update_ops_D = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'discrim' in var.name]
    update_ops_G = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'generator' in var.name]
    update_ops_CNN = [var for var in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if 'cnn' in var.name]

    print('----------Update_ops_D--------')
    for var in update_ops_D:
        print(var.name)
    print('----------Update_ops_G--------')
    for var in update_ops_G:
        print(var.name)
    print('----------Update_ops_CNN--------')
    for var in update_ops_CNN:
        print(var.name)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr, trainable=False)

    with tf.control_dependencies(update_ops_D):
        d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)

    with tf.control_dependencies(update_ops_G):
        g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)

    with tf.control_dependencies(update_ops_CNN):
        grads, _ = tf.clip_by_global_norm(tf.gradients(rnn_loss, rnn_vars + cnn_vars), 10)
        optimizer = tf.train.AdamOptimizer(lr_v, beta1=beta1)
        rnn_optim = optimizer.apply_gradients(zip(grads, rnn_vars + cnn_vars))

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=5)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=tf.global_variables())
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('no checkpoints find.')

    n_epoch = 600
    n_batch_epoch = int(n_images_train / batch_size)
    for epoch in range(n_epoch):
        start_time = time.time()

        if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr * new_lr_decay))
            log = " ** new learning rate: %f" % (lr * new_lr_decay)
            print(log)
            
        elif epoch == 0:
            log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (lr, decay_every, lr_decay)
            print(log)

        for step in range(n_batch_epoch):
            step_time = time.time()

            ## get matched text & image
            idexs = get_random_int(min=0, max=n_captions_train-1, number=batch_size)
            b_real_caption = train_captions[idexs]
            b_real_images = train_images[np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')]

            """ check for loading right images
            save_images(b_real_images, [ni, ni], 'train_samples/train_00.png')
            for caption in b_real_caption[:8]:
                print(IdList2sent(caption))
            exit()
            """

            ## get wrong caption & wrong image
            idexs = get_random_int(min=0, max=n_captions_train-1, number=batch_size)
            b_wrong_caption = train_captions[idexs]
            idexs2 = get_random_int(min=0, max=n_images_train-1, number=batch_size)
            b_wrong_images = train_images[idexs2]

            ## get noise
            b_z = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, z_dim)).astype(np.float32)

            b_real_images = threading_data(b_real_images, prepro_img, mode='train')   # [0, 255] --> [-1, 1] + augmentation
            b_wrong_images = threading_data(b_wrong_images, prepro_img, mode='train')

            ## update RNN
            if epoch < 80:
                errRNN, _ = sess.run([rnn_loss, rnn_optim], feed_dict={
                                                t_real_image : b_real_images,
                                                t_wrong_image : b_wrong_images,
                                                t_real_caption : b_real_caption,
                                                t_wrong_caption : b_wrong_caption})
            else:
                errRNN = 0

            ## updates D
            errD, _ = sess.run([d_loss, d_optim], feed_dict={
                            t_real_image : b_real_images,
                            t_wrong_caption : b_wrong_caption,
                            t_real_caption : b_real_caption,
                            t_z : b_z})
            ## updates G
            errG, _ = sess.run([g_loss, g_optim], feed_dict={
                            t_real_caption : b_real_caption,
                            t_z : b_z})

            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4fs, d_loss: %.8f, g_loss: %.8f, rnn_loss: %.8f" \
                        % (epoch, n_epoch, step, n_batch_epoch, time.time() - step_time, errD, errG, errRNN))
        
        if (epoch + 1) % 1 == 0:
            print(" ** Epoch %d took %fs" % (epoch, time.time()-start_time))
            img_gen, rnn_out = sess.run([net_g.outputs, net_rnn.outputs], feed_dict={
                                        t_real_caption : sample_sentence,
                                        t_z : sample_seed})

            save_images(img_gen, [ni, ni], 'train_samples/train_{:02d}.png'.format(epoch))

        if (epoch != 0) and (epoch % 10) == 0:
            save(saver, sess, checkpoint_dir, epoch)
            print("[*] Save checkpoints SUCCESS!")

testData = os.path.join('dataset', 'testData.pkl')
def test():
    data = pd.read_pickle(testData)
    captions = data['Captions'].values
    caption = []
    for i in range(len(captions)):
        caption.append(captions[i])
    caption = np.asarray(caption)
    index = data['ID'].values
    index = np.asarray(index)

    t_real_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='real_caption_input')
    t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')

    net_g = Generator(t_z, rnn_encoder(t_real_caption, is_training=False, reuse=False).outputs,
                    is_training=False, reuse=False)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=tf.global_variables())
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('no checkpoints find.')

    n_caption_test = len(caption)
    n_batch_epoch = int(n_caption_test / batch_size) + 1

    ## repeat
    caption = np.tile(caption, (2, 1))
    index = np.tile(index, 2)

    assert index[0] == index[n_caption_test]

    for i in range(n_batch_epoch):
        test_cap = caption[i*batch_size: (i+1)*batch_size]

        z = np.random.normal(loc=0.0, scale=1.0, size=(batch_size, z_dim)).astype(np.float32)
        gen = sess.run(net_g.outputs, feed_dict={t_real_caption: test_cap, t_z: z})
        for j in range(batch_size):
            save_images(np.expand_dims(gen[j], axis=0), [1, 1], 'inference/inference_{:04d}.png'.format(index[i*batch_size + j]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Text-to-image")
    parser.add_argument("--mode", type=str, default='train',
                        help="train/test")

    args = parser.parse_args()
    if args.mode == 'train':
        print('In training mode.')
        train()
    elif args.mode == 'test':
        print('In testing mode.')
        test()
    