# Basic Code is taken from https://github.com/ckmarkoh/GAN-tensorflow

import tensorflow as tf
import numpy as np
from scipy.misc import imsave
import os
import shutil
import time
import random




from layers import *

img_height = 256
img_width = 256
img_layer = 3
img_size = img_height * img_width

batch_size = 1
pool_size = 50
n_units_gen = 32
n_units_disc = 64


def build_resnet_block(inputres, dim, name="resnet"):
    
    with tf.variable_scope(name):

        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "VALID","c1")
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "VALID","c2",do_relu=False)
        
        return tf.nn.relu(out_res + inputres)


def build_generator_resnet_6blocks(inputgen, name="generator"):
    with tf.variable_scope(name):
        f = 7
        ks = 3
        
        pad_input = tf.pad(inputgen,[[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
        o_c1 = general_conv2d(pad_input, n_units_gen, f, f, 1, 1, 0.02, name="c1")
        o_c2 = general_conv2d(o_c1, n_units_gen * 2, ks, ks, 2, 2, 0.02, "SAME", "c2")
        o_c3 = general_conv2d(o_c2, n_units_gen * 4, ks, ks, 2, 2, 0.02, "SAME", "c3")

        o_r1 = build_resnet_block(o_c3, n_units_gen * 4, "r1")
        o_r2 = build_resnet_block(o_r1, n_units_gen * 4, "r2")
        o_r3 = build_resnet_block(o_r2, n_units_gen * 4, "r3")
        o_r4 = build_resnet_block(o_r3, n_units_gen * 4, "r4")
        o_r5 = build_resnet_block(o_r4, n_units_gen * 4, "r5")
        o_r6 = build_resnet_block(o_r5, n_units_gen * 4, "r6")

        o_c4 = general_deconv2d(o_r6, [batch_size, 64, 64, n_units_gen * 2], n_units_gen * 2, ks, ks, 2, 2, 0.02, "SAME", "c4")
        o_c5 = general_deconv2d(o_c4, [batch_size, 128, 128, n_units_gen], n_units_gen, ks, ks, 2, 2, 0.02, "SAME", "c5")
        o_c5_pad = tf.pad(o_c5,[[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
        o_c6 = general_conv2d(o_c5_pad, img_layer, f, f, 1, 1, 0.02,"VALID","c6",do_relu=False)

        # Adding the tanh layer

        out_gen = tf.nn.tanh(o_c6,"t1")


        return out_gen

def build_generator_resnet_9blocks(inputgen, name="generator"):
    with tf.variable_scope(name):
        kernel_side = 7
        stride = 3
        
        pad_input = tf.pad(inputgen,[[0, 0], [stride, stride], [stride, stride], [0, 0]], "REFLECT")
        o_c1 = general_conv2d(pad_input, n_units_gen, kernel_side, kernel_side, 1, 1, 0.02, name="c1")
        o_c2 = general_conv2d(o_c1, n_units_gen * 2, stride, stride, 2, 2, 0.02, "SAME", "c2")
        o_c3 = general_conv2d(o_c2, n_units_gen * 4, stride, stride, 2, 2, 0.02, "SAME", "c3")

        o_r1 = build_resnet_block(o_c3, n_units_gen * 4, "r1")
        o_r2 = build_resnet_block(o_r1, n_units_gen * 4, "r2")
        o_r3 = build_resnet_block(o_r2, n_units_gen * 4, "r3")
        o_r4 = build_resnet_block(o_r3, n_units_gen * 4, "r4")
        o_r5 = build_resnet_block(o_r4, n_units_gen * 4, "r5")
        o_r6 = build_resnet_block(o_r5, n_units_gen * 4, "r6")
        o_r7 = build_resnet_block(o_r6, n_units_gen * 4, "r7")
        o_r8 = build_resnet_block(o_r7, n_units_gen * 4, "r8")
        o_r9 = build_resnet_block(o_r8, n_units_gen * 4, "r9")

        o_c4 = general_deconv2d(o_r9, [batch_size, 128, 128, n_units_gen * 2], n_units_gen * 2, stride, stride, 2, 2, 0.02, "SAME", "c4")
        o_c5 = general_deconv2d(o_c4, [batch_size, 256, 256, n_units_gen], n_units_gen, stride, stride, 2, 2, 0.02, "SAME", "c5")
        o_c6 = general_conv2d(o_c5, img_layer, kernel_side, kernel_side, 1, 1, 0.02,"SAME","c6",do_relu=False)

        # Adding the tanh layer

        out_gen = tf.nn.tanh(o_c6,"t1")


        return out_gen


def build_gen_discriminator(inputdisc, name="discriminator"):

    with tf.variable_scope(name):
        f = 4

        o_c1 = general_conv2d(inputdisc, n_units_disc, f, f, 2, 2, 0.02, "SAME", "c1", do_norm=False, relufactor=0.2)
        o_c2 = general_conv2d(o_c1, n_units_disc * 2, f, f, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = general_conv2d(o_c2, n_units_disc * 4, f, f, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = general_conv2d(o_c3, n_units_disc * 8, f, f, 1, 1, 0.02, "SAME", "c4", relufactor=0.2)
        o_c5 = general_conv2d(o_c4, 1, f, f, 1, 1, 0.02, "SAME", "c5",do_norm=False,do_relu=False)

        return o_c5


def patch_discriminator(inputdisc, name="discriminator"):

    with tf.variable_scope(name):
        f= 4

        patch_input = tf.random_crop(inputdisc,[1,70,70,3])
        o_c1 = general_conv2d(patch_input, n_units_disc, f, f, 2, 2, 0.02, "SAME", "c1", do_norm="False", relufactor=0.2)
        o_c2 = general_conv2d(o_c1, n_units_disc * 2, f, f, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = general_conv2d(o_c2, n_units_disc * 4, f, f, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = general_conv2d(o_c3, n_units_disc * 8, f, f, 2, 2, 0.02, "SAME", "c4", relufactor=0.2)
        o_c5 = general_conv2d(o_c4, 1, f, f, 1, 1, 0.02, "SAME", "c5",do_norm=False,do_relu=False)

        return o_c5