import tensorflow as tf

from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, BatchNormalization, \
    Add, Activation
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.activations import selu
from tensorflow.python.keras import backend as K
import memory_saving_gradients

# tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory

img_height = 256
img_width = 256
img_layer = 3
img_size = img_height * img_width

batch_size = 4
pool_size = 50
n_units_gen = 32
n_units_disc = 64

K.set_learning_phase(True)


def build_resnet_block(inputres, dim, name="resnet"):
    
    with tf.variable_scope(name):

        out_res = Conv2D(dim, kernel_size= (3,3), padding='same', activation='selu') (inputres)
        out_res = Conv2D(dim, kernel_size=(3, 3), padding='same') (out_res)
        out_res = Add() ([out_res, inputres])
        out_res = Activation(activation='selu')(out_res)
        out_res = BatchNormalization()(out_res)

        return out_res


def build_generator_resnet_n_blocks(input_shape, n, name="generator"):
    with tf.variable_scope(name):
        ks = 3
        
        inp = Input(input_shape)

        img = Conv2D(n_units_gen, kernel_size=[ks, ks], activation='selu') (inp)
        img = BatchNormalization()(img)
        img = Conv2D(n_units_gen*2, kernel_size=[ks, ks], strides=[2,2], padding='same', activation='selu') (img)
        img = Conv2D(n_units_gen*4, kernel_size=[ks, ks], strides=[2,2], padding='same',activation='selu') (img)
        img = BatchNormalization()(img)

        for i in range(n):
            img = build_resnet_block(img, n_units_gen * 4, "r{}".format(i))

        img = Conv2DTranspose(n_units_gen*2, kernel_size=[ks, ks], strides=[2, 2], padding='same',
                              activation='selu') (img)
        img = BatchNormalization()(img)
        img = Conv2DTranspose(n_units_gen, kernel_size=[ks, ks], strides=[2, 2], padding='same',
                              activation='selu') (img)
        img = Conv2D(3, kernel_size=[7, 7], padding='same', activation='tanh') (img)

        return Model(inp,img)

def build_discriminator(input_shape, n, name="discriminator"):

    with tf.variable_scope(name):
        f = 4

        inp = Input(input_shape)

        c = Conv2D(n_units_gen, kernel_size=[f, f], strides=[2,2], activation='selu')(inp)
        c = BatchNormalization()(c)

        for i in range(n-2):
            c = Conv2D(min(n_units_gen * (2**i),256), kernel_size=[f, f], strides=[2, 2], activation='selu')(c)

        c = Conv2D(min(n_units_gen * (2 ** n), 512), kernel_size=[f, f], activation='selu')(c)
        c = BatchNormalization()(c)
        c = Conv2D(min(n_units_gen * (2 ** n), 512), kernel_size=[f, f], activation='selu')(c)
        c = Flatten() (c)
        c = Dense(1, activation='sigmoid')(c)

        return Model(inp, c)


# def patch_discriminator(inputdisc, name="discriminator"):
#
#     with tf.variable_scope(name):
#         f= 4
#
#         patch_input = tf.random_crop(inputdisc,[1,70,70,3])
#         o_c1 = general_conv2d(patch_input, n_units_disc, f, f, 2, 2, 0.02, "SAME", "c1", do_norm="False", relufactor=0.2)
#         o_c2 = general_conv2d(o_c1, n_units_disc * 2, f, f, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)
#         o_c3 = general_conv2d(o_c2, n_units_disc * 4, f, f, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)
#         o_c4 = general_conv2d(o_c3, n_units_disc * 8, f, f, 2, 2, 0.02, "SAME", "c4", relufactor=0.2)
#         o_c5 = general_conv2d(o_c4, 1, f, f, 1, 1, 0.02, "SAME", "c5",do_norm=False,do_relu=False)
#
#         return o_c5


# y = build_generator_resnet_n_blocks([256, 256, 3], 5)
# y.summary()

#
# flat_list = [layer for sublist in l for layer in y.layers]
# d_A_vars = [w for w in [l.weights for l in y.layers]]