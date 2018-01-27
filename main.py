import tensorflow as tf
import numpy as np
import cv2
from read_val import rainy, sunny
from model import *

img_h = 256
img_w = 256
img_layer = 3
img_size = img_h * img_w

to_train = True
to_test = False
to_restore = False
output_path = "./output"
check_dir = "./output/checkpoints/"
temp_check = 0

max_epoch = 1
max_images = 606
batch_size = 2
pool_size = 100
sample_size = 10
save_training_images = True
ngf = 32
ndf = 64

class CycleGAN:

    def model_setup(self):

        ''' This function sets up the model to train

        self.fake_A/self.fake_B -> Generated images by corresponding generator of input_A and input_B
        self.cyc_A/ self.cyc_B -> Images generated after feeding self.fake_A/self.fake_B to corresponding generator. This is use to calcualte cyclic loss
        '''

        # shape (606, 256, 256, 3)
        data_A = tf.constant(rainy[:max_images])
        data_B = tf.constant(sunny[:max_images])

        curr_img = tf.random_uniform((),0, max_images-batch_size, dtype=tf.int32)
        curr_fake = tf.random_uniform((), 0, pool_size - batch_size, dtype=tf.int32)

        # rainy = None
        # sunny = None

        self.input_A = data_A[curr_img: curr_img+ batch_size]
        self.input_B = data_B[curr_img: curr_img+ batch_size]
        
        self.fake_pool_A = tf.get_variable("fake_pool_A", [pool_size, img_w, img_h, img_layer], tf.float32, trainable=False)
        self.fake_pool_B = tf.get_variable("fake_pool_B" ,[pool_size, img_w, img_h, img_layer], tf.float32, trainable=False)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        starter_learning_rate = 2e-4
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                   100000, 0.96, staircase=True)

        with tf.variable_scope("Model") as scope:
            self.fake_B = build_generator_resnet_9blocks(self.input_A, name="g_B")
            self.fake_A = build_generator_resnet_9blocks(self.input_B, name="g_A")
            self.disc_true_A = build_gen_discriminator(self.input_A, "d_A")
            self.disc_true_B = build_gen_discriminator(self.input_B, "d_B")

            slcA = self.fake_pool_A[curr_fake: curr_fake + batch_size]
            slcB = self.fake_pool_B[curr_fake: curr_fake + batch_size]

            scope.reuse_variables()

            self.disc_fake_A = build_gen_discriminator(self.fake_A, "d_A")
            self.disc_fake_B = build_gen_discriminator(self.fake_B, "d_B")
            self.cyc_A = build_generator_resnet_9blocks(self.fake_B, "g_A")
            self.cyc_B = build_generator_resnet_9blocks(self.fake_A, "g_B")

            scope.reuse_variables()

            self.fake_pool_rec_A = build_gen_discriminator(slcA, "d_A")
            self.fake_pool_rec_B = build_gen_discriminator(slcB, "d_B")

        self.save_fakes = [tf.assign(slcA, self.fake_A), tf.assign(slcB, self.fake_B)]


    def loss_calc(self):

        ''' In this function we are defining the variables for loss calcultions and traning model

        d_loss_A/d_loss_B -> loss for discriminator A/B
        g_loss_A/g_loss_B -> loss for generator A/B
        *_trainer -> Variaous trainer for above loss functions
        *_summ -> Summary variables for above loss functions'''

        cyc_loss = tf.reduce_mean(tf.abs(self.input_A-self.cyc_A)) + tf.reduce_mean(tf.abs(self.input_B-self.cyc_B))
        
        adv_loss_A = tf.reduce_mean(tf.squared_difference(self.disc_fake_A, 1))
        adv_loss_B = tf.reduce_mean(tf.squared_difference(self.disc_fake_B, 1))
        
        # g_loss_A = cyc_loss*10 + adv_loss_A
        # g_loss_B = cyc_loss*10 + adv_loss_B

        g_loss = cyc_loss*10 + adv_loss_A + adv_loss_B

        d_loss_A = (tf.reduce_mean(tf.square(self.fake_pool_rec_A)) + tf.reduce_mean(tf.squared_difference(self.disc_true_A, 1))) / 2.0
        d_loss_B = (tf.reduce_mean(tf.square(self.fake_pool_rec_B)) + tf.reduce_mean(tf.squared_difference(self.disc_true_B, 1))) / 2.0

        d_loss = d_loss_A + d_loss_B

        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)

        self.model_vars = tf.trainable_variables()

        d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
        g_A_vars = [var for var in self.model_vars if 'g_A' in var.name]
        d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
        g_B_vars = [var for var in self.model_vars if 'g_B' in var.name]

        self.model_vars = d_A_vars + d_B_vars + g_A_vars + g_B_vars


        
        # self.d_A_trainer = optimizer.minimize(d_loss_A, var_list=d_A_vars, global_step = self.global_step)
        # self.d_B_trainer = optimizer.minimize(d_loss_B, var_list=d_B_vars, global_step = self.global_step)
        # self.g_A_trainer = optimizer.minimize(g_loss_A, var_list=g_A_vars, global_step = self.global_step)
        # self.g_B_trainer = optimizer.minimize(g_loss_B, var_list=g_B_vars, global_step = self.global_step)

        self.d_trainer = optimizer.minimize(d_loss, var_list=d_A_vars+d_B_vars, global_step = self.global_step)
        self.g_trainer = optimizer.minimize(g_loss, var_list=g_A_vars+g_B_vars, global_step = self.global_step)


        # for var in self.model_vars: print(var.name)

        #Summary variables for tensorboard

        self.g_A_loss_summ = tf.summary.scalar("g_loss", g_loss)
        # self.g_A_loss_summ = tf.summary.scalar("g_A_loss", g_loss)
        # self.g_B_loss_summ = tf.summary.scalar("g_B_loss", g_loss_B)
        self.d_A_loss_summ = tf.summary.scalar("d_A_loss", d_loss_A)
        self.d_B_loss_summ = tf.summary.scalar("d_B_loss", d_loss_B)

        self.summary = tf.summary.merge_all()

    def save_training_images(self, sess, epoch):

        if not os.path.exists("./output/imgs"):
            os.makedirs("./output/imgs")



        in_A, in_B, fake_A, fake_B, cyc_A, cyc_B = \
            sess.run([self.input_A, self.input_B,self.fake_A,self.fake_B,
                                                             self.cyc_A,
                                                             self.cyc_B])
        for i in range(0,batch_size):
            cv2.imwrite("./output/imgs/fakeB_"+ str(epoch) + "_" + str(i)+".jpg",((fake_A[i]+1)*127.5).astype(
                np.uint8))
            cv2.imwrite("./output/imgs/fakeA_"+ str(epoch) + "_" + str(i)+".jpg",((fake_B[i]+1)*127.5).astype(
                np.uint8))
            cv2.imwrite("./output/imgs/cycA_"+ str(epoch) + "_" + str(i)+".jpg",((cyc_A[i]+1)*127.5).astype(np.uint8))
            cv2.imwrite("./output/imgs/cycB_"+ str(epoch) + "_" + str(i)+".jpg",((cyc_B[i]+1)*127.5).astype(np.uint8))
            cv2.imwrite("./output/imgs/inputA_"+ str(epoch) + "_" + str(i)+".jpg",((in_A[i]+1)*127.5).astype(np.uint8))
            cv2.imwrite("./output/imgs/inputB_"+ str(epoch) + "_" + str(i)+".jpg",((in_B[i]+1)*127.5).astype(np.uint8))

    ''' Training Function '''
    def train(self):
        with tf.Session().as_default() as sess:

            self.model_setup()
            self.loss_calc()

            saver = tf.train.Saver(self.model_vars)
            if to_restore:
                chkpt_fname = tf.train.latest_checkpoint(check_dir)
                saver.restore(sess, chkpt_fname)
            else:
                init = (tf.global_variables_initializer(), tf.local_variables_initializer())
                sess.run(init)

            writer = tf.summary.FileWriter("./output/2")

            writer.add_graph(sess.graph)
            writer.flush()

            if not os.path.exists(check_dir):
                os.makedirs(check_dir)

            # Training Loop
            # init = (tf.global_variables_initializer(), tf.local_variables_initializer())
            # sess.run(init)

            for epoch in range(402):
                print ("In the epoch ", epoch)
                if epoch % 50 == 1:
                    saver.save(sess,os.path.join(check_dir,"cyclegan"),global_step=epoch)
                if epoch % 2 == 1 and save_training_images:
                    self.save_training_images(sess, epoch)

                # sys.exit()
                for ptr in range(0, max_images // batch_size):
                    if ptr % 20 == 0:
                        print("In the iteration ",ptr)

                    sess.run(self.g_trainer)
                    sess.run(self.save_fakes)
                    sess.run(self.d_trainer)
                    # sess.run(self.g_A_trainer)
                    # sess.run(self.g_B_trainer)
                    # sess.run(self.d_A_trainer)
                    # sess.run(self.d_B_trainer)
                    # sess.run([self.d_B_trainer, self.d_A_trainer])

                    # summary_str = sess.run(self.summary)
                    # writer.add_summary(summary_str, epoch*max_images + ptr)
                    



    def test(self):
        ''' Testing Function: load the model, generate images, save them'''
        print("Testing the results")
        self.model_setup()
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:

            sess.run(init)

            chkpt_fname = tf.train.latest_checkpoint(check_dir)
            saver.restore(sess, chkpt_fname)

            if not os.path.exists("./output/imgs/test/"):
                os.makedirs("./output/imgs/test/")            

            # for i in range(0,100):
            #     fake_A_temp, fake_B_temp = sess.run([self.fake_A, self.fake_B], feed_dict={self.input_A:self.mem_A[i], self.input_B:self.mem_B[i]})
            #     imsave("./output/imgs/test/fakeB_"+str(i)+".jpg",((fake_A_temp[0]+1)*127.5).astype(np.uint8))
            #     imsave("./output/imgs/test/fakeA_"+str(i)+".jpg",((fake_B_temp[0]+1)*127.5).astype(np.uint8))
            #     imsave("./output/imgs/test/inputA_" + str(i) +".jpg", ((self.mem_A[i][0] + 1) * 127.5).astype(np.uint8))
            #     imsave("./output/imgs/test/inputB_" + str(i) +".jpg", ((self.mem_B[i][0] + 1) * 127.5).astype(np.uint8))


def main():
    
    model = CycleGAN()
    if to_train:
        model.train()
    elif to_test:
        model.test()

if __name__ == '__main__':

    main()