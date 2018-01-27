from tensorflow.python.keras import backend as K
import cv2
from read_val import rainy, sunny
from model import *
import os
import numpy as np

img_h = 256
img_w = 256
img_layer = 3
img_shape = [256, 256, 3]
img_size = img_h * img_w

to_train = True
to_test = False
to_restore = False
output_path = "./output"
check_dir = "./output/checkpoints/"
img_dir = "./output/imgs/"
temp_check = 0

max_epoch = 1
max_images = 606
batch_size = 1
pool_size = 100
sample_size = 10
save_training_images = True
ngf = 32
ndf = 64


def restore(img):
    return ((img + 1) * 127.5).astype(np.uint8)

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

            self.gen_A_to_B = build_generator_resnet_n_blocks(img_shape, 9, name="g_B")
            self.gen_B_to_A = build_generator_resnet_n_blocks(img_shape, 9, name="g_A")
            self.disc_A = build_discriminator(img_shape, 5, "d_A")
            self.disc_B = build_discriminator(img_shape, 5, "d_B")

            self.fake_B = self.gen_A_to_B(self.input_A)
            self.fake_A = self.gen_B_to_A(self.input_B)

            self.disc_true_A = self.disc_A(self.input_A)
            self.disc_true_B = self.disc_B(self.input_B)

            self.disc_fake_A = self.disc_A(self.fake_A)
            self.disc_fake_B = self.disc_B(self.fake_B)

            self.cyc_A = self.gen_B_to_A(self.fake_B)
            self.cyc_B = self.gen_A_to_B(self.fake_A)

            slcA = self.fake_pool_A[curr_fake: curr_fake + batch_size]
            slcB = self.fake_pool_B[curr_fake: curr_fake + batch_size]

            self.fake_pool_rec_A = self.disc_A(slcA)
            self.fake_pool_rec_B = self.disc_B(slcB)

        self.save_fakes = [tf.assign(slcA, self.fake_A), tf.assign(slcB, self.fake_B)]


    def loss_calc(self):

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

        model_vars = tf.trainable_variables()

        d_A_vars = [var for var in model_vars if 'd_A' in var.name]
        g_A_vars = [var for var in model_vars if 'g_A' in var.name]
        d_B_vars = [var for var in model_vars if 'd_B' in var.name]
        g_B_vars = [var for var in model_vars if 'g_B' in var.name]

        print([var.name for var in d_A_vars + d_B_vars + g_A_vars + g_B_vars])

        # self.d_A_trainer = optimizer.minimize(d_loss_A, var_list=d_A_vars, global_step = self.global_step)
        # self.d_B_trainer = optimizer.minimize(d_loss_B, var_list=d_B_vars, global_step = self.global_step)
        # self.g_A_trainer = optimizer.minimize(g_loss_A, var_list=g_A_vars, global_step = self.global_step)
        # self.g_B_trainer = optimizer.minimize(g_loss_B, var_list=g_B_vars, global_step = self.global_step)

        self.d_trainer = optimizer.minimize(d_loss, var_list=d_A_vars+d_B_vars, global_step = self.global_step)
        self.g_trainer = optimizer.minimize(g_loss, var_list=g_A_vars+g_B_vars, global_step = self.global_step)


        # for var in self.model_vars: print(var.name)

        #Summary variables for tensorboard

        # self.g_A_loss_summ = tf.summary.scalar("g_loss", g_loss)
        # self.g_A_loss_summ = tf.summary.scalar("g_A_loss", g_loss)
        # self.g_B_loss_summ = tf.summary.scalar("g_B_loss", g_loss_B)
        # self.d_A_loss_summ = tf.summary.scalar("d_A_loss", d_loss_A)
        # self.d_B_loss_summ = tf.summary.scalar("d_B_loss", d_loss_B)

        self.summary = tf.summary.merge_all()

    def save_training_images(self, sess, epoch):

        if not os.path.exists("./output/imgs"):
            os.makedirs("./output/imgs")



        in_A, in_B, fake_A, fake_B, cyc_A, cyc_B = \
            sess.run([self.input_A, self.input_B,self.fake_A,self.fake_B,
                                                             self.cyc_A,
                                                             self.cyc_B])
        for i in range(0,batch_size):
            cv2.imwrite(img_dir+"fakeB_"+ str(epoch) + "_" + str(i)+".jpg",(restore(fake_A[i])))
            cv2.imwrite(img_dir+"fakeA_"+ str(epoch) + "_" + str(i)+".jpg",(restore(fake_B[i])))
            cv2.imwrite(img_dir+"cycA_"+ str(epoch) + "_" + str(i)+".jpg",(restore(cyc_A[i])))
            cv2.imwrite(img_dir+"cycB_"+ str(epoch) + "_" + str(i)+".jpg",(restore(cyc_B[i])))
            cv2.imwrite(img_dir+"inputA_"+ str(epoch) + "_" + str(i)+".jpg",(restore(in_A[i])))
            cv2.imwrite(img_dir+"inputB_"+ str(epoch) + "_" + str(i)+".jpg",(restore(in_B[i])))



    ''' Training Function '''
    def train(self):
        with tf.Session().as_default() as sess:

            K.set_session(sess)

            self.model_setup()
            self.loss_calc()

            init = (tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)

            if to_restore:
                self.gen_A_to_B.load_weights(check_dir +"g_B")
                self.gen_B_to_A.load_weights(check_dir +"g_A")
                self.disc_A.load_weights(check_dir +"d_A")
                self.disc_B.load_weights(check_dir +"d_B")

            # writer = tf.summary.FileWriter("./output/2")

            # writer.add_graph(sess.graph)
            # writer.flush()

            if not os.path.exists(check_dir):
                os.makedirs(check_dir)

            # Training Loop
            # init = (tf.global_variables_initializer(), tf.local_variables_initializer())
            # sess.run(init)


            for epoch in range(402):
                print ("In the epoch ", epoch)
                if epoch % 50 == 1:
                    self.gen_A_to_B.save(check_dir +"g_B")
                    self.gen_B_to_A.save(check_dir +"g_A")
                    self.disc_A.save(check_dir +"d_A")
                    self.disc_B.save(check_dir +"d_B")
                    # saver.save(sess,os.path.join(check_dir,"cyclegan"),global_step=epoch)
                if epoch % 2 == 1 and save_training_images:
                    self.save_training_images(sess, epoch)

                # sys.exit()
                for ptr in range(0, max_images // batch_size):
                    if ptr % 20 == 0:
                        print("In the iteration ",ptr)
                        sess.run(self.g_trainer)
                        sess.run(self.save_fakes)
                        sess.run(self.d_trainer)
                    else:
                        sess.run(self.g_trainer)
                        # sess.run(self.g_A_trainer)
                        # sess.run(self.g_B_trainer)
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