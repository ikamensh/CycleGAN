import tensorflow as tf
import numpy as np
import cv2
from read_val import rainy, sunny
from model import *

img_height = 256
img_width = 256
img_layer = 3
img_size = img_height * img_width

to_train = True
to_test = False
to_restore = True
output_path = "./output"
check_dir = "./output/checkpoints/"
temp_check = 0



max_epoch = 1
max_images = 606
h1_size = 150
h2_size = 300
z_size = 100
batch_size = 1
pool_size = 50
sample_size = 10
save_training_images = True
ngf = 32
ndf = 64

class CycleGAN:


    def input_setup(self):
        self.fake_images_A = np.zeros((pool_size,1,img_height, img_width, img_layer))
        self.fake_images_B = np.zeros((pool_size,1,img_height, img_width, img_layer))

        self.mem_A = np.expand_dims(rainy[:606], axis=1)
        self.mem_B = np.expand_dims(sunny, axis=1)


    def model_setup(self):

        ''' This function sets up the model to train

        self.input_A/self.input_B -> Set of training images.
        self.fake_A/self.fake_B -> Generated images by corresponding generator of input_A and input_B
        self.lr -> Learning rate variable
        self.cyc_A/ self.cyc_B -> Images generated after feeding self.fake_A/self.fake_B to corresponding generator. This is use to calcualte cyclic loss
        '''

        self.input_A = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_A")
        self.input_B = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_B")
        
        self.fake_pool_A = tf.placeholder(tf.float32, [None, img_width, img_height, img_layer], name="fake_pool_A")
        self.fake_pool_B = tf.placeholder(tf.float32, [None, img_width, img_height, img_layer], name="fake_pool_B")

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.num_fake_inputs = 0

        starter_learning_rate = 2e-4
        self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                   100000, 0.96, staircase=True)

        with tf.variable_scope("Model") as scope:
            self.fake_B = build_generator_resnet_9blocks(self.input_A, name="g_A")
            self.fake_A = build_generator_resnet_9blocks(self.input_B, name="g_B")
            self.rec_A = build_gen_discriminator(self.input_A, "d_A")
            self.rec_B = build_gen_discriminator(self.input_B, "d_B")

            scope.reuse_variables()

            self.fake_rec_A = build_gen_discriminator(self.fake_A, "d_A")
            self.fake_rec_B = build_gen_discriminator(self.fake_B, "d_B")
            self.cyc_A = build_generator_resnet_9blocks(self.fake_B, "g_B")
            self.cyc_B = build_generator_resnet_9blocks(self.fake_A, "g_A")

            scope.reuse_variables()

            self.fake_pool_rec_A = build_gen_discriminator(self.fake_pool_A, "d_A")
            self.fake_pool_rec_B = build_gen_discriminator(self.fake_pool_B, "d_B")

    def loss_calc(self):

        ''' In this function we are defining the variables for loss calcultions and traning model

        d_loss_A/d_loss_B -> loss for discriminator A/B
        g_loss_A/g_loss_B -> loss for generator A/B
        *_trainer -> Variaous trainer for above loss functions
        *_summ -> Summary variables for above loss functions'''

        cyc_loss = tf.reduce_mean(tf.abs(self.input_A-self.cyc_A)) + tf.reduce_mean(tf.abs(self.input_B-self.cyc_B))
        
        disc_loss_A = tf.reduce_mean(tf.squared_difference(self.fake_rec_A,1))
        disc_loss_B = tf.reduce_mean(tf.squared_difference(self.fake_rec_B,1))
        
        g_loss_A = cyc_loss*10 + disc_loss_B*3
        g_loss_B = cyc_loss*10 + disc_loss_A

        d_loss_A = (tf.reduce_mean(tf.square(self.fake_pool_rec_A)) + tf.reduce_mean(tf.squared_difference(self.rec_A,1)))/2.0
        d_loss_B = (tf.reduce_mean(tf.square(self.fake_pool_rec_B)) + tf.reduce_mean(tf.squared_difference(self.rec_B,1)))/2.0

        
        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)

        self.model_vars = tf.trainable_variables()

        d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
        g_A_vars = [var for var in self.model_vars if 'g_A' in var.name]
        d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
        g_B_vars = [var for var in self.model_vars if 'g_B' in var.name]
        
        self.d_A_trainer = optimizer.minimize(d_loss_A, var_list=d_A_vars, global_step = self.global_step)
        self.d_B_trainer = optimizer.minimize(d_loss_B, var_list=d_B_vars, global_step = self.global_step)
        self.g_A_trainer = optimizer.minimize(g_loss_A, var_list=g_A_vars, global_step = self.global_step)
        self.g_B_trainer = optimizer.minimize(g_loss_B, var_list=g_B_vars, global_step = self.global_step)

        # for var in self.model_vars: print(var.name)

        #Summary variables for tensorboard

        self.g_A_loss_summ = tf.summary.scalar("g_A_loss", g_loss_A)
        self.g_B_loss_summ = tf.summary.scalar("g_B_loss", g_loss_B)
        self.d_A_loss_summ = tf.summary.scalar("d_A_loss", d_loss_A)
        self.d_B_loss_summ = tf.summary.scalar("d_B_loss", d_loss_B)

        self.summary = tf.summary.merge_all()

    def save_training_images(self, sess, epoch):

        if not os.path.exists("./output/imgs"):
            os.makedirs("./output/imgs")

        for i in range(0,10):
            fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run([self.fake_A, self.fake_B, self.cyc_A, self.cyc_B], feed_dict={self.input_A:self.mem_A[i], self.input_B:self.mem_B[i]})
            cv2.imwrite("./output/imgs/fakeB_"+ str(epoch) + "_" + str(i)+".jpg",((fake_A_temp[0]+1)*127.5).astype(np.uint8))
            cv2.imwrite("./output/imgs/fakeA_"+ str(epoch) + "_" + str(i)+".jpg",((fake_B_temp[0]+1)*127.5).astype(np.uint8))
            cv2.imwrite("./output/imgs/cycA_"+ str(epoch) + "_" + str(i)+".jpg",((cyc_A_temp[0]+1)*127.5).astype(np.uint8))
            cv2.imwrite("./output/imgs/cycB_"+ str(epoch) + "_" + str(i)+".jpg",((cyc_B_temp[0]+1)*127.5).astype(np.uint8))
            # cv2.imwrite("./output/imgs/inputA_"+ str(epoch) + "_" + str(i)+".jpg",((self.A_input[i][0]+1)*127.5).astype(np.uint8))
            # cv2.imwrite("./output/imgs/inputB_"+ str(epoch) + "_" + str(i)+".jpg",((self.B_input[i][0]+1)*127.5).astype(np.uint8))

    def fake_image_pool(self, num_fakes, fake, fake_pool):
        ''' This function saves the generated image to corresponding pool of images.
        In starting. It keeps on feeling the pool till it is full and then randomly selects an
        already stored image and replace it with new one.'''

        if(num_fakes < pool_size):
            fake_pool[num_fakes] = fake
            return fake
        else :
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0,pool_size-1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else :
                return fake

    ''' Training Function '''
    def train(self):
        with tf.Session().as_default() as sess:

            init = (tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)


            # Load Dataset from the dataset folder
            self.input_setup()

            #Build the network
            self.model_setup()

            #Loss function calculations
            self.loss_calc()

            saver = tf.train.Saver()

            #Restore the model to run the model from last checkpoint
            if to_restore:
                chkpt_fname = tf.train.latest_checkpoint(check_dir)
                saver.restore(sess, chkpt_fname)

            writer = tf.summary.FileWriter("./output/2")

            if not os.path.exists(check_dir):
                os.makedirs(check_dir)

            # Training Loop
            init = (tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)

            for epoch in range(100):
                print ("In the epoch ", epoch)
                if epoch % 5 == 0:
                    saver.save(sess,os.path.join(check_dir,"cyclegan"),global_step=epoch)

                # Dealing with the learning rate as per the epoch number


                if(save_training_images):
                    self.save_training_images(sess, epoch)

                # sys.exit()

                for ptr in range(0,max_images):
                    if ptr % 25 == 0:
                        print("In the iteration ",ptr)

                    # Optimizing the G_A network
                    _, fake_B_temp = sess.run([self.g_A_trainer, self.fake_B], feed_dict={self.input_A:self.mem_A[ptr],
                                                                                          self.input_B:self.mem_B[ptr]})

                    fake_B_temp1 = self.fake_image_pool(self.num_fake_inputs, fake_B_temp, self.fake_images_B)
                    
                    # Optimizing the D_B network
                    sess.run(self.d_B_trainer, feed_dict={self.input_A: self.mem_A[ptr],
                                                          self.input_B:self.mem_B[ptr],
                                                          self.fake_pool_B:fake_B_temp1})

                    # Optimizing the G_B network
                    _, fake_A_temp= sess.run([self.g_B_trainer, self.fake_A], feed_dict={self.input_A:self.mem_A[ptr],
                                                                                         self.input_B:self.mem_B[ptr]})

                    fake_A_temp1 = self.fake_image_pool(self.num_fake_inputs, fake_A_temp, self.fake_images_A)

                    # Optimizing the D_A network
                    sess.run(self.d_A_trainer, feed_dict={self.input_A:self.mem_A[ptr],
                                                          self.input_B:self.mem_B[ptr],
                                                          self.fake_pool_A:fake_A_temp1})

                    summary_str = sess.run(self.summary)
                    writer.add_summary(summary_str, epoch*max_images + ptr)
                    
                    self.num_fake_inputs+=1
            


            writer.add_graph(sess.graph)

    def test(self):
        ''' Testing Function'''
        print("Testing the results")
        self.input_setup()
        self.model_setup()
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:

            sess.run(init)

            chkpt_fname = tf.train.latest_checkpoint(check_dir)
            saver.restore(sess, chkpt_fname)

            if not os.path.exists("./output/imgs/test/"):
                os.makedirs("./output/imgs/test/")            

            for i in range(0,100):
                fake_A_temp, fake_B_temp = sess.run([self.fake_A, self.fake_B], feed_dict={self.input_A:self.mem_A[i], self.input_B:self.mem_B[i]})
                imsave("./output/imgs/test/fakeB_"+str(i)+".jpg",((fake_A_temp[0]+1)*127.5).astype(np.uint8))
                imsave("./output/imgs/test/fakeA_"+str(i)+".jpg",((fake_B_temp[0]+1)*127.5).astype(np.uint8))
                imsave("./output/imgs/test/inputA_" + str(i) +".jpg", ((self.mem_A[i][0] + 1) * 127.5).astype(np.uint8))
                imsave("./output/imgs/test/inputB_" + str(i) +".jpg", ((self.mem_B[i][0] + 1) * 127.5).astype(np.uint8))


def main():
    
    model = CycleGAN()
    if to_train:
        model.train()
    elif to_test:
        model.test()

if __name__ == '__main__':

    main()