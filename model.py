import tensorflow as tf
import ops
from generator import Generator
from discriminator import Discriminator
from inverseGenerator import inverseGenerator

class AEGAN:
    def __init__(self,
                 batch_size = 10,
                 norm = 'batch',
                 learning_rate_disc = 1e-4,
                 learning_rate_gen = 1e-3,
                 learning_rate_igen = 1e-2):
        self.batch_size = batch_size
        self.norm = norm
        self.learning_rate_disc = learning_rate_disc
        self.learning_rate_gen = learning_rate_gen
        self.learning_rate_igen = learning_rate_igen

        self.is_training = tf.placeholder_with_default(True, shape = [], name = 'is_training')
        self.x = tf.placeholder(tf.float32, shape = [None, 28, 28, 1])
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.z_con = tf.random_normal((batch_size, 2))
        self.z_rand = tf.random_normal((batch_size, 38))
        self.z = tf.concat([self.y, self.z_con, self.z_rand], axis = 1)
        self.G = Generator('G', is_training = self.is_training,
                           norm = self.norm)
        self.D = Discriminator('D', is_training = self.is_training,
                               norm = self.norm)
        self.IG = inverseGenerator('IG', is_training = self.is_training,
                                   norm = self.norm)

    def loss_function(self, D, G, IG, z, z_con, x, y):
        disc_real, class_real, _ = D(x)
        disc_fake, class_fake, con_fake = D(G(z))
        loss_d_r = -tf.reduce_mean(tf.log(disc_real))
        loss_d_f = -tf.reduce_mean(tf.log(1 - disc_fake))
        loss_d = 0.5 * (loss_d_r + loss_d_f)
        loss_g = -tf.reduce_mean(tf.log(disc_fake))
        loss_c_r = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits = class_real, labels = y))
        loss_c_f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits = class_fake, labels = y))
        loss_c = 0.5 * (loss_c_f + loss_c_r)
        loss_con = tf.reduce_mean(tf.square(con_fake - z_con))
        igen = G(IG(G(z)))
        loss_ae = tf.reduce_mean(tf.square(igen - G(z)))
        return loss_d, loss_g, loss_c, loss_con, loss_ae

    def optimize(self, loss_d, loss_g, loss_c, loss_con, loss_ae):
        def make_optimizer(loss, learning_rate, variables, name = 'Adam'):
            global_step = tf.Variable(0, trainable = False)
            starter_learning_rate = learning_rate
            end_learning_rate = 0.0
            start_decay_step = 10000
            decay_steps = 10000
            learning_rate = (
                tf.where(
                    tf.greater_equal(global_step, start_decay_step),
                    tf.train.polynomial_decay(starter_learning_rate,
                                              global_step - start_decay_step,
                                              decay_steps,
                                              end_learning_rate,
                                              power = 1.0),
                    starter_learning_rate
                )
            )
            tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate, name = name)\
                .minimize(loss, var_list = variables, global_step = global_step)
            return optimizer

        train_disc = make_optimizer(loss_d + loss_c + loss_con,
                                    self.learning_rate_disc,
                                    self.D.variables, name = 'discAdam')
        train_gen = make_optimizer(loss_g + loss_c + loss_con,
                                   self.learning_rate_gen,
                                   self.G.variables, name = 'genAdam')
        train_ae = make_optimizer(loss_ae, self.learning_rate_igen,
                                  self.IG.variables, name = 'igenAdam')
        with tf.control_dependencies([train_disc, train_gen, train_ae]):
            return tf.no_op(name='optimizers')


    def model(self):
        loss_d, loss_g, loss_c, loss_con, loss_ae = self.loss_function(
            self.D, self.G, self.IG, self.z, self.z_con, self.x, self.y)
        return loss_d, loss_g, loss_c, loss_con, loss_ae

    def out(self):
        gen = self.G(self.z)
        genout = tf.squeeze(gen, -1)

        igen = self.G(self.IG(self.x))
        igenout = tf.squeeze(igen, -1)
        return genout, igenout
