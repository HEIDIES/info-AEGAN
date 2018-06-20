import tensorflow as tf
import layers
import ops

class Discriminator:
    def __init__(self, name, is_training, norm = 'batch'):
        self.name = name
        self.is_training = is_training
        self.norm = norm
        self.reuse = False

    def __call__(self, input):
        self.reuse = len([var for var in tf.global_variables()
                          if var.name.startswith(self.name)]) > 0
        with tf.variable_scope(self.name):
            input = tf.reshape(input, [-1, 28, 28, 1])
            c4s2k64 = layers.c4s2k64(input, reuse = self.reuse,
                                     is_training = self.is_training, norm = self.norm)
            c4s2k128 = layers.c4s2k128(c4s2k64, reuse = self.reuse,
                                       is_training = self.is_training, norm = self.norm)
            d1024 = layers.d1024(c4s2k128, reuse = self.reuse,
                                 is_training = self.is_training, norm = self.norm,
                                 activation = ops.leaky_relu)
            d1 = layers.d1(d1024, reuse = self.reuse,
                           is_training = self.is_training,
                           norm = None,
                           activation = tf.nn.sigmoid)
            ds128 = layers.ds128(d1024, reuse = self.reuse,
                                 is_training = self.is_training,
                                 norm = self.norm,
                                 activation = ops.leaky_relu)
            d10 = layers.d10(ds128, reuse = self.reuse,
                             is_training = self.is_training,
                             norm = None,
                             activation = None)
            d2 = layers.d2(ds128, reuse = self.reuse,
                           is_training = self.is_training,
                           norm = None,
                           activation = tf.nn.sigmoid)

            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               scope = self.name)
            return d1, d10, d2
