import tensorflow as tf
import layers

class Generator:
    def __init__(self, name, is_training, norm = 'batch'):
        self.name = name
        self.is_training = is_training
        self.norm = norm
        self.reuse = False

    def __call__(self, input):
        self.reuse = len([var for var in tf.global_variables()
                          if var.name.startswith(self.name)]) > 0
        with tf.variable_scope(self.name):
            d1024 = layers.d1024(input, reuse = self.reuse,
                                   is_training = self.is_training, norm = self.norm)
            d7x7x128 = layers.d7x7x128(d1024, reuse = self.reuse,
                                   is_training = self.is_training, norm = self.norm)
            uc4s2k64 = layers.uc4s2k64(d7x7x128, reuse = self.reuse,
                                     is_training = self.is_training, norm = self.norm)
            uc4s2k1 = layers.uc4s2k1(uc4s2k64, reuse = self.reuse,
                                    is_training = self.is_training, norm = None)

            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                               scope = self.name)
            return uc4s2k1

    '''
    def sample(self, input):
        image = utils.batch_convert2int(self.__call__(input))
        image = tf.image.encode_jpeg(tf.squeeze(image, [0]))
        return image
    '''
