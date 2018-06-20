import tensorflow as tf
import numpy as np
import ops

def d1024(x, reuse = False, is_training = True, norm = 'batch',
          name = 'd1024', activation = tf.nn.relu):
    # 全连接层，1024输出
    return ops.fully_connected(x, 1024, use_bias = False, reuse = reuse,
                               activation = activation, is_training = is_training,
                               name = name, norm = norm)

def d7x7x128(x, reuse = False, is_training = True, norm = 'batch',
             name = 'd7x7x128', activation = tf.nn.relu):
    # 全连接层，7*7*128 = 6272输出
    return ops.fully_connected(x, 7 * 7 * 128, use_bias = False, reuse = reuse,
                               activation = activation, is_training = is_training,
                               name = name, norm = norm)

def d10(x, reuse = False, is_training = True, norm = None,
        name = 'd10', activation = None):
    # 全链接层，10输出
    return ops.fully_connected(x, 10, use_bias = False, reuse = reuse,
                               activation = activation, is_training = is_training,
                               name = name, norm = norm)

def d2(x, reuse = False, is_training = True, norm = None,
       name = 'd2', activation = tf.nn.sigmoid):
    # 全链接层，2输出
    return ops.fully_connected(x, 2, use_bias = False, reuse = reuse,
                               activation = activation, is_training = is_training,
                               name = name, norm = norm)

def d1(x, reuse = False, is_training = True, name = 'd1', norm = None,
       activation = None):
    return tf.squeeze(ops.fully_connected(x, 1, use_bias = False, is_training = is_training,
                                          activation = activation, reuse = reuse,
                                          name = name, norm = norm), -1)

def ds128(x, reuse = False, is_training = True, name = 'ds128', norm = None,
          activation = ops.leaky_relu):
    return ops.fully_connected(x, 128, use_bias = False, is_training = is_training,
                               activation = activation, reuse = reuse,
                               name = name, norm = norm)

def d50(x, reuse = False, is_training = True, name = 'd50', norm = None,
        activation = ops.leaky_relu):
    return ops.fully_connected(x, 50, use_bias = False, is_training = is_training,
                               activation = activation, reuse = reuse,
                               name = name, norm = norm)

def uc4s2k64(x, reuse = False, is_training = True, norm = 'batch',
             name = 'uc4s2k64', activation = tf.nn.relu):
    # 反卷积层，4x4卷积核，2步长，64输出通道
    x = tf.reshape(x, [-1, 7, 7, 128])
    return ops.unconv2d(x, 64, 4, stride = 2, norm = norm,
                        activation = activation, is_training = is_training,
                        reuse = reuse, name = name)
def uc4s2k1(x, reuse = False, is_training = True, norm = None,
            name = 'uc4s2k1', activation = tf.nn.sigmoid):
    # 反卷积层，4x4卷积核，2步长，1输出通道
    return ops.unconv2d(x, 1, 4, stride = 2, norm = norm,
                        activation = activation, is_training = is_training,
                        reuse = reuse, name = name)

def c4s2k64(x, reuse = False, is_training = True, norm = 'batch',
            name = 'c4s2k64', activation = ops.leaky_relu):
    # 卷积层，4x4卷积核，2步长，64输出通道
    x = tf.reshape(x, [-1, 28, 28, 1])
    return ops.conv2d(x, 64, 4, stride = 2, norm = norm,
                      activation = activation, is_training = is_training,
                      reuse = reuse, name = name)
def c4s2k128(x, reuse = False, is_training = True, norm = 'batch',
             name = 'c4s2k128', activation = ops.leaky_relu):
    # 卷积层，4x4卷积核，2步长，128输出通道
    return tf.reshape(ops.conv2d(x, 128, 4, stride = 2, norm = norm,
                      activation = activation, is_training = is_training,
                      reuse = reuse, name = name), [-1, 7 * 7 * 128])
