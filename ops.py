# -*- coding: utf-8 -*-
"""
@author: shoh4486
tf.__version__ == '1.12.0' ~ '1.14.0'
"""
import tensorflow as tf

def lr_polynomial_decay(init_lr, global_step, decay_steps, end_lr=0.0, power=1.0, name='lr_polynomial_decay'):
    """
    Polynomial decay of learning rate
    (https://www.tensorflow.org/api_docs/python/tf/train/polynomial_decay)
    """
    return (init_lr - end_lr)*(1 - global_step/decay_steps)**power + end_lr

def miou_tf(pred, gt, keep_batch_dim=False):
    """
    mean Intersection-Over-Union (mIOU) calculation using TensorFlow
    
    Parameters
    pred: prediction results (shape: [N, H, W, C])
    gt: ground truths (shape: [N, H, W, C])
    
    - pixel values must be either 0 or 1 (label).
    """
    intersection = tf.reduce_sum(pred*gt, axis=[1, 2]) # calculated over spatial dimension, shape: [N, C]
    union = tf.reduce_sum(pred + gt, axis=[1, 2]) - intersection
    iou = intersection/union 
    
    if not keep_batch_dim:
        miou = tf.reduce_mean(iou) # averaging across the batch
        return miou
    else:
        miou = tf.reshape(tf.reduce_mean(iou, axis=-1), [-1, 1]) # not averaged across the batch, shape: [N, 1]
        return miou
    
def pixel_acc_4D_tf(pred, gt, return_axis='ALL'):
    """
    pixel accuracy calculatiaon using TensorFlow
    
    Parameters
    pred: prediction results (shape: [N, H, W, C])
    gt: ground truths (shape: [N, H, W, C])
    
    - pixel values must be either 0 or 1 (label).
    - return_axis: dimension to return ('BC', 'B', 'C', or 'ALL'; B and C denote batch and channel dimension, respectively.)
    """
    true_positive = tf.reduce_sum(pred*gt, axis=[1, 2])
    true_negative = tf.reduce_sum((1-pred)*(1-gt), axis=[1, 2])
    
    pixel_acc_batch_channel = tf.cast(true_positive + true_negative, tf.float32)/tf.cast(tf.shape(pred)[1]*tf.shape(pred)[2], tf.float32)
    # calculated over spatial dimension, shape: [N, C]
    
    if return_axis == 'BC':
        return pixel_acc_batch_channel # shape: [N, C]
    elif return_axis == 'B':
        pixel_acc_batch = tf.reshape(tf.reduce_mean(pixel_acc_batch_channel, axis=-1), [-1, 1])
        return pixel_acc_batch # shape: [N, 1]
    elif return_axis == 'C':
        pixel_acc_channel = tf.reshape(tf.reduce_mean(pixel_acc_batch_channel, axis=0), [-1, 1])
        return pixel_acc_channel # shape: [C, 1]
    elif return_axis == 'ALL':
        pixel_acc_all = tf.reduce_mean(pixel_acc_batch_channel)
        return pixel_acc_all # total averaged, shape: a scalar
    else:
        raise NotImplementedError('Parameter \'return_axis\' should be one of \'BC\', \'B\', \'C\' and \'ALL\'.')
        
def l_relu(inputs, alpha=0.2, name='leaky_relu'):
    """
    Leaky ReLU
    (Maas, A. L. et al., Rectifier nonlinearities imporve neural network acoustic models, Proc. icml. Vol.30. No.1. 2013)
    """
    return tf.maximum(inputs, alpha*inputs) # == tf.nn.leaky_relu(inputs, alpha)

class BN:
    """
    Batch normalization
    (Ioffe, S. and Szegedy, C., Batch normalization: Accelerating deep network training by reducing internal covariate shift,
     arXiv preprint arXiv:1502.03167, 2015)
    
    - Add tf.control_dependencies to the optimizer.
    """
    def __init__(self, momentum=0.99, epsilon=1e-5, center=True):
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        
    def __call__(self, inputs, is_training, name='batch_norm'):
        """
        Parameters
        inputs: [N, H, W, C]
        is_training: training mode check
        """
        with tf.variable_scope(name):
            return tf.layers.batch_normalization(inputs, momentum=self.momentum, epsilon=self.epsilon, 
                                                 center=self.center, training=is_training)

class Conv2D:
    """
    (standard) 2-D convolution
    """
    def __init__(self, FH=3, FW=3, weight_decay_lambda=None, truncated=False, stddev=0.02):
        """
        Parameters
        FH, FW: (int) filter height, filter width
        """
        self.FH, self.FW = FH, FW       
        self.weight_decay_lambda = weight_decay_lambda
        self.truncated = truncated
        self.stddev = stddev
        
    def __call__(self, inputs, FN, s=1, name='conv2d', padding='SAME', bias=False):
        """
        Parameters
        inputs: [N, H, W, C]
        FN: (int) filter number
        s: (int) convolutional stride (sdy=sdx=s)
        
        - filters: [FH, FW, C, FN]
        - outputs: [N, OH, OW, FN]
        """
        with tf.variable_scope(name):
            sdy, sdx = s, s
            C = inputs.get_shape()[-1]
            initializer = tf.truncated_normal_initializer(stddev=self.stddev) if self.truncated else tf.random_normal_initializer(stddev=self.stddev)
                 
            if not self.weight_decay_lambda:
                w = tf.get_variable(name='weight', shape=[self.FH, self.FW, C, FN], dtype=tf.float32, initializer=initializer)
            else:
                w = tf.get_variable(name='weight', shape=[self.FH, self.FW, C, FN], dtype=tf.float32, initializer=initializer,
                                    regularizer=tf.contrib.layers.l2_regularizer(scale=self.weight_decay_lambda))
                    
            conv = tf.nn.conv2d(inputs, w, strides=[1, sdy, sdx, 1], padding=padding)
            if not bias:
                return conv
            else:
                b = tf.get_variable(name='bias', shape=[FN], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
                conv_ = tf.nn.bias_add(conv, b)
                return conv_

class AConv2D:
    """
    Atrous 2-D convolution
    """
    def __init__(self, FH=3, FW=3, weight_decay_lambda=None, truncated=False, stddev=0.02):
        """
        Parameters
        FH, FW: (int) filter height, filter width
        """
        self.FH, self.FW = FH, FW
        self.weight_decay_lambda = weight_decay_lambda
        self.truncated = truncated
        self.stddev = stddev
        
    def __call__(self, inputs, FN, rate=1, name='atrous_conv2d', padding='SAME', bias=False):
        """
        Parameters
        inputs: [N, H, W, C]
        FN: (int) filter number
        rate: (int) dilation rate
        
        - filters: [FH, FW, C, FN]
        - outputs: [N, OH, OW, FN]
        - Strides are always 1.
        - If rate = 1, it performs standard 2-D convolution.
        """
        with tf.variable_scope(name):
            C = inputs.get_shape()[-1]
            initializer = tf.truncated_normal_initializer(stddev=self.stddev) if self.truncated else tf.random_normal_initializer(stddev=self.stddev)
                 
            if not self.weight_decay_lambda:
                w = tf.get_variable(name='weight', shape=[self.FH, self.FW, C, FN], dtype=tf.float32, initializer=initializer)
            else:
                w = tf.get_variable(name='weight', shape=[self.FH, self.FW, C, FN], dtype=tf.float32, initializer=initializer,
                                    regularizer=tf.contrib.layers.l2_regularizer(scale=self.weight_decay_lambda))
                    
            conv = tf.nn.atrous_conv2d(inputs, w, rate, padding)
            if not bias:
                return conv
            else:
                b = tf.get_variable(name='bias', shape=[FN], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
                conv_ = tf.nn.bias_add(conv, b)
                return conv_
           
class SepConv2D:
    """
    2-D atrous sepearable convolution (Depthwise + Atrous + BN + Pointwise)
    """
    def __init__(self, FH=3, FW=3, CM=1, weight_decay_lambda=None, truncated=False, stddev=0.02):
        """
        Parameters
        FH, FW: (int) filter height and filter width in depthwise conv
        CM: (int) channel multiplier in depthwise conv
        """
        self.FH, self.FW = FH, FW
        self.CM = CM
        self.weight_decay_lambda = weight_decay_lambda
        self.truncated = truncated
        self.stddev = stddev
        self.bn = BN()
        
    def __call__(self, inputs, FN, s=1, rate=None, is_BN=True, is_training=True, name='sep_conv2d', padding='SAME', bias=False):
        """
        Parameters
        inputs: [N, H, W, C]
        FN: (int) filter number in pointwise conv
        s: (int) stride in depthwise conv
        rate: (int) dilation rate (assume rate=rH=rW). If None, no atrous convolution.
        
        - Depthwise conv.: [N, H, W, C] x [FH, FW, C, CM] -> [N, OH, OW, C*CM]
        - Pointwise conv.: [N, OH, OW, C*CM] x [1, 1, C*CM, FN] -> [N, OH, OW, FN]
        """
        with tf.variable_scope(name):
            sdy, sdx = s, s
            C = inputs.get_shape()[-1]
            initializer = tf.truncated_normal_initializer(stddev=self.stddev) if self.truncated else tf.random_normal_initializer(stddev=self.stddev)
                
            if not self.weight_decay_lambda:
                w1 = tf.get_variable(name='depthwise_weight', shape=[self.FH, self.FW, C, self.CM], dtype=tf.float32, initializer=initializer)
                w2 = tf.get_variable(name='pointwise_weight', shape=[1, 1, int(C*self.CM), FN], dtype=tf.float32, initializer=initializer)
            else:
                w1 = tf.get_variable(name='depthwise_weight', shape=[self.FH, self.FW, C, self.CM], dtype=tf.float32, initializer=initializer,
                                     regularizer=tf.contrib.layers.l2_regularizer(scale=self.weight_decay_lambda))
                w2 = tf.get_variable(name='pointwise_weight', shape=[1, 1, int(C*self.CM), FN], dtype=tf.float32, initializer=initializer,
                                     regularizer=tf.contrib.layers.l2_regularizer(scale=self.weight_decay_lambda))
            h = []
            if rate is None:
                h.append(tf.nn.depthwise_conv2d(inputs, w1, strides=[1, sdy, sdx, 1], padding=padding, rate=rate))
                # depthwise convolution (no atrous)
            elif rate == 1:
                h.append(tf.nn.depthwise_conv2d(inputs, w1, strides=[1, sdy, sdx, 1], padding=padding, rate=[rate, rate]))
                # depthwise convolution (no atrous)
            else: # rate > 1
                h.append(tf.nn.depthwise_conv2d(inputs, w1, strides=[1, 1, 1, 1], padding=padding, rate=[rate, rate]))
                # atrous depthwise convolution
                # parameter 'rate': 1-D of size 2 ([rH, rW]; dilation rates in H and W dimensions)
                # in atrous conv., all the strides should be 1.
            if bias:
                b1 = tf.get_variable(name='depthwise_bias', shape=[C], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
                h.append(tf.reshape(tf.nn.bias_add(h[-1], b1), [-1, h[-1].get_shape()[1], h[-1].get_shape()[2], h[-1].get_shape()[3]]))
                
            if is_BN:
                h.append(self.bn(h[-1], is_training, 'bn'))
            
            h.append(tf.nn.conv2d(h[-1], w2, strides=[1, 1, 1, 1], padding='SAME')) # pointwise convolution
            if not bias:
                return h[-1]
            else:
                b2 = tf.get_variable(name='pointwise_bias', shape=[FN], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
                h.append(tf.reshape(tf.nn.bias_add(h[-1], b2), [-1, h[-1].get_shape()[1], h[-1].get_shape()[2], h[-1].get_shape()[3]]))
                return h[-1]  

class Block:
    """
    'Modified Aligned Xception' block used in DeepLabv3+
    """
    def __init__(self, weight_decay_lambda=None, truncated=False, stddev=0.02):
        self.weight_decay_lambda = weight_decay_lambda
        self.truncated = truncated
        self.stddev = stddev
        self.sepconv2d = SepConv2D(3, 3, 1, self.weight_decay_lambda, self.truncated, self.stddev)
        self.bn = BN()
        self.conv2d = Conv2D(1, 1, self.weight_decay_lambda, self.truncated, self.stddev)
        
    def __call__(self, inputs, FN1, FN2, FN3, first_relu=True, downsampling=False, rates=None, 
                 is_training=True, is_res_conv=True, name='block'):
        """
        Parameters
        inputs: [N, H, W, C]
        FN%d: (int) output channel number (filter number) in each inner conv.
        downsampling: (bool) whether reduction in spatial dimensions ([H, W]) is on.
        rates: (int) dilation rates in three different seperable convs., shape: [r1, r2, r3] (assume r1=rH1=rW1, r2=rH2=rW2, r3=rH3=rW3).
               If None, no atrous convolution.
        """
        with tf.variable_scope(name):
            if rates is None:
                rates = [None, None, None]
    
            h = [inputs]
            if first_relu:
                h.append(tf.nn.relu(h[-1]))
    
            # first inner conv.
            h.append(self.sepconv2d(h[-1], FN1, 1, rate=rates[0], is_training=is_training, name='conv0'))
            h.append(tf.nn.relu(self.bn(h[-1], is_training, 'bn0')))
            
            # second inner conv.
            h.append(self.sepconv2d(h[-1], FN2, 1, rate=rates[1], is_training=is_training, name='conv1'))
            h.append(tf.nn.relu(self.bn(h[-1], is_training, 'bn1')))
            
            # third inner conv.
            if not downsampling:
                h.append(self.sepconv2d(h[-1], FN3, 1, rate=rates[2], is_training=is_training, name='conv2'))
                # residual network
                if is_res_conv:
                    res = self.bn(self.conv2d(inputs, FN3, 1, 'residual'), is_training, 'residual_bn')
                else: # no convolution in residual connection (just add)
                    res = inputs
                
            else: # downsampling by 2 / if strides > 1, atrous cannot be applied.
                h.append(self.sepconv2d(h[-1], FN3, 2, rate=None, is_training=is_training, name='conv2'))
                # residual network
                if is_res_conv:
                    res = self.bn(self.conv2d(inputs, FN3, 2, 'residual'), is_training, 'residual_bn')
                else: 
                    res = inputs
                
            h.append(self.bn(h[-1], is_training, 'bn2'))
            return h[-1] + res