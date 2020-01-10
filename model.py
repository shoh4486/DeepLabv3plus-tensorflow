# -*- coding: utf-8 -*-
"""
@author: shoh4486
tf.__version__ == '1.12.0' ~ '1.14.0'
"""
import tensorflow as tf
import numpy as np
import time
from ops import *
from utils import *

class Xception:
    """
    Modified aligned Xception
    """
    def __init__(self, is_training, weight_decay_lambda=None):
        self.is_training = is_training
        self.weight_decay_lambda = weight_decay_lambda
        self.conv2d = Conv2D(3, 3, self.weight_decay_lambda)
        self.bn = BN()
        self.block = Block(self.weight_decay_lambda)
        self.sepconv2d = SepConv2D(3, 3, 1, self.weight_decay_lambda)
        
    def entry_flow(self, inputs, downsampling):
        """
        Entry flow
        """
        h = []
        h.append(tf.nn.relu(self.bn(self.conv2d(inputs, 32, 2, 'entry0'), 
                                    self.is_training, 'entry0_bn')))
    
        h.append(tf.nn.relu(self.bn(self.conv2d(h[-1], 64, 1, 'entry1'), 
                                    self.is_training, 'entry1_bn')))
        
        h.append(self.block(h[-1], 128, 128, 128, first_relu=False, downsampling=True, 
                            is_training=self.is_training, name='entry_block0'))
        c128_feature = tf.nn.relu(h[-1])
        
        h.append(self.block(h[-1], 256, 256, 256, downsampling=True, 
                            is_training=self.is_training, name='entry_block1'))

        h.append(self.block(h[-1], 728, 728, 728, downsampling=downsampling, 
                            is_training=self.is_training, name='entry_block2'))
        return c128_feature, h[-1]  
    
    def middle_flow(self, inputs, md):
        """
        Middle flow
        
        Parameters
        inputs: entry_flow's return
        """
        h = [inputs]
        for _ in range(16):
            h.append(self.block(h[-1], 728, 728, 728, rates=md, 
                                is_training=self.is_training, is_res_conv=False, 
                                name='middle_block%d' % (len(h)-1)))
        return h[-1]

    def exit_flow(self, inputs, ed):
        """
        Exit flow
        
        Parameters
        inputs: middle_flow's return
        """
        h = [inputs]
        h.append(self.block(h[-1], 728, 1024, 1024, rates=[ed[0], ed[0], None],
                            is_training=self.is_training, name='exit_block0'))
        
        h.append(self.bn(self.sepconv2d(tf.nn.relu(h[-1]), 1536, 1, rate=ed[1], 
                                        is_training=self.is_training, name='exit0'), 
                         self.is_training, 'exit0_bn'))
        
        h.append(self.bn(self.sepconv2d(tf.nn.relu(h[-1]), 1536, 1, rate=ed[1], 
                                        is_training=self.is_training, name='exit1'), 
                         self.is_training, 'exit1_bn'))
        
        h.append(self.bn(self.sepconv2d(tf.nn.relu(h[-1]), 2048, 1, rate=ed[1], 
                                        is_training=self.is_training, name='exit2'), 
                         self.is_training, 'exit2_bn'))

        h.append(tf.nn.relu(h[-1]))
        return h[-1]
    
    def forward(self, inputs, output_stride):
        if output_stride == 16:
            downsampling = True 
            md = [1, 1, 1] # md: middle flow's dilation rate
            ed = [1, 2] # ed: exit flow's dilation rate
            
        elif output_stride == 8:
            downsampling = False
            md = [2, 2, 2]
            ed = [2, 4]
            
        c128_feature, x = self.entry_flow(inputs, downsampling)
        x = self.middle_flow(x, md)
        x = self.exit_flow(x, ed)
        return c128_feature, x
    
class ASPP:
    """
    Atrous Spatial Pyramid Pooling
    (Here, ASSPP (Atrous Separable Spatial Pyramid Pooling) is also supported.)
    """
    def __init__(self, H, W, is_training, separable=True, drop_rate=0.5, 
                 weight_decay_lambda=None):
        """
        Parameters
        H, W: training image size
        seperable: if True, ASSPP, else, ASPP.
        drop_rate: the probability that each element of x is discarded. 
                   It should be set by a placeholder (0.0 in the test phase).
        """
        self.H, self.W = H, W
        self.is_training = is_training
        self.separable = separable
        self.drop_rate = drop_rate
        self.weight_decay_lambda = weight_decay_lambda  
        self.conv2d = Conv2D(1, 1, self.weight_decay_lambda)
        self.bn = BN()
        self.block = Block(self.weight_decay_lambda)
        self.sepconv2d = SepConv2D(3, 3, 1, self.weight_decay_lambda)
        self.aconv2d0 = AConv2D(1, 1, self.weight_decay_lambda)
        self.aconv2d1 = AConv2D(3, 3, self.weight_decay_lambda)

    def global_average_pooling_2d(self, inputs, H, W):
        """
        Global average pooling 2-D
        
        Parameters
        inputs: Xception's output
        H, W: height and width that will be upscaled bilinearly
        """
        h = []
        # global average pooling 2-D
        h.append(tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)) 
        # 1x1 conv.
        h.append(tf.nn.relu(self.bn(self.conv2d(h[-1], 256, 1, 'global_average_pooling_2d'), 
                                    self.is_training, 'global_average_pooling_2d_bn')))
        # bilinear upsampling
        h.append(tf.image.resize_bilinear(h[-1], size=[H, W], align_corners=True))
        return h[-1] 
        
    def forward(self, inputs, output_stride):
        """
        Parameters       
        inputs: Xception's output
        """
        if output_stride == 16: 
            d = [1, 6, 12, 18]
            H_encoded = tf.cast(tf.ceil(self.H/16), tf.int32)
            W_encoded = tf.cast(tf.ceil(self.W/16), tf.int32)
        
        elif output_stride == 8: 
            d = [1, 12, 24, 36]
            H_encoded = tf.cast(tf.ceil(self.H/8), tf.int32)
            W_encoded = tf.cast(tf.ceil(self.W/8), tf.int32)
            
        if self.separable: # ASSPP
            h0 = tf.nn.relu(self.bn(self.aconv2d0(inputs, 256, rate=d[0], 
                                                  name='aspp0'), 
                                    self.is_training, 'aspp0_bn'))
            
            h1 = tf.nn.relu(self.bn(self.sepconv2d(inputs, 256, 1, rate=d[1], 
                                                   is_training=self.is_training, 
                                                   name='aspp1'), 
                                    self.is_training, 'aspp1_bn'))
            
            h2 = tf.nn.relu(self.bn(self.sepconv2d(inputs, 256, 1, rate=d[2], 
                                                   is_training=self.is_training, 
                                                   name='aspp2'), 
                                    self.is_training, 'aspp2_bn'))
            
            h3 = tf.nn.relu(self.bn(self.sepconv2d(inputs, 256, 1, rate=d[3], 
                                                   is_training=self.is_training, 
                                                   name='aspp3'), 
                                    self.is_training, 'aspp3_bn'))
            
            h4 = self.global_average_pooling_2d(inputs, H=H_encoded, W=W_encoded)
            
            h = tf.concat([h0, h1, h2, h3, h4], axis=-1) 
            h = tf.nn.relu(self.bn(self.conv2d(h, 256, 1, 'last_encoding'), 
                                   self.is_training, 'last_encoding_bn'))
            #if self.drop_rate: return tf.nn.dropout(h, rate=self.drop_rate)
            #else: return h
            return tf.nn.dropout(h, keep_prob=1.0-self.drop_rate) 
            # tf.nn.dropout(h, rate=self.drop_rate)
            
        else: # ASPP
            h0 = tf.nn.relu(self.bn(self.aconv2d0(inputs, 256, rate=d[0], 
                                                  name='aspp0'), 
                                    self.is_training, 'aspp0_bn'))
            
            h1 = tf.nn.relu(self.bn(self.aconv2d1(inputs, 256, rate=d[1], 
                                                  name='aspp1'), 
                                    self.is_training, 'aspp1_bn'))
            
            h2 = tf.nn.relu(self.bn(self.aconv2d1(inputs, 256, rate=d[2], 
                                                  name='aspp2'), 
                                    self.is_training, 'aspp2_bn'))
            
            h3 = tf.nn.relu(self.bn(self.aconv2d1(inputs, 256, rate=d[3], 
                                                  name='aspp3'), 
                                    self.is_training, 'aspp3_bn'))
            
            h4 = self.global_average_pooling_2d(inputs, H=H_encoded, W=W_encoded)
            
            h = tf.concat([h0, h1, h2, h3, h4], axis=-1)
            h = tf.nn.relu(self.bn(self.conv2d(h, 256, 1, 'last_encoding'), 
                                   self.is_training, 'last_encoding_bn'))
            #if self.drop_rate: return tf.nn.dropout(h, rate=self.drop_rate)
            #else: return h
            return tf.nn.dropout(h, keep_prob=1.0-self.drop_rate) 
            # tf.nn.dropout(h, rate=self.drop_rate)
        
class Decoder:
    """
    U-Net style decoder
    """
    def __init__(self, H, W, num_class, is_training, separable=True, 
                 drop_rate1=0.5, drop_rate2=0.1, weight_decay_lambda=None):
        """
        Parameters
        H, W: training image size
        num_class: the number of segmentation classes
        seperable: if True, seperable conv. will be applied.
        drop_rate: the probability that each element of x is discarded. 
                   It should be set by a placeholder (0.0 in the test phase).
        """
        self.H, self.W = H, W
        self.num_class = num_class
        self.is_training = is_training
        self.separable = separable
        self.drop_rate1, self.drop_rate2 = drop_rate1, drop_rate2
        self.weight_decay_lambda = weight_decay_lambda
        self.conv2d0 = Conv2D(1, 1, self.weight_decay_lambda)
        self.conv2d1 = Conv2D(3, 3, self.weight_decay_lambda)
        self.bn = BN()
        self.sepconv2d = SepConv2D(3, 3, 1, self.weight_decay_lambda)
        
    def forward(self, c128_feature, inputs):
        """
        Parameters
        c128_feature: comes from Xception-Entry flow
        inputs: ASPP's output
        """
        if self.separable:
            h = []
            h.append(tf.nn.relu(self.bn(self.conv2d0(c128_feature, 48, 1, 'decoder0'), 
                                        self.is_training, 'decoder0_bn')))
            
            h.append(tf.image.resize_bilinear(inputs, size=[tf.cast(tf.ceil(self.H/4), tf.int32), 
                                                            tf.cast(tf.ceil(self.W/4), tf.int32)], 
                                              align_corners=True))
            h.append(tf.concat([h[-2], h[-1]], axis=-1))
            
            h.append(tf.nn.relu(self.bn(self.sepconv2d(h[-1], 256, 1, is_training=self.is_training, 
                                                       name='decoder1'), 
                                        self.is_training, 'decoder1_bn')))
            #h.append(tf.nn.dropout(h[-1], drop_rate=self.drop_rate1))
            h.append(tf.nn.dropout(h[-1], keep_prob=1.0-self.drop_rate1))
            
            h.append(tf.nn.relu(self.bn(self.sepconv2d(h[-1], 256, 1, is_training=self.is_training, 
                                                       name='decoder2'), 
                                        self.is_training, 'decoder2_bn')))
            #h.append(tf.nn.dropout(h[-1], drop_rate=self.drop_rate2))
            h.append(tf.nn.dropout(h[-1], keep_prob=1.0-self.drop_rate2))
            
            h.append(self.conv2d0(h[-1], self.num_class, 1, 'last_conv', bias=True))
            
            h.append(tf.image.resize_bilinear(h[-1], size=[self.H, self.W], align_corners=True))
            return h[-1]
        
        else:
            h = []
            h.append(tf.nn.relu(self.bn(self.conv2d0(c128_feature, 48, 1, 'decoder0'), 
                                        self.is_training, 'decoder0_bn')))
            
            h.append(tf.image.resize_bilinear(inputs, size=[tf.cast(tf.ceil(self.H/4), tf.int32), 
                                                            tf.cast(tf.ceil(self.W/4), tf.int32)], 
                                              align_corners=True))
            h.append(tf.concat([h[-2], h[-1]], axis=-1))
            
            h.append(tf.nn.relu(self.bn(self.conv2d1(h[-1], 256, 1, 'decoder1'), 
                                        self.is_training, 'decoder1_bn')))
            #h.append(tf.nn.dropout(h[-1], drop_rate=self.drop_rate1))
            h.append(tf.nn.dropout(h[-1], keep_prob=1.0-self.drop_rate1))
            
            h.append(tf.nn.relu(self.bn(self.conv2d1(h[-1], 256, 1, 'decoder2'), 
                                        self.is_training, 'decoder2_bn')))
            #h.append(tf.nn.dropout(h[-1], drop_rate=self.drop_rate2))
            h.append(tf.nn.dropout(h[-1], keep_prob=1.0-self.drop_rate2))
            
            h.append(self.conv2d0(h[-1], self.num_class, 1, 'last_conv', bias=True))
            
            h.append(tf.image.resize_bilinear(h[-1], size=[self.H, self.W], align_corners=True))
            return h[-1]
        
class DeepLabv3plus:
    """
    DeepLabv3+ by Chen, L.-C. et al., Encoder-Decoder with atrous separable 
    convolution for semantic image segmentation,
    Proceedings of the European conference on computer vision (ECCV), 2018.

    - Only Xception applied as a backbone structure of the encoder (no DeepLabv3 as the backbone).
    """
    def __init__(self, sess, C_in, num_class, separable_aspp_decoder, seed, 
                 weight_decay_lambda=None, optimizer='Adam', gpu_alloc=[0]):
        """
        Parameters
        sess: TensorFlow sesson
        C_in: (int) the number of input channels
        num_class: (int) the number of segmentation classes
        separable_aspp_decoder: (bool) if True, separable conv. is applied to 
                                both aspp and decoder modules.
        seed: (int) random seed for random modules in numpy and TensorFlow
        weight_decay_lambda: (float) L2 weight decay lambda (0.0: do not employ)
        optimizer: (str) only Adam adopted
        gpu_alloc: (list) specifying which GPU(s) to be used; [] if to use only cpu
        """
        self.sess = sess
        self.C_in = C_in
        self.num_class = num_class       
        self.separable_aspp_decoder = separable_aspp_decoder
        self.seed = seed
        self.weight_decay_lambda = weight_decay_lambda
        self.optimizer = optimizer
        self._beta1 = 0.9 # beta1 in Adam optimizer
        self.gpu_alloc = gpu_alloc
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        self.build_model()

    def build_model(self):
        with tf.name_scope('placeholders'):
            with tf.name_scope('inputs'):
                self.inputs = tf.placeholder(tf.float32, shape=(None, None, None, self.C_in), name='inputs')

            with tf.name_scope('ground_truths'):
                self.gts = tf.placeholder(tf.float32, shape=(None, None, None, self.num_class), name='ground_truths')
                # class dimension: last dimension
                
            with tf.name_scope('inputs_size'):
                self.H = tf.placeholder(tf.int32, shape=None, name='height')
                self.W = tf.placeholder(tf.int32, shape=None, name='width')
                
            with tf.name_scope('is_training'):
                self.is_training = tf.placeholder(tf.bool, shape=None, name='is_training')

            with tf.name_scope('drop_rates'): # The probability that each element of x is discarded.
                self.drop_rate0 = tf.placeholder(tf.float32, shape=None, name='drop_rate0') # -> ASPP
                self.drop_rate1 = tf.placeholder(tf.float32, shape=None, name='drop_rate1') # -> Decoder
                self.drop_rate2 = tf.placeholder(tf.float32, shape=None, name='drop_rate2') # -> Decoder

            with tf.name_scope('learning_rate'):
                self.lr = tf.placeholder(tf.float32, shape=None, name='learning_rate')
        
        with tf.name_scope('instantiations'):
            xception = Xception(self.is_training, self.weight_decay_lambda) 
            aspp = ASPP(self.H, self.W, self.is_training, 
                        self.separable_aspp_decoder, self.drop_rate0, 
                        self.weight_decay_lambda)
            decoder = Decoder(self.H, self.W, self.num_class, self.is_training, 
                              self.separable_aspp_decoder, self.drop_rate1, 
                              self.drop_rate2, self.weight_decay_lambda)
            
        with tf.variable_scope('Xception') as xception_scope:
            if len(self.gpu_alloc) == 2:
                with tf.device('/device:GPU:1'):
                    c128_feature0, x0 = xception.forward(self.inputs, output_stride=16)
                    xception_scope.reuse_variables() 
                    c128_feature1, x1 = xception.forward(self.inputs, output_stride=8)
            else:
                c128_feature0, x0 = xception.forward(self.inputs, output_stride=16)
                xception_scope.reuse_variables() 
                c128_feature1, x1 = xception.forward(self.inputs, output_stride=8)

        with tf.variable_scope('ASPP') as aspp_scope: 
            aspp_x0 = aspp.forward(x0, output_stride=16)
            aspp_scope.reuse_variables()
            aspp_x1 = aspp.forward(x1, output_stride=8)
            
        with tf.variable_scope('Decoder') as decoder_scope:
            # when output_stride=16
            de_aspp_x0 = decoder.forward(c128_feature0, aspp_x0)
            self.hardmax_de_aspp_x0 = tf.contrib.seq2seq.hardmax(de_aspp_x0) # used in a test phase
            
            decoder_scope.reuse_variables()
            # when output_stride=8
            de_aspp_x1 = decoder.forward(c128_feature1, aspp_x1)
            self.hardmax_de_aspp_x1 = tf.contrib.seq2seq.hardmax(de_aspp_x1)
            
        with tf.name_scope('loss'):
            softmax_cee0 = tf.losses.softmax_cross_entropy(onehot_labels=self.gts, 
                                                           logits=de_aspp_x0) 
            # output_stride=16
            softmax_cee1 = tf.losses.softmax_cross_entropy(onehot_labels=self.gts, 
                                                           logits=de_aspp_x1) 
            # output_stride=8
            
            if self.weight_decay_lambda:
                weight_decay_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                weight_decay_list = [var for var in weight_decay_vars]
                softmax_cee0 += tf.add_n(weight_decay_list)
                softmax_cee1 += tf.add_n(weight_decay_list)

        with tf.name_scope('optimizer'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if self.optimizer == 'Adam':
                    self.train_step0 = tf.train.AdamOptimizer(learning_rate=self.lr, 
                                                              beta1=self._beta1).minimize(softmax_cee0) 
                    # output_stride=16       
                    self.train_step1 = tf.train.AdamOptimizer(learning_rate=self.lr, 
                                                              beta1=self._beta1).minimize(softmax_cee1) 
                    # output_stride=8
                else:
                    raise NotImplementedError('Other optimizers have not been considered.')
        
        with tf.name_scope('performance_measures'):
            with tf.name_scope('error'): # as a loss checker / not used in the training phase
                self.softmax_cee0_ = tf.losses.softmax_cross_entropy(onehot_labels=self.gts, 
                                                                     logits=de_aspp_x0) 
                # output_stride=16
                self.softmax_cee1_ = tf.losses.softmax_cross_entropy(onehot_labels=self.gts, 
                                                                     logits=de_aspp_x1) 
                # output_stride=8
                
            with tf.name_scope('accuracy'):
                with tf.name_scope('mIOU'):
                    self.miou0 = miou_tf(pred=self.hardmax_de_aspp_x0, gt=self.gts) 
                    # a scalar, output_stride=16
                    self.miou1 = miou_tf(pred=self.hardmax_de_aspp_x1, gt=self.gts) 
                    # a scalar, output_stride=8
                    
                with tf.name_scope('percent_accuracy'):
                    self.PA_C0 = pixel_acc_4D_tf(pred=self.hardmax_de_aspp_x0, 
                                                 gt=self.gts, 
                                                 return_axis='C') 
                    # [C, 1] # output_stride=16
                    self.PA_C1 = pixel_acc_4D_tf(pred=self.hardmax_de_aspp_x1, 
                                                 gt=self.gts, 
                                                 return_axis='C') 
                    # [C, 1] # output_stride=8
                    self.PA_ALL0 = pixel_acc_4D_tf(pred=self.hardmax_de_aspp_x0, 
                                                   gt=self.gts, 
                                                   return_axis='ALL') 
                    # a scalar, output_stride=16
                    self.PA_ALL1 = pixel_acc_4D_tf(pred=self.hardmax_de_aspp_x1, 
                                                   gt=self.gts, 
                                                   return_axis='ALL') 
                    # a scalar, output_stride=8
        
        tf.summary.image('Xception_entry_flow', tf.slice(c128_feature0, begin=[0, 0, 0, 0], 
                                                         size=[4, tf.cast(tf.ceil(self.H/4), tf.int32), 
                                                               tf.cast(tf.ceil(self.W/4), tf.int32), 1]),
                         max_outputs=4)
        
        tf.summary.image('Xception_result', tf.slice(x0, begin=[0, 0, 0, 0], 
                                                     size=[4, tf.cast(tf.ceil(self.H/16), tf.int32), 
                                                           tf.cast(tf.ceil(self.W/16), tf.int32), 1]), 
                         max_outputs=4)
        
        tf.summary.image('ASPP_result', tf.slice(aspp_x0, begin=[0, 0, 0, 0], 
                                          size=[4, tf.cast(tf.ceil(self.H/16), tf.int32), 
                                                tf.cast(tf.ceil(self.W/16), tf.int32), 1]), 
                         max_outputs=4)
        
        gamma_var = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) \
                     if 'gamma' in var.name]
        for gv in range(len(gamma_var)):
            tf.summary.histogram('gamma_var_%d' % gv, gamma_var[gv])
                
        beta_var = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) \
                    if 'beta' in var.name]
        for bv in range(len(beta_var)):
            tf.summary.histogram('beta_var_%d' % bv, beta_var[bv])
    
        #moving_var = [var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) \
        #              if 'moving_' in var.name]
        #for mv in range(len(moving_var)):
        #    tf.summary.histogram('moving_var_%d' % mv, moving_var[mv])
            
        weight_var = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) \
                      if 'weight' in var.name]
        for wv in range(len(weight_var)):
            tf.summary.histogram('weight_var_%d' % wv, weight_var[wv])
        
        bias_var = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) \
                    if 'bias' in var.name]
        for biv in range(len(bias_var)):
            tf.summary.histogram('bias_var_%d' % biv, bias_var[biv])
    
    def grid(self, H, W, H_, W_, random_pick=False, seed=0):
        """
        Returns list of coordinates of top-left corner for random cropping
        
        Parameters
        H, W: input image
        H_, W_: cropping size (H_, W_ < H, W)
        random_pick: if True, randomly return just one coordinate.
        """
        h = list(range(H - H_ + 1)) # list of possible top-left coordinate (height)
        w = list(range(W - W_ + 1)) # list of possible top-left coordinate (width)
        grid = np.meshgrid(h, w)
        grid_list = [(i, j) for (i, j) in zip(grid[0].flatten(), grid[1].flatten())]
        if not random_pick:
            return grid_list
        else:
            return grid_list[np.random.RandomState(seed=seed).choice(len(grid_list), 1)[0]] 
            # randomly pick one integer among 0~len(grid_list)-1 and return a tuple
            
    def random_brightness_contrast(self, tmp_data):
        """
        This method can be ignored (in a segmentation task, this can lower the model performance).
        """
        tmp = []
        tmp.append(tf.image.random_brightness(tmp_data, max_delta=12.75/255)) # change max_delta manually
        tmp.append(tf.image.random_contrast(tmp[-1], lower=0.75, upper=1.25))
        tmp.append(tf.image.per_image_standardization(tmp[-1]))
        return tf.reshape(tmp[-1], [1, tf.shape(tmp_data)[0], tf.shape(tmp_data)[1], -1])
            
    def train(self, inputs, gts, config): 
        """
        Parameters
        inputs: a tuple consisting of (inputs_train, inputs_train_, inputs_valid) ([N, H, W, C]) (0~1)
        gts: a tuple consisting of (gts_train, gts_train_, gts_valid) ([N, H, W, num_class]) (0 or 1)
        config: configuration defined by tf.app.flags
        
        xxx_train: training data
        xxx_train_: to measure the training loss, acc
        xxx_valid: to measure the validation loss, acc
        """     
        inputs_train, inputs_train_, inputs_valid = inputs # unpacking
        gts_train, gts_train_, gts_valid = gts
                
        H_orig, W_orig = inputs_train.shape[1], inputs_train.shape[2] # original image size
        H_train, W_train = config.H_train, config.W_train 
        # training size (fixed)
        # in a training phase, random cropping and random scaling were employed and 
        # then resized to the fixed training image size (H_train, W_train)
        
        if config.start_epoch == 0:
            self.sess.run(tf.global_variables_initializer())
        
        merge = tf.summary.merge_all()
        writer = tf.summary.FileWriter(config.save_dir, self.sess.graph)
        
        self.CEE_train_vals, self.miou_train_vals, self.PA_ALL_train_vals = [], [], []
        self.CEE_valid_vals, self.miou_valid_vals, self.PA_ALL_valid_vals = [], [], []
        
        ########################################
        if config.output_stride_training == 16:
            train_step = self.train_step0
            hardmax_de_aspp_x = self.hardmax_de_aspp_x0
            softmax_cee = self.softmax_cee0_
            miou = self.miou0
            PA_ALL = self.PA_ALL0
            
        elif config.output_stride_training == 8:
            train_step = self.train_step1
            hardmax_de_aspp_x = self.hardmax_de_aspp_x1
            softmax_cee = self.softmax_cee1_
            miou = self.miou1
            PA_ALL = self.PA_ALL1
            
        else:
            raise NotImplementedError('output_stride_training should be either 16 or 8.')
        ########################################
        
        if not config.lr_decay:
            lr_tmp = config.lr_init
            
        total_train_num = int(inputs_train.shape[0]*config.n_aug)
        iters_per_epoch = int(total_train_num/config.batch_size_training)
        
        tmp_data = tf.placeholder(tf.float32, shape=(1, None, None, None)) # one by one
        tmp_nn = tf.image.resize_images(tmp_data, size=[H_train, W_train], 
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, 
                                        align_corners=True, preserve_aspect_ratio=True)
        tmp_data2 = tf.placeholder(tf.float32, shape=(H_train, W_train, None)) # one by one
        tmp_random = self.random_brightness_contrast(tmp_data2)
        
        ######################################## list1, len: total_train_num
        train_index_list = np.tile(list(range(inputs_train.shape[0])), config.n_aug).tolist() 
        ########################################
        
        ######################################## list2 for random scaling
        if config.random_scaling_keep_aspect_ratio:
            gcd = np.gcd(H_orig, W_orig) # greatest common divisor
            H_list = list(range(H_train, H_orig + 1, int(H_orig/gcd))) 
            # height list for random scaling / integers between H_train and H_orig
            W_list = list(range(W_train, W_orig + 1, int(W_orig/gcd))) 
            # width list for random scaling / integers between W_train and W_orig / len(W_list) = len(H_list)
            HW_list = list(zip(H_list, W_list))
            # a list containing (height, width) as elements
        else:
            H_list = list(range(H_train, H_orig + 1, 1))
            W_list = list(range(W_train, W_orig + 1, 1))
            HW_list = list(zip(H_list, W_list))
        ########################################
        
        for epoch in range(config.start_epoch, config.end_epoch):
            t1 = time.time()
            k = np.random.RandomState(seed=epoch).choice(len(H_list), total_train_num, 
                                      replace=True)
            HW_list_ = [HW_list[i] for i in k] # list2, len: total_train_num
            
            ######################################## list3 for random cropping, len: total_train_num
            hw_list = [] # top-left corner coordinates
            for kk in range(len(HW_list_)):
                hw_list.append(self.grid(H_orig, W_orig, HW_list_[kk][0], 
                                         HW_list_[kk][1], random_pick=True, seed=epoch))
            ########################################
            
            ######################################## list4 for random flipping, len: total_train_num
            random_flip_list = np.random.RandomState(seed=epoch).choice([True, False], 
                                                    total_train_num, replace=True).tolist()
            ########################################
            
            batch_number = np.random.RandomState(seed=epoch).choice(total_train_num, 
                                                (iters_per_epoch, config.batch_size_training), 
                                                replace=False) # index for list1~4
            t2 = time.time() # check data-preprocessing time / t2-t1 ~ 20 s
            
            ############### 1 epoch ###############
            for i, batch in enumerate(batch_number):
                tmp0, tmp1 = [], []
                t3 = time.time()
                for j in range(config.batch_size_training):
                    rH = HW_list_[batch[j]][0] # random scaling
                    rW = HW_list_[batch[j]][1]
                    
                    rh = hw_list[batch[j]][0] # random crop
                    rw = hw_list[batch[j]][1]
                
                    tmp_inputs = inputs_train[train_index_list[batch[j]], 
                                              rh:rh+rH, rw:rw+rW, :].reshape(1, rH, rW, -1)
                    tmp_gts = gts_train[train_index_list[batch[j]], 
                                        rh:rh+rH, rw:rw+rW, :].reshape(1, rH, rW, -1)
                    
                    if random_flip_list[batch[j]]: # random flip
                        tmp_inputs = np.flip(tmp_inputs, axis=2)
                        tmp_gts = np.flip(tmp_gts, axis=2)
                    
                    tmp0.append(self.sess.run(tmp_nn, feed_dict={tmp_data: tmp_inputs})) 
                    # resizing to [H_train, W_train]
                    tmp1.append(self.sess.run(tmp_nn, feed_dict={tmp_data: tmp_gts}))

                inputs_batch = np.concatenate(tuple(tmp0), axis=0).reshape(config.batch_size_training, 
                                             H_train, W_train, -1)
                gts_batch = np.concatenate(tuple(tmp1), axis=0).reshape(config.batch_size_training, 
                                          H_train, W_train, -1)
                
                if config.random_brightness_contrast:
                    tmp2 = []
                    for image in inputs_batch:
                        tmp2.append(self.sess.run(tmp_random, feed_dict={tmp_data2: image})) 
                        # random brightness and random contrast   
                    inputs_batch = np.concatenate(tuple(tmp2), axis=0).reshape(config.batch_size_training, 
                                                 H_train, W_train, -1)
                t4 = time.time() # t4-t3 << 1 s
                
                if config.lr_decay: # dependent on end_epoch
                    lr_tmp = lr_polynomial_decay(
                            config.lr_init, 
                            global_step=(epoch - config.start_epoch)*iters_per_epoch + i, # real-time iteration number
                            decay_steps=(config.end_epoch - config.start_epoch)*iters_per_epoch, # the number of entire iterations
                            end_lr=0.0, 
                            power=0.9
                            )
                
                self.sess.run(train_step, feed_dict={self.inputs: inputs_batch, 
                                                     self.gts: gts_batch, 
                                                     self.H: H_train, 
                                                     self.W: W_train,
                                                     self.is_training: config.bn_training,
                                                     self.drop_rate0: 0.5, 
                                                     self.drop_rate1: 0.5, 
                                                     self.drop_rate2: 0.1,
                                                     self.lr: lr_tmp})
            #######################################
            
            if epoch % config.check_epoch == 0:
                self.seg_train, CEE_train_val, miou_train_val, PA_ALL_train_val, summary_train \
                = self.sess.run([hardmax_de_aspp_x, softmax_cee, miou, PA_ALL, merge],
                                feed_dict={self.inputs: inputs_train_, 
                                           self.gts: gts_train_, 
                                           self.H: H_train, 
                                           self.W: W_train,
                                           self.is_training: False,
                                           self.drop_rate0: 0.0, 
                                           self.drop_rate1: 0.0, 
                                           self.drop_rate2: 0.0})
    
                self.CEE_train_vals.append(CEE_train_val)
                self.miou_train_vals.append(miou_train_val)
                self.PA_ALL_train_vals.append(PA_ALL_train_val)
                writer.add_summary(summary_train, epoch)
                
                self.seg_valid, CEE_valid_val, miou_valid_val, PA_ALL_valid_val \
                = self.sess.run([hardmax_de_aspp_x, softmax_cee, miou, PA_ALL], 
                                feed_dict={self.inputs: inputs_valid, 
                                           self.gts: gts_valid, 
                                           self.H: H_train, 
                                           self.W: W_train,
                                           self.is_training: False,
                                           self.drop_rate0: 0.0, 
                                           self.drop_rate1: 0.0, 
                                           self.drop_rate2: 0.0})
    
                self.CEE_valid_vals.append(CEE_valid_val)
                self.miou_valid_vals.append(miou_valid_val)
                self.PA_ALL_valid_vals.append(PA_ALL_valid_val)
                    
                print('Epoch: %d, lr: %f, dt: (%f, %f), CEE_train: %f, \
                      miou_train: %f, PA_train: %f, CEE_valid: %f, miou_valid: %f, \
                      PA_valid: %f' \
                      % (epoch, lr_tmp, t2-t1, t4-t3, CEE_train_val, miou_train_val, 
                         PA_ALL_train_val, CEE_valid_val, miou_valid_val, PA_ALL_valid_val))

    def evaluation(self, inputs, output_stride, gts=None):
        """
        Test set evaluation after the training and the validation
        
        Parameters
        inputs: input images ([N, H, W, C]) (0~1)
        output_stride: (int) 16 or 8
        gts: (optional) ground truths ([N, H, W, num_class]) (0 or 1)
        """
        assert output_stride == 16 or output_stride == 8
        if output_stride == 16:
            if gts is None:
                return self.sess.run(self.hardmax_de_aspp_x0, 
                                     feed_dict={self.inputs: inputs,
                                                self.H: inputs.shape[1], 
                                                self.W: inputs.shape[2],
                                                self.is_training: False,
                                                self.drop_rate0: 0.0, 
                                                self.drop_rate1: 0.0, 
                                                self.drop_rate2: 0.0})
            
            else: # gts is given
                return self.sess.run([self.hardmax_de_aspp_x0, self.softmax_cee0_, 
                                      self.miou0, self.PA_ALL0], 
                                     feed_dict={self.inputs: inputs, 
                                                self.gts: gts,
                                                self.H: inputs.shape[1], 
                                                self.W: inputs.shape[2],
                                                self.is_training: False,
                                                self.drop_rate0: 0.0, 
                                                self.drop_rate1: 0.0, 
                                                self.drop_rate2: 0.0})
        
        elif output_stride == 8:
            if gts is None:
                return self.sess.run(self.hardmax_de_aspp_x1, 
                                     feed_dict={self.inputs: inputs,
                                                self.H: inputs.shape[1], 
                                                self.W: inputs.shape[2],
                                                self.is_training: False,
                                                self.drop_rate0: 0.0, 
                                                self.drop_rate1: 0.0, 
                                                self.drop_rate2: 0.0})
            
            else:
                return self.sess.run([self.hardmax_de_aspp_x1, self.softmax_cee1_, 
                                      self.miou1, self.PA_ALL1], 
                                     feed_dict={self.inputs: inputs, 
                                                self.gts: gts,
                                                self.H: inputs.shape[1], 
                                                self.W: inputs.shape[2],
                                                self.is_training: False,
                                                self.drop_rate0: 0.0, 
                                                self.drop_rate1: 0.0, 
                                                self.drop_rate2: 0.0})