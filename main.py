# -*- coding: utf-8 -*-
"""
@author: shoh4486
tf.__version__ == '1.12.0' ~ '1.14.0'
"""
import tensorflow as tf
import numpy as np
import os
import pprint
from model import DeepLabv3plus
from utils import *

flags = tf.app.flags
flags.DEFINE_integer('trial_num', 1, 'trial number')
flags.DEFINE_integer('C_in', 3, 'the number of input channels')
flags.DEFINE_integer('num_class', 4, 'the number of classes')
flags.DEFINE_bool('separable', True, 'applying separable convoluion')
flags.DEFINE_integer('seed', 1, 'seed number')
flags.DEFINE_float('weight_decay_lambda', 1e-07, 'L2 weight decay lambda')
flags.DEFINE_bool('truncated', False, 'truncated weight distribution')
flags.DEFINE_string('optimizer', 'Adam', 'optimizer')
flags.DEFINE_integer('gpu_num', 2, 'the number of GPUs')
flags.DEFINE_integer('H_train', 300, 'height dimension while training (fixed)')
flags.DEFINE_integer('W_train', 400, 'width dimension while training (fixed)')
flags.DEFINE_integer('output_stride_training', 16, 'output stride in the training mode')
flags.DEFINE_boolean('random_scaling_keep_aspect_ratio', True, 'keep aspect ratio when rescaling augmentation')
flags.DEFINE_boolean('bn_training', True, 'training the BN parameters while training')
flags.DEFINE_boolean('random_brightness_contrast', False, 'applying random brightness and random contrast')
flags.DEFINE_float('lr_init', 1e-03, 'initial learning rate')
flags.DEFINE_bool('lr_decay', True, 'learning rate decay')
flags.DEFINE_integer('n_aug', 200, 'the number of augmentations')
flags.DEFINE_integer('batch_size_training', 6, 'batch size')
flags.DEFINE_integer('check_epoch', 1, 'check epoch')
flags.DEFINE_integer('start_epoch', 0, 'start epoch') 
flags.DEFINE_integer('end_epoch', 50, 'end epoch')
flags.DEFINE_boolean('train', True, 'True for training, False for evaluation')
flags.DEFINE_boolean('restore', False, 'True for retoring, False for raw training')
flags.DEFINE_integer('restore_trial_num', 1, 'directory number of pretrained model')
flags.DEFINE_integer('restore_sess_num', 49, 'sess number of pretrained model')
flags.DEFINE_integer('restart_epoch', 50, 'train restart epoch') 
flags.DEFINE_integer('re_end_epoch', 200, 'train re-end epoch')
flags.DEFINE_boolean('eval_with_test_acc', True, 'True for test accuracies evaluation')
flags.DEFINE_integer('output_stride_testing', 8, 'output stride in the training mode')
FLAGS = flags.FLAGS

def main(_):
    flags.DEFINE_string('save_dir', os.path.join("./trials", "trial_{0}".format(FLAGS.trial_num)), 'output saving directory')    
    pprint.pprint(flags.FLAGS.__flags)
    
    mkdir(FLAGS.save_dir)
    mkdir(os.path.join(FLAGS.save_dir, "test"))
    
    if FLAGS.gpu_num: # gpu_num >= 1
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        sess = tf.Session(config=run_config)
    else: # only cpu
        sess = tf.Session()
    
    deeplabv3plus = DeepLabv3plus(
                                  sess=sess,
                                  C_in=FLAGS.C_in,
                                  num_class=FLAGS.num_class,
                                  separable_aspp_decoder=FLAGS.separable,
                                  seed=FLAGS.seed,
                                  weight_decay_lambda=FLAGS.weight_decay_lambda,
                                  truncated=FLAGS.truncated,
                                  optimizer=FLAGS.optimizer,
                                  save_dir=FLAGS.save_dir,
                                  gpu_num=FLAGS.gpu_num
                                  )
    
    global_variables_list()
    
    if FLAGS.train:
        from data.data_preprocessing import inputs_train, inputs_train_, inputs_valid, gts_train, gts_train_, gts_valid
        inputs_col = [inputs_train, inputs_train_, inputs_valid]
        for i in inputs_col:
            inputs_pixel_checker(i)
            
        gts_col = [set(gts_train.flatten()), set(gts_train_.flatten()), set(gts_valid.flatten())]
        for i in gts_col:
            gts_pixel_checker(i)
        
        if FLAGS.restore:
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join("./trials", "trial_{0}".format(FLAGS.restore_trial_num), "sess-{0}".format(FLAGS.restore_sess_num)))
            FLAGS.start_epoch = FLAGS.restart_epoch
            FLAGS.end_epoch = FLAGS.re_end_epoch
            deeplabv3plus.train(
                                inputs=(inputs_train, inputs_train_, inputs_valid),
                                gts=(gts_train, gts_train_, gts_valid),
                                config=FLAGS
                                )
            saver = tf.train.Saver()
            saver.save(sess, os.path.join(FLAGS.save_dir, "sess"), global_step=FLAGS.end_epoch-1)
        
        else:  
            deeplabv3plus.train(
                                inputs=(inputs_train, inputs_train_, inputs_valid),
                                gts=(gts_train, gts_train_, gts_valid),
                                config=FLAGS
                                )
            saver = tf.train.Saver()
            saver.save(sess, os.path.join(FLAGS.save_dir, "sess"), global_step=FLAGS.end_epoch-1)
        
        np.savetxt(os.path.join(FLAGS.save_dir, "CEE_train.txt"), deeplabv3plus.CEE_train_vals)
        np.savetxt(os.path.join(FLAGS.save_dir, "miou_train.txt"), deeplabv3plus.miou_train_vals)
        np.savetxt(os.path.join(FLAGS.save_dir, "PA_ALL_train.txt"), deeplabv3plus.PA_ALL_train_vals)
        
        np.savetxt(os.path.join(FLAGS.save_dir, "CEE_valid.txt"), deeplabv3plus.CEE_valid_vals)
        np.savetxt(os.path.join(FLAGS.save_dir, "miou_valid.txt"), deeplabv3plus.miou_valid_vals)
        np.savetxt(os.path.join(FLAGS.save_dir, "PA_ALL_valid.txt"), deeplabv3plus.PA_ALL_valid_vals)
        
    else: # testing mode
        try:
            from data.test_data_preprocessing import inputs_test, gts_test  
        except ImportError: # when gts_test is not given
            from data.test_data_preprocessing import inputs_test
        else:
            gts_pixel_checker(set(gts_test.flatten()))
        finally:
            inputs_pixel_checker(inputs_test)
        
        if FLAGS.restore:
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join("./trials", "trial_{0}".format(FLAGS.restore_trial_num), "sess-{0}".format(FLAGS.restore_sess_num)))
            if FLAGS.eval_with_test_acc:
                test_results = deeplabv3plus.evaluation(
                                                        inputs=inputs_test,
                                                        output_stride=FLAGS.output_stride_testing,
                                                        gts=gts_test
                                                        )
                seg_test, CEE_test, miou_test, PA_ALL_test = test_results
                np.savetxt(os.path.join(FLAGS.save_dir, "test", "CEE_test.txt"), CEE_test)
                np.savetxt(os.path.join(FLAGS.save_dir, "test", "miou_test.txt"), miou_test)
                np.savetxt(os.path.join(FLAGS.save_dir, "test", "PA_ALL_test.txt"), PA_ALL_test)
            
            else:
                seg_test = deeplabv3plus.evaluation(
                                                    inputs=inputs_test,
                                                    output_stride=FLAGS.output_stride_testing,
                                                    gts=None
                                                    )
            for i in range(len(seg_test)):
                for c in range(FLAGS.num_class):
                    np.savetxt(os.path.join(FLAGS.save_dir, "test", "test_result%d_class%d.txt" % (i, c)), seg_test[i, :, :, c])
        else:
            raise NotImplementedError('pretrained session must be restored.')
            
if __name__ == '__main__':
    tf.app.run()