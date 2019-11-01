# -*- coding: utf-8 -*-
"""
@author: shoh4486
"""
import os
import tensorflow as tf

def mkdir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)     
    except OSError:
        print('Cannot make the directory "{0}"'.format(directory))

def global_variables_list():
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

def pixel_checker(image_set):
    """
    image_set: set(image)
    """
    if (0 in image_set) and (1 in image_set) and (len(image_set) == 2):
        pass
    else:
        raise ValueError('Label map should include only 0 and 1.')