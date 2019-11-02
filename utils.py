# -*- coding: utf-8 -*-
"""
@author: shoh4486
"""
import os
import tensorflow as tf
import numpy as np

def mkdir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)     
    except OSError:
        print('Cannot make the directory "{0}"'.format(directory))

def global_variables_list():
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

def inputs_pixel_checker(image):
    if (np.min(image) < 0) or (np.max(image) > 1):
        raise ValueError('Input pixel values should be in 0~1 range.')
    else:
        pass
        
def gts_pixel_checker(image_set):
    """
    image_set: set(image)
    """
    if (0 in image_set) and (1 in image_set) and (len(image_set) == 2):
        pass
    else:
        raise ValueError('Label maps should include only 0 and 1.')